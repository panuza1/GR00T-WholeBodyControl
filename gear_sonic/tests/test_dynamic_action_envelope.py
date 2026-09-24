from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from gear_sonic.envs.env_utils.g1_action_envelope import (
    CONTROL_DT,
    DYNAMIC_MARGIN_RATIO,
    G1_ACTION_SCALE,
    G1_DEFAULT_ANGLES,
    G1_JOINT_LOWER,
    G1_JOINT_NAMES,
    G1_JOINT_UPPER,
    MAX_TARGET_ACCELERATION,
    MAX_TARGET_VELOCITY,
    dynamically_feasible_targets,
    tensors_for_joint_names,
)


def _state(dtype=torch.float64):
    names = list(G1_JOINT_NAMES)
    lower, upper, offset, scale = tensors_for_joint_names(names, device="cpu", dtype=dtype)
    return (
        lower.unsqueeze(0),
        upper.unsqueeze(0),
        offset.unsqueeze(0),
        scale.unsqueeze(0),
        offset.unsqueeze(0).clone(),
        torch.zeros((1, 29), dtype=dtype),
    )


def test_generated_targets_obey_position_velocity_and_acceleration_limits():
    lower, upper, offset, scale, target, velocity = _state()
    generator = torch.Generator().manual_seed(7)
    effective_velocity = MAX_TARGET_VELOCITY * (1.0 - DYNAMIC_MARGIN_RATIO)
    effective_acceleration = MAX_TARGET_ACCELERATION * (1.0 - DYNAMIC_MARGIN_RATIO)

    for _ in range(10_000):
        raw = torch.randn((1, 29), generator=generator, dtype=torch.float64) * 20.0
        next_target, next_velocity = dynamically_feasible_targets(
            raw, target, velocity, offset, scale, lower, upper
        )
        acceleration = (next_velocity - velocity) / CONTROL_DT
        assert torch.all(next_target > lower)
        assert torch.all(next_target < upper)
        assert torch.max(torch.abs(next_velocity)) <= effective_velocity + 1.0e-10
        assert torch.max(torch.abs(acceleration)) <= effective_acceleration + 1.0e-8
        target, velocity = next_target, next_velocity


def test_asymmetric_ranges_and_valid_zero_action():
    lower, upper, offset, scale, target, velocity = _state()
    zero_target, zero_velocity = dynamically_feasible_targets(
        torch.zeros_like(target), target, velocity, offset, scale, lower, upper
    )
    torch.testing.assert_close(zero_target, offset, atol=0.0, rtol=0.0)
    torch.testing.assert_close(zero_velocity, torch.zeros_like(velocity), atol=0.0, rtol=0.0)

    ankle = G1_JOINT_NAMES.index("left_ankle_pitch_joint")
    positive = torch.zeros_like(target)
    positive[0, ankle] = 100.0
    for _ in range(200):
        target, velocity = dynamically_feasible_targets(
            positive, target, velocity, offset, scale, lower, upper
        )
    positive_target = target[0, ankle].item()

    negative = torch.zeros_like(target)
    negative[0, ankle] = -100.0
    for _ in range(400):
        target, velocity = dynamically_feasible_targets(
            negative, target, velocity, offset, scale, lower, upper
        )
    negative_target = target[0, ankle].item()
    assert positive_target < G1_JOINT_UPPER[ankle]
    assert negative_target > G1_JOINT_LOWER[ankle]
    assert positive_target > 0.0 and negative_target < -0.5


def test_named_map_rejects_missing_duplicate_or_unknown_joints():
    bad = list(G1_JOINT_NAMES)
    bad[-1] = bad[0]
    with pytest.raises(ValueError, match="exactly once"):
        tensors_for_joint_names(bad, device="cpu", dtype=torch.float64)


def test_nonfinite_policy_action_is_rejected():
    lower, upper, offset, scale, target, velocity = _state()
    raw = torch.zeros_like(target)
    raw[0, 4] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        dynamically_feasible_targets(raw, target, velocity, offset, scale, lower, upper)


def test_training_and_deployment_transforms_match(tmp_path: Path):
    """Compile the deployment header and compare a deterministic action sequence."""
    repo = Path(__file__).resolve().parents[2]
    include = repo / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include"
    source = tmp_path / "parity.cpp"
    binary = tmp_path / "parity"
    source.write_text(
        """
#include \"dynamic_action_envelope.hpp\"
#include <array>
#include <iomanip>
#include <iostream>
int main() {
  DynamicActionEnvelope envelope;
  std::array<double, G1_NUM_MOTOR> raw{};
  for (int step = 0; step < 6; ++step) {
    for (int i = 0; i < G1_NUM_MOTOR; ++i) raw[i] = (step - 2.0) * (i + 1.0) / 7.0;
    const auto target = envelope.Transform(raw);
    std::cout << std::setprecision(17);
    for (double value : target) std::cout << value << ' ';
    std::cout << '\\n';
  }
}
"""
    )
    subprocess.run(
        ["c++", "-std=c++20", "-O0", "-fno-fast-math", f"-I{include}", str(source), "-o", str(binary)],
        check=True,
    )
    cpp = torch.tensor(
        [[float(value) for value in line.split()] for line in subprocess.check_output([binary], text=True).splitlines()],
        dtype=torch.float64,
    )

    lower, upper, offset, scale, target, velocity = _state()
    python = []
    for step in range(6):
        raw = torch.tensor(
            [[(step - 2.0) * (i + 1.0) / 7.0 for i in range(29)]], dtype=torch.float64
        )
        target, velocity = dynamically_feasible_targets(
            raw, target, velocity, offset, scale, lower, upper
        )
        python.append(target.squeeze(0).clone())
    torch.testing.assert_close(torch.stack(python), cpp, rtol=1.0e-6, atol=1.0e-7)


def test_authoritative_constants_match_deployment_header():
    header = (
        Path(__file__).resolve().parents[2]
        / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp"
    ).read_text()
    for name in G1_JOINT_NAMES:
        assert f'"{name}"' in header
    assert len(G1_JOINT_LOWER) == len(G1_JOINT_UPPER) == len(G1_ACTION_SCALE) == 29
