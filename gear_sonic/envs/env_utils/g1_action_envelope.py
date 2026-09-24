"""Shared SONIC G1 action definition used by training-side tests.

The policy still emits the released residual-action coordinates.  This module
maps those coordinates smoothly into the joint-position, target-velocity, and
target-acceleration envelope enforced by deployment.
"""

from __future__ import annotations

import math

import torch


G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

G1_JOINT_LOWER = (
    -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.618, -0.52, -0.52, -3.0892, -1.5882, -2.618, -1.0472,
    -1.97222, -1.61443, -1.61443, -3.0892, -2.2515, -2.618,
    -1.0472, -1.97222, -1.61443, -1.61443,
)
G1_JOINT_UPPER = (
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
    2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
    2.618, 0.52, 0.52, 2.6704, 2.2515, 2.618, 2.0944,
    1.97222, 1.61443, 1.61443, 2.6704, 1.5882, 2.618,
    2.0944, 1.97222, 1.61443, 1.61443,
)
G1_DEFAULT_ANGLES = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
)

_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_W = 10.0 * 2.0 * math.pi


def _scale(effort: float, armature: float) -> float:
    return 0.25 * effort / (armature * _W * _W)


_S5020 = _scale(25.0, _ARMATURE_5020)
_S7520_14 = _scale(88.0, _ARMATURE_7520_14)
_S7520_22 = _scale(139.0, _ARMATURE_7520_22)
_S4010 = _scale(5.0, _ARMATURE_4010)
G1_ACTION_SCALE = (
    _S7520_22, _S7520_22, _S7520_14, _S7520_22, _S5020, _S5020,
    _S7520_22, _S7520_22, _S7520_14, _S7520_22, _S5020, _S5020,
    _S7520_14, _S5020, _S5020, _S5020, _S5020, _S5020, _S5020,
    _S5020, _S4010, _S4010, _S5020, _S5020, _S5020, _S5020,
    _S5020, _S4010, _S4010,
)

CONTROL_DT = 0.02
POSITION_MARGIN = 0.002
MAX_TARGET_VELOCITY = 8.0
MAX_TARGET_ACCELERATION = 80.0
DYNAMIC_MARGIN_RATIO = 0.05


def tensors_for_joint_names(names: list[str], *, device, dtype) -> tuple[torch.Tensor, ...]:
    """Return authoritative limits/defaults/scales in an action term's named order."""
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(G1_JOINT_NAMES):
        raise ValueError("G1 action term must resolve each authoritative 29-DOF joint exactly once")
    indices = [G1_JOINT_NAMES.index(name) for name in names]
    values = (G1_JOINT_LOWER, G1_JOINT_UPPER, G1_DEFAULT_ANGLES, G1_ACTION_SCALE)
    return tuple(torch.tensor([value[i] for i in indices], device=device, dtype=dtype) for value in values)


def dynamically_feasible_targets(
    raw_action: torch.Tensor,
    previous_target: torch.Tensor,
    previous_velocity: torch.Tensor,
    offset: torch.Tensor,
    scale: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    dt: float = CONTROL_DT,
    position_margin: float = POSITION_MARGIN,
    max_velocity: float = MAX_TARGET_VELOCITY,
    max_acceleration: float = MAX_TARGET_ACCELERATION,
    dynamic_margin_ratio: float = DYNAMIC_MARGIN_RATIO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smoothly map residual actions to one-step and braking-feasible targets."""
    if dt <= 0.0 or position_margin < 0.0 or not 0.0 <= dynamic_margin_ratio < 1.0:
        raise ValueError("invalid action-envelope parameters")
    if raw_action.shape != previous_target.shape or raw_action.shape != previous_velocity.shape:
        raise ValueError("raw action and action-envelope state shapes must match")
    if not torch.isfinite(raw_action).all():
        raise ValueError("policy action contains a non-finite value")

    low = lower + position_margin
    high = upper - position_margin
    if torch.any(offset <= low) or torch.any(offset >= high):
        raise ValueError("action offset must be strictly inside the margined joint limits")

    positive_range = high - offset
    negative_range = offset - low
    scaled = raw_action * scale
    desired_target = offset + torch.where(
        raw_action >= 0.0,
        positive_range * torch.tanh(scaled / positive_range),
        negative_range * torch.tanh(scaled / negative_range),
    )

    acceleration = max_acceleration * (1.0 - dynamic_margin_ratio)
    velocity = max_velocity * (1.0 - dynamic_margin_ratio)
    upper_distance = torch.clamp(high - previous_target, min=0.0)
    lower_distance = torch.clamp(previous_target - low, min=0.0)
    adt = acceleration * dt
    max_upper_speed = -adt + torch.sqrt(adt * adt + 2.0 * acceleration * upper_distance)
    max_lower_speed = -adt + torch.sqrt(adt * adt + 2.0 * acceleration * lower_distance)

    velocity_low = torch.maximum(
        torch.maximum(previous_velocity - adt, torch.full_like(previous_velocity, -velocity)),
        -max_lower_speed,
    )
    velocity_high = torch.minimum(
        torch.minimum(previous_velocity + adt, torch.full_like(previous_velocity, velocity)),
        max_upper_speed,
    )
    if torch.any(velocity_low > velocity_high):
        raise ValueError("previous action state has no dynamically feasible successor")

    desired_velocity = (desired_target - previous_target) / dt
    midpoint = 0.5 * (velocity_low + velocity_high)
    half_range = 0.5 * (velocity_high - velocity_low)
    safe_half_range = torch.clamp(half_range, min=torch.finfo(raw_action.dtype).eps)
    next_velocity = midpoint + half_range * torch.tanh((desired_velocity - midpoint) / safe_half_range)
    next_target = previous_target + next_velocity * dt
    return next_target, next_velocity
