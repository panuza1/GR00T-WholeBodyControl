from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
import torch

from gear_sonic.envs.env_utils.g1_action_envelope import (
    CONTROL_DT,
    DYNAMIC_MARGIN_RATIO,
    MAX_TARGET_ACCELERATION,
    MAX_TARGET_VELOCITY,
    POSITION_MARGIN,
    dynamically_feasible_targets,
    tensors_for_joint_names,
)

# Joint ordering constants
G1_MUJOCO_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = None


class DynamicallyFeasibleJointPositionAction(JointPositionAction):
    """SONIC residual action with the same smooth safety envelope as deployment."""

    cfg: DynamicallyFeasibleJointPositionActionCfg

    def __init__(self, cfg: DynamicallyFeasibleJointPositionActionCfg, env):
        super().__init__(cfg, env)
        if abs(env.step_dt - cfg.control_dt) > 1.0e-9:
            raise ValueError(f"action control_dt={cfg.control_dt} does not match env step_dt={env.step_dt}")
        lower, upper, _, _ = tensors_for_joint_names(
            self._joint_names, device=self.device, dtype=self._raw_actions.dtype
        )
        self._lower = lower.unsqueeze(0)
        self._upper = upper.unsqueeze(0)
        self._previous_target = self._offset.clone()
        self._target_velocity = torch.zeros_like(self._raw_actions)
        self._target_acceleration = torch.zeros_like(self._raw_actions)

    @property
    def target_velocity(self) -> torch.Tensor:
        return self._target_velocity

    @property
    def target_acceleration(self) -> torch.Tensor:
        return self._target_acceleration

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        target, velocity = dynamically_feasible_targets(
            actions,
            self._previous_target,
            self._target_velocity,
            self._offset,
            self._scale,
            self._lower,
            self._upper,
            dt=self.cfg.control_dt,
            position_margin=self.cfg.position_margin,
            max_velocity=self.cfg.max_velocity,
            max_acceleration=self.cfg.max_acceleration,
            dynamic_margin_ratio=self.cfg.dynamic_margin_ratio,
        )
        self._target_acceleration[:] = (velocity - self._target_velocity) / self.cfg.control_dt
        self._target_velocity[:] = velocity
        self._previous_target[:] = target
        self._processed_actions[:] = target

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._previous_target[env_ids] = self._offset[env_ids]
        self._processed_actions[env_ids] = self._offset[env_ids]
        self._target_velocity[env_ids] = 0.0
        self._target_acceleration[env_ids] = 0.0


@configclass
class DynamicallyFeasibleJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = DynamicallyFeasibleJointPositionAction
    control_dt: float = CONTROL_DT
    position_margin: float = POSITION_MARGIN
    max_velocity: float = MAX_TARGET_VELOCITY
    max_acceleration: float = MAX_TARGET_ACCELERATION
    dynamic_margin_ratio: float = DYNAMIC_MARGIN_RATIO
