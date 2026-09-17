"""Headless MuJoCo checks; no SDK channel or DDS traffic is created."""

from types import SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.configs import BaseConfig


def test_g1_order_neutral_and_safety():
    config = BaseConfig(enable_onscreen=False).load_wbc_yaml()
    assert DefaultEnv({**config, "ENABLE_ELASTIC_BAND": False}, onscreen=False).elastic_band is None
    env = DefaultEnv(config, onscreen=False)
    model, data = env.mj_model, env.mj_data

    assert model.nu == 43 and model.nq == 50
    assert model.opt.timestep == 0.005
    assert np.allclose(model.opt.gravity, [0, 0, -9.81])
    assert np.allclose(data.qpos[:7], [0, 0, 0.793, 1, 0, 0, 0])
    assert np.allclose(data.qpos[env.body_qpos_index], config["DEFAULT_DOF_ANGLES"])
    assert np.allclose(env.prepare_obs()["body_q"], config["DEFAULT_DOF_ANGLES"])
    assert np.all(np.diff(env.body_qpos_index)[:21] == 1)
    assert env.body_qpos_index[22] > env.body_qpos_index[21] + 1  # left hand is interleaved
    for indices in (env.body_joint_index, env.left_hand_index, env.right_hand_index):
        for joint_id in indices:
            assert model.actuator_trnid[joint_id - 1, 0] == joint_id

    command = lambda q, kp, kd: SimpleNamespace(q=q, dq=0.0, tau=0.0, kp=kp, kd=kd)
    state_publishes = []
    env.unitree_bridge = SimpleNamespace(
        low_cmd_received=True,
        policy_started=True,
        low_cmd=SimpleNamespace(motor_cmd=[command(q, kp, kd) for q, kp, kd in zip(
            config["DEFAULT_MOTOR_ANGLES"], config["MOTOR_KP"], config["MOTOR_KD"]
        )]),
        left_hand_cmd=SimpleNamespace(motor_cmd=[command(0, 1, 0.1)] * 7),
        right_hand_cmd=SimpleNamespace(motor_cmd=[command(0, 1, 0.1)] * 7),
        num_body_motor=29, num_hand_motor=7, use_sensor=False, joystick=None,
        PublishLowState=lambda obs: state_publishes.append(obs),
    )
    env.sim_step()
    assert len(state_publishes) == 1 and not env.fall

    data.qpos[env.body_qpos_index[0]] = np.nan
    env.sim_step()
    assert env.fall and len(state_publishes) == 1
    assert np.isfinite(data.qpos).all()

    data.qpos[env.body_qpos_index[0]] = model.jnt_range[1, 1] + 0.1
    env.sim_step()
    assert env.fall and len(state_publishes) == 1
    assert np.allclose(data.qpos[env.body_qpos_index], config["DEFAULT_DOF_ANGLES"])

    data.qpos[2] = 0.1
    env.check_fall()
    assert env.fall and np.isclose(data.qpos[2], 0.793)

    env.unitree_bridge.low_cmd.motor_cmd[0].q = np.nan
    env.sim_step()
    assert env.fall and np.isfinite(data.qpos).all()

    env.unitree_bridge.low_cmd_received = False
    time_before = data.time
    env.sim_step()
    assert data.time == time_before

    env.unitree_bridge.low_cmd_received = True
    env.unitree_bridge.policy_started = False
    env.sim_step()
    assert data.time == time_before
