"""
Minimal robosuite tabletop scene for VLA demo.

Panda robot arm + table + movable objects.
No LIBERO / BDDL dependency — just MuJoCo + robosuite.
"""

import numpy as np
import robosuite as suite


def create_scene(render_resolution: int = 512) -> tuple:
    """
    Create a simple tabletop scene with Panda arm and objects.

    Returns (env, task_description) tuple.
    task_description is a placeholder — actual instruction comes from the browser.
    """
    from robosuite.controllers import load_part_controller_config
    ctrl = load_part_controller_config(default_controller="OSC_POSE")

    config = {
        "env_name": "Lift",
        "robots": "Panda",
        "controller_configs": ctrl,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "render_camera": "agentview",
        "camera_heights": render_resolution,
        "camera_widths": render_resolution,
        "use_object_obs": False,
        "use_camera_obs": True,
        "control_freq": 20,
        "horizon": 100000,
        "reward_shaping": False,
        "ignore_done": True,  # never auto-terminate
    }

    env = suite.make(**config)
    return env


def get_image(env) -> np.ndarray:
    """Get RGB rendering from the agentview camera, rotated 180°."""
    img = env.sim.render(
        camera_name="agentview",
        width=512,
        height=512,
    )
    return img[::-1, ::-1]  # rotate 180° to match LIBERO convention


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle (copied from robosuite)."""
    import math
    quat = np.array(quat)
    if quat[3] > 1.0:
        quat = quat / np.linalg.norm(quat)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_state(env) -> np.ndarray:
    """Extract robot end-effector pose + gripper state (7D)."""
    obs = env._get_observations()
    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    gripper = obs["robot0_gripper_qpos"]
    return np.concatenate([eef_pos, _quat2axisangle(eef_quat), gripper])
