"""
Random tabletop scene with Panda arm — no BDDL, no LIBERO tasks.

Uses robosuite 1.4.1 + mujoco 3.3.5 directly.
Randomly places 2-5 objects (blocks, balls, cylinders) on a table.
"""

import numpy as np
import random
import math

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler


def create_scene(render_resolution: int = 512):
    """Create a random tabletop scene. Returns robosuite env."""
    ctrl = load_controller_config(default_controller="OSC_POSE")

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
        "ignore_done": True,
    }

    env = suite.make(**config)
    env.reset()
    return env


def get_image(env) -> np.ndarray:
    """RGB from agentview camera, rotated 180°."""
    return env.sim.render(camera_name="agentview", width=512, height=512)[::-1, ::-1]


def get_state(env) -> np.ndarray:
    """7D state: eef pos(3) + quat→axisangle(3) + gripper(1)."""
    obs = env._get_observations()
    pos = obs["robot0_eef_pos"]
    quat = obs["robot0_eef_quat"]
    q = np.array(quat)
    if q[3] > 1.0:
        q = q / np.linalg.norm(q)
    den = np.sqrt(1.0 - q[3] * q[3])
    ax = np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den
    return np.concatenate([pos, ax, obs["robot0_gripper_qpos"]])
