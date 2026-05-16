"""
6-DOF arm MuJoCo environment for fault detection data collection.
Reuses the arm_6dof.xml model from helix-core/projects/demos/.

Observation: [positions (6), velocities (6)] → shape (12,)
Control: sinusoidal per-joint, same as the MuJoCo bridge demo.
"""

import pathlib
import numpy as np
import mujoco

MODEL_PATH = pathlib.Path(__file__).parents[3] / "models" / "arm_6dof.xml"

N_JOINTS = 6
EPISODE_LENGTH = 200
PHYSICS_HZ = 500
PUBLISH_HZ = 50
STEPS_PER_TICK = PHYSICS_HZ // PUBLISH_HZ


class ArmEnv:
    """
    Minimal arm simulation environment.
    """

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.t = 0.0

        inertias = np.array(
            [self.model.body_inertia[i + 1, 0] for i in range(self.model.nu)]
        )
        inertias = np.clip(inertias, 1e-6, None)
        self._ctrl_scale = inertias / inertias.max()

    def _observe(self) -> dict:
        return {
            "positions": self.data.qpos[:N_JOINTS].copy(),
            "velocities": self.data.qvel[:N_JOINTS].copy(),
        }

    def _apply_control(self):
        for i in range(self.model.nu):
            phase = i * np.pi / 3.0
            self.data.ctrl[i] = np.sin(self.t + phase) * 0.5 * self._ctrl_scale[i]

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.t = 0.0
        return self._observe()

    def step(self) -> np.ndarray:
        self._apply_control()
        for _ in range(STEPS_PER_TICK):
            mujoco.mj_step(self.model, self.data)
        self.t += 1.0 / PUBLISH_HZ
        return self._observe()
