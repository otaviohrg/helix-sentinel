"""
Fault injection wrapper for ArmEnv.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from .arm_env import ArmEnv


class FaultType(str, Enum):
    NONE = "none"
    JOINT_STICTION = "joint_stiction"  # mechanical
    ENCODER_DRIFT = "encoder_drift"  # sensor
    VELOCITY_NOISE = "velocity_noise"  # sensor
    MOTOR_SATURATION = "motor_saturation"  # mechanical


FAULT_LABEL = {
    FaultType.NONE: 0,
    FaultType.JOINT_STICTION: 1,
    FaultType.ENCODER_DRIFT: 2,
    FaultType.VELOCITY_NOISE: 3,
    FaultType.MOTOR_SATURATION: 4,
}


@dataclass
class FaultConfig:
    fault_type: FaultType = FaultType.NONE
    affected_joint: int = 0
    onset_timestep: int = 50
    drift_rate: float = 0.01
    saturation_scale: float = 3.0
    noise_scale: float = 0.5
    saturation_limit: float = 0.3


class FaultInjector:
    """
    Wraps ArmEnv and injects a fault at onset_timestep.

    Usage:
        config = FaultConfig(
            fault_type=FaultType.ENCODER_DRIFT,
            affected_joint=2,
            onset_timestep=50,
        )
        env = FaultInjector(config)
        obs = env.reset()
        for t in range(200):
            obs = env.step(t)
            # obs has the fault injected after t >= 50
    """

    def __init__(self, config: FaultConfig) -> None:
        self.env = ArmEnv()
        self.config = config
        self._drift_accum = 0.0
        self._stuck_pos = None
        self._rng = np.random.default_rng()

    def reset(self) -> np.ndarray:
        self._drift_accum = 0.0
        return self.env.reset()

    def step(self, timestep: int) -> np.ndarray:
        obs = self.env.step().copy()

        if timestep < self.config.onset_timestep:
            return obs

        j = self.config.affected_joint
        fault = self.config.fault_type

        if fault == FaultType.JOINT_STICTION:
            # Capture stuck position on first fault timestep
            if self._stuck_pos is None:
                self._stuck_pos = obs["positions"][j]
            obs["positions"][j] = self._stuck_pos
            obs["velocities"][j] = obs["velocities"][j] * 0.02

        elif fault == FaultType.ENCODER_DRIFT:
            # Encoder reading drifts linearly — position corrupted
            self._drift_accum += self.config.drift_rate
            obs["positions"][j] += self._drift_accum

        elif fault == FaultType.VELOCITY_NOISE:
            # Velocity sensor fault — Gaussian noise added
            obs["velocities"][j] += self._rng.normal(0, self.config.noise_scale)

        elif fault == FaultType.MOTOR_SATURATION:
            # Mechanical — velocity physically clipped
            obs["velocities"][j] = np.clip(
                obs["velocities"][j],
                -self.config.saturation_limit,
                self.config.saturation_limit,
            )

        return obs
