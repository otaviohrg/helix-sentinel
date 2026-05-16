import numpy as np
from sentinel.sim.fault_injector import (
    FaultConfig,
    FaultInjector,
    FaultType,
    FAULT_LABEL,
)
from sentinel.sim.arm_env import EPISODE_LENGTH, N_JOINTS


def collect(fault_type, joint=0, onset=50) -> np.ndarray:
    """Run one episode and return flat observations (EPISODE_LENGTH, OBS_DIM)."""
    config = FaultConfig(
        fault_type=fault_type, affected_joint=joint, onset_timestep=onset
    )
    env = FaultInjector(config)
    env.reset()
    episodes = []
    for t in range(EPISODE_LENGTH):
        obs = env.step(t)
        episodes.append(np.concatenate([obs["positions"], obs["velocities"]]))
    return np.array(episodes, dtype=np.float32)


def test_normal_no_fault():
    """Normal episode should have smooth, bounded observations."""
    obs = collect(FaultType.NONE)
    print(obs)
    assert obs.shape == (EPISODE_LENGTH, N_JOINTS * 2)
    assert np.all(np.isfinite(obs))
    assert np.abs(obs).max() < 10.0


def test_stiction_freezes_position_and_velocity():
    """After onset, affected joint position freezes and velocity drops to near zero."""
    obs = collect(FaultType.JOINT_STICTION, joint=0, onset=50)
    post_pos = obs[55:, 0]  # joint 0 position after onset
    post_vel = obs[55:, N_JOINTS]  # joint 0 velocity after onset
    # Position should be nearly constant
    assert np.std(post_pos) < 0.01
    # Velocity should be near zero
    assert np.abs(post_vel).mean() < 0.05


def test_encoder_drift_monotonic():
    """After onset, affected joint position should drift monotonically."""
    obs = collect(FaultType.ENCODER_DRIFT, joint=0, onset=50)
    post_onset_pos = obs[51:, 0]
    diffs = np.diff(post_onset_pos)
    assert (diffs > 0).mean() > 0.7 or (diffs < 0).mean() > 0.7


def test_velocity_noise_increases_variance():
    """After onset, affected joint velocity variance should increase."""
    normal = collect(FaultType.NONE, joint=0)
    noisy = collect(FaultType.VELOCITY_NOISE, joint=0, onset=20)
    normal_std = np.std(normal[25:, N_JOINTS])
    noisy_std = np.std(noisy[25:, N_JOINTS])
    assert noisy_std > normal_std * 2


def test_motor_saturation_clips_velocity():
    """After onset, affected joint velocity should be clipped."""
    obs = collect(FaultType.MOTOR_SATURATION, joint=0, onset=20)
    post_vel = obs[25:, N_JOINTS]
    assert np.abs(post_vel).max() <= 0.5 + 1e-6  # within saturation limit


def test_fault_labels():
    """Label map should cover all fault types."""
    assert FAULT_LABEL[FaultType.NONE] == 0
    assert FAULT_LABEL[FaultType.JOINT_STICTION] == 1
    assert FAULT_LABEL[FaultType.ENCODER_DRIFT] == 2
    assert FAULT_LABEL[FaultType.VELOCITY_NOISE] == 3
    assert FAULT_LABEL[FaultType.MOTOR_SATURATION] == 4
    assert len(FAULT_LABEL) == 5


def test_all_fault_types_run():
    """All fault types should complete without error."""
    for ft in FaultType:
        obs = collect(ft)
        assert obs.shape == (EPISODE_LENGTH, N_JOINTS * 2)
        assert np.all(np.isfinite(obs))
