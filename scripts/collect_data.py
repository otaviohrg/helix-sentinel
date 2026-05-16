#!/usr/bin/env python3
"""
Collect 10,000 labelled episodes from the fault-injected arm sim.

Split: 8,000 train / 1,000 val / 1,000 test
Fault probability: 50% of episodes have a fault
One fault type and joint per episode, chosen randomly

Fault types:
  0 — none              (normal)
  1 — joint_stiction    (mechanical)
  2 — encoder_drift     (sensor)
  3 — velocity_noise    (sensor)
  4 — motor_saturation  (mechanical)

Output: data/episodes.h5
"""

import pathlib
import numpy as np
import h5py
from tqdm import tqdm

from sentinel.sim.arm_env import EPISODE_LENGTH, N_JOINTS
from sentinel.sim.fault_injector import (
    FaultConfig,
    FaultInjector,
    FaultType,
    FAULT_LABEL,
)

# ─── config ───────────────────────────────────────────────────────────────────

N_EPISODES = 10_000
FAULT_PROB = 0.5
OUTPUT_PATH = pathlib.Path("data/episodes.h5")
SEED = 42

FAULT_TYPES = [
    FaultType.JOINT_STICTION,
    FaultType.ENCODER_DRIFT,
    FaultType.VELOCITY_NOISE,
    FaultType.MOTOR_SATURATION,
]

# Observation is a dict {"positions": (6,), "velocities": (6,)}
# Flattened to [positions | velocities] → shape (12,)
OBS_DIM = N_JOINTS * 2

# ─── helpers ──────────────────────────────────────────────────────────────────


def flatten_obs(obs: dict) -> np.ndarray:
    return np.concatenate([obs["positions"], obs["velocities"]])


def random_fault_config(
    rng: np.random.Generator,
    fault_type: FaultType,
) -> FaultConfig:
    """
    Build a FaultConfig with randomised parameters for the given fault type.
    Randomising parameters (drift rate, noise scale, etc.) makes the dataset
    more diverse — the MAE must generalise across fault severities.
    """
    return FaultConfig(
        fault_type=fault_type,
        affected_joint=int(rng.integers(0, N_JOINTS)),
        onset_timestep=int(rng.integers(20, 100)),
        # encoder_drift — vary severity
        drift_rate=float(rng.uniform(0.005, 0.02)),
        # velocity_noise — vary noise level
        noise_scale=float(rng.uniform(0.2, 0.8)),
        # motor_saturation — vary clip limit
        saturation_limit=float(rng.uniform(0.2, 0.5)),
    )


# ─── episode collection ───────────────────────────────────────────────────────


def collect_episode(
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """
    Collect one episode.
    Returns (observations, fault_label).
    observations shape: (EPISODE_LENGTH, OBS_DIM)
    """
    if rng.random() < FAULT_PROB:
        fault_type = FaultType(rng.choice([ft.value for ft in FAULT_TYPES]))
        config = random_fault_config(rng, fault_type)
    else:
        config = FaultConfig(fault_type=FaultType.NONE)

    env = FaultInjector(config)
    episode = [flatten_obs(env.reset())]

    for t in range(EPISODE_LENGTH - 1):
        episode.append(flatten_obs(env.step(t)))

    return np.array(episode, dtype=np.float32), FAULT_LABEL[config.fault_type]


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=SEED)

    print(f"Collecting {N_EPISODES:,} episodes...")
    print(f"  Fault probability: {FAULT_PROB:.0%}")
    print(f"  Fault types:       {[ft.value for ft in FAULT_TYPES]}")
    print(
        f"  Episode length:    {EPISODE_LENGTH} timesteps @ 50Hz = {EPISODE_LENGTH / 50:.1f}s"
    )
    print(f"  Observation dim:   {OBS_DIM}")
    print(f"  Output:            {OUTPUT_PATH}")
    print()

    all_obs = np.empty((N_EPISODES, EPISODE_LENGTH, OBS_DIM), dtype=np.float32)
    all_labels = np.empty(N_EPISODES, dtype=np.int32)

    for i in tqdm(range(N_EPISODES), unit="episode"):
        all_obs[i], all_labels[i] = collect_episode(rng)

    # ── split and save ────────────────────────────────────────────────────────
    splits = {"train": (0, 8000), "val": (8000, 9000), "test": (9000, 10000)}

    with h5py.File(OUTPUT_PATH, "w") as f:
        f.attrs["n_episodes"] = N_EPISODES
        f.attrs["episode_length"] = EPISODE_LENGTH
        f.attrs["obs_dim"] = OBS_DIM
        f.attrs["seed"] = SEED
        f.attrs["fault_prob"] = FAULT_PROB
        f.attrs["fault_types"] = [ft.value for ft in FAULT_TYPES]
        f.attrs["obs_layout"] = "positions (6) | velocities (6)"

        for split, (start, end) in splits.items():
            grp = f.create_group(split)
            grp.create_dataset(
                "observations",
                data=all_obs[start:end],
                compression="gzip",
                compression_opts=4,
            )
            grp.create_dataset(
                "fault_labels",
                data=all_labels[start:end],
            )

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\nDone. Saved to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")
    print()
    print("Label distribution (all episodes):")
    for ft in [FaultType.NONE] + FAULT_TYPES:
        count = int((all_labels == FAULT_LABEL[ft]).sum())
        print(f"  {ft.value:20s}  {count:5d}  ({count / N_EPISODES:.1%})")
    print()
    print("Split sizes:")
    for split, (start, end) in splits.items():
        print(f"  {split:6s}  {end - start:,} episodes")


if __name__ == "__main__":
    main()
