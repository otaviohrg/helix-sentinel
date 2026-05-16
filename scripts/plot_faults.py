#!/usr/bin/env python3

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sentinel.sim.fault_injector import FaultInjector, FaultConfig, FaultType
from sentinel.sim.arm_env import EPISODE_LENGTH

# ─── parameters ───────────────────────────────────────────────────────────────

AFFECTED_JOINT = 5
ONSET_TIMESTEP = 20
DRIFT_RATE     = 0.01
NOISE_SCALE    = 0.1
SAT_LIMIT      = 0.3
STUCK_RESET    = True   # reset _stuck_pos between runs

FAULT_CONFIGS = [
    (FaultType.NONE,             "Normal",           {}),
    (FaultType.JOINT_STICTION,   "Stiction",         {}),
    (FaultType.ENCODER_DRIFT,    "Encoder drift",    {"drift_rate": DRIFT_RATE}),
    (FaultType.VELOCITY_NOISE,   "Velocity noise",   {"noise_scale": NOISE_SCALE}),
    (FaultType.MOTOR_SATURATION, "Motor saturation", {"saturation_limit": SAT_LIMIT}),
]

COLORS = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6", "#8b5cf6"]

# ─── collect episodes ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(len(FAULT_CONFIGS), 2, figsize=(12, 3 * len(FAULT_CONFIGS)))
fig.suptitle(
    f"Fault signatures — joint {AFFECTED_JOINT} · onset t={ONSET_TIMESTEP}",
    fontsize=14, y=1.01,
)

for i, (ft, label, kwargs) in enumerate(FAULT_CONFIGS):
    config = FaultConfig(
        fault_type=ft,
        affected_joint=AFFECTED_JOINT,
        onset_timestep=ONSET_TIMESTEP,
        **kwargs,
    )
    env = FaultInjector(config)
    obs = env.reset()

    positions  = [obs["positions"][AFFECTED_JOINT]]
    velocities = [obs["velocities"][AFFECTED_JOINT]]

    for t in range(EPISODE_LENGTH - 1):
        obs = env.step(t)
        positions.append(obs["positions"][AFFECTED_JOINT])
        velocities.append(obs["velocities"][AFFECTED_JOINT])

    color = COLORS[i]

    for col, data, ylabel in [
        (0, positions,  "position (rad)"),
        (1, velocities, "velocity (rad/s)"),
    ]:
        ax = axes[i, col]
        ax.plot(data, color=color, linewidth=1.2)
        ax.axvline(
            ONSET_TIMESTEP,
            color="#64748b", linestyle="--", linewidth=0.8,
            alpha=0.6, label="fault onset",
        )
        ax.set_title(f"{label} — {ylabel}", fontsize=10)
        ax.set_xlabel("timestep")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

# ─── save ─────────────────────────────────────────────────────────────────────

pathlib.Path("data").mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig("data/fault_signatures.png", dpi=150, bbox_inches="tight")
print("Saved to data/fault_signatures.png")