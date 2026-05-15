import h5py
import numpy as np

with h5py.File("data/episodes.h5", "r") as f:
    print("Metadata:")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")

    print("\nSplits:")
    for split in ["train", "val", "test"]:
        obs    = f[split]["observations"]
        labels = f[split]["fault_labels"]
        print(f"  {split}: obs={obs.shape}, labels={labels.shape}")
        print(f"    obs range: [{obs[:].min():.3f}, {obs[:].max():.3f}]")
        print(f"    label counts: {np.bincount(labels[:])}")

    # Spot check: plot one faulty and one normal episode
    import matplotlib.pyplot as plt

    obs    = f["train"]["observations"][:]
    labels = f["train"]["fault_labels"][:]

    fault_idx  = np.where(labels > 0)[0][0]
    normal_idx = np.where(labels == 0)[0][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(obs[normal_idx, :, :6])  # positions
    ax1.set_title("Normal episode — joint positions")
    ax1.set_xlabel("timestep")

    ax2.plot(obs[fault_idx, :, :6])
    ax2.set_title(f"Fault episode (label={labels[fault_idx]}) — joint positions")
    ax2.set_xlabel("timestep")

    plt.tight_layout()
    plt.savefig("data/dataset_verify.png", dpi=150)
    print("\nPlot saved to data/dataset_verify.png")