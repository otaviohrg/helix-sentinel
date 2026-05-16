from torch.utils.data import DataLoader
from sentinel.data.dataset import FaultDataset

ds = FaultDataset("data/episodes.h5", split="train")
print("Dataset shape info:", ds.shape_info)
print("Train size:", len(ds))

# Test one item
item = ds[0]
for k, v in item.items():
    shape = v.shape if hasattr(v, "shape") else v
    print(f"  {k}: {shape}")

# Test DataLoader
dl = DataLoader(ds, batch_size=64, shuffle=True)
batch = next(iter(dl))
print("\nBatch shapes:")
for k, v in batch.items():
    print(f"  {k}: {v.shape}")

# Verify masking ratio
n_masked = batch["mask"].float().mean()
print(f"\nActual mask ratio: {n_masked:.3f} (expect ~0.75)")
