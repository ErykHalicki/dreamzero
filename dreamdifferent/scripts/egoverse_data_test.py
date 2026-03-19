import h5py
import matplotlib.pyplot as plt
import numpy as np

H5_PATH = "data/test_data.h5"
NUM_IMAGES = 6

with h5py.File(H5_PATH, "r") as f:
    images = f["observations/images/aria_rgb_cam/color"]
    total = images.shape[0]
    indices = np.linspace(0, total - 1, NUM_IMAGES, dtype=int)
    frames = images[indices]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, img, idx in zip(axes.flat, frames, indices):
    ax.imshow(img)
    ax.set_title(f"Frame {idx}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("data/sample_frames.png", dpi=150)
plt.show()
print(f"Saved sample_frames.png — {NUM_IMAGES} frames from {total} total")
