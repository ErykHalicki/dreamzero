import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

H5_PATH = "data/egoverse_test_data.h5"

f = h5py.File(H5_PATH, "r")
dataset = f["observations/images/aria_rgb_cam/color"]
start_frame = 2500
total = dataset.shape[0] - start_frame

fig, ax = plt.subplots()
im = ax.imshow(dataset[start_frame])
ax.axis("off")
#title = ax.set_title("Frame 0")

def update(i):
    im.set_data(dataset[i+start_frame])
    #title.set_text(f"Frame {i}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=total, interval=1, blit=True)
plt.tight_layout()
plt.show()
f.close()
print(f"Played {total} frames")
