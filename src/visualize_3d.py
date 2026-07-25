import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev

CSV_FILE = "data/session_18_20260720_205751.csv"

WINDOW_LENGTH = 31
POLY_ORDER = 3
INTERPOLATED_POINTS = 2000
MIN_DISTANCE = 3.5#2.55

df = pd.read_csv(CSV_FILE)

x = df["accel_x"].to_numpy()
y = df["accel_y"].to_numpy()
z = df["accel_z"].to_numpy()

x = savgol_filter(x, WINDOW_LENGTH, POLY_ORDER)
y = savgol_filter(y, WINDOW_LENGTH, POLY_ORDER)
z = savgol_filter(z, WINDOW_LENGTH, POLY_ORDER)

tck, u = splprep([x, y, z], s=0)

u_fine = np.linspace(0, 1, INTERPOLATED_POINTS)
x_fine, y_fine, z_fine = splev(u_fine, tck)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

points = np.array([x_fine, y_fine, z_fine]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = Line3DCollection(
    segments,
    cmap="turbo",
    norm=Normalize(0, len(segments)),
    linewidth=2.5
)

lc.set_array(np.arange(len(segments)))
ax.add_collection3d(lc)
ax.auto_scale_xyz(x_fine, y_fine, z_fine)

ax.scatter(x[0], y[0], z[0], color="green", s=100, label="Start")
ax.scatter(x[-1], y[-1], z[-1], color="red", s=100, label="End")

last_x = x[0]
last_y = y[0]
last_z = z[0]

ax.scatter(x[0], y[0], z[0], color="black", s=25)

ax.text(
    x[0],
    y[0],
    z[0],
    "0",
    fontsize=9,
    color="black",
    weight="bold",
    path_effects=[
        pe.withStroke(linewidth=3, foreground="white")
    ]
)

for i in range(1, len(x)):
    distance = np.sqrt(
        (x[i] - last_x) ** 2 +
        (y[i] - last_y) ** 2 +
        (z[i] - last_z) ** 2
    )

    if distance >= MIN_DISTANCE:
        ax.scatter(
            x[i],
            y[i],
            z[i],
            color="black",
            s=25
        )

        ax.text(
            x[i],
            y[i],
            z[i],
            str(i),
            fontsize=9,
            color="black",
            weight="bold",
            path_effects=[
                pe.withStroke(linewidth=3, foreground="white")
            ]
        )

        last_x = x[i]
        last_y = y[i]
        last_z = z[i]

ax.set_title("3D Accelerometer Trajectory")
ax.set_xlabel("Accel X")
ax.set_ylabel("Accel Y")
ax.set_zlabel("Accel Z")

ax.legend()

plt.tight_layout()
plt.show()