import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from io import BytesIO

from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev

WINDOW_LENGTH = 31
POLY_ORDER = 3
INTERPOLATED_POINTS = 2000
MIN_DISTANCE = 3.5


def create_visualization(df: pd.DataFrame):

    df = df[["accel_x", "accel_y", "accel_z"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    x = df["accel_x"].to_numpy(dtype=float)
    y = df["accel_y"].to_numpy(dtype=float)
    z = df["accel_z"].to_numpy(dtype=float)

    if len(x) < WINDOW_LENGTH:
        window = len(x)
        if window % 2 == 0:
            window -= 1
        window = max(window, 5)
    else:
        window = WINDOW_LENGTH

    x = savgol_filter(x, window, POLY_ORDER)
    y = savgol_filter(y, window, POLY_ORDER)
    z = savgol_filter(z, window, POLY_ORDER)

    points = np.column_stack((x, y, z))

    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8
    points = points[keep]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    k = min(3, len(points) - 1)

    tck, u = splprep(
        [x, y, z],
        k=k,
        s=1e-6
    )

    u_fine = np.linspace(0, 1, INTERPOLATED_POINTS)
    x_fine, y_fine, z_fine = splev(u_fine, tck)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    fine_points = np.array([x_fine, y_fine, z_fine]).T.reshape(-1, 1, 3)
    segments = np.concatenate([fine_points[:-1], fine_points[1:]], axis=1)

    lc = Line3DCollection(
        segments,
        cmap="turbo",
        norm=Normalize(0, len(segments)),
        linewidth=2.5
    )

    lc.set_array(np.arange(len(segments)))
    ax.add_collection3d(lc)
    ax.auto_scale_xyz(x_fine, y_fine, z_fine)

    ax.scatter(x[0], y[0], z[0], color="green", s=100)
    ax.scatter(x[-1], y[-1], z[-1], color="red", s=100)

    last = np.array([x[0], y[0], z[0]])

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
        current = np.array([x[i], y[i], z[i]])

        if np.linalg.norm(current - last) >= MIN_DISTANCE:

            ax.scatter(x[i], y[i], z[i], color="black", s=25)

            ax.text(
                x[i],
                y[i],
                z[i],
                str(i),
                fontsize=9,
                color="black",
                weight="bold",
                path_effects=[
                    pe.withStroke(
                        linewidth=3,
                        foreground="white"
                    )
                ]
            )

            last = current

    ax.set_title("3D Accelerometer Trajectory")
    ax.set_xlabel("Accel X")
    ax.set_ylabel("Accel Y")
    ax.set_zlabel("Accel Z")

    plt.tight_layout()

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=200
    )

    plt.close(fig)

    buffer.seek(0)

    return buffer