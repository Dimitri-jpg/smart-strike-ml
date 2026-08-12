import pandas as pd
import numpy as np

from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev

from src.schemas.visualization import VisualizationResponse, VisualizationPoint, VisualizationMarker


WINDOW_LENGTH = 31
POLY_ORDER = 3
INTERPOLATED_POINTS = 2000
MIN_DISTANCE = 3.5


def create_visualization_data(df: pd.DataFrame):

    df = df[
        ["accel_x", "accel_y", "accel_z"]
    ].copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(df) < 2:
        return VisualizationResponse(
            points=[],
            markers=[]
        )

    x = df["accel_x"].to_numpy(dtype=float)
    y = df["accel_y"].to_numpy(dtype=float)
    z = df["accel_z"].to_numpy(dtype=float)

    if len(x) >= WINDOW_LENGTH:

        window = WINDOW_LENGTH

    else:

        window = len(x)

        if window % 2 == 0:
            window -= 1

        window = max(
            window,
            5
        )

        if window > len(x):
            window = len(x)

            if window % 2 == 0:
                window -= 1

    # Savitzky-Golay requires the polynomial order
    # to be smaller than the window.
    poly_order = min(
        POLY_ORDER,
        window - 1
    )

    if window >= 3 and poly_order >= 1:

        x = savgol_filter(
            x,
            window,
            poly_order
        )

        y = savgol_filter(
            y,
            window,
            poly_order
        )

        z = savgol_filter(
            z,
            window,
            poly_order
        )

    points = np.column_stack(
        (
            x,
            y,
            z
        )
    )

    keep = np.ones(
        len(points),
        dtype=bool
    )

    if len(points) > 1:

        keep[1:] = (
            np.linalg.norm(
                np.diff(
                    points,
                    axis=0
                ),
                axis=1
            ) > 1e-8
        )

    points = points[keep]

    if len(points) < 2:

        return VisualizationResponse(
            points=[
                VisualizationPoint(
                    x=float(points[0][0]),
                    y=float(points[0][1]),
                    z=float(points[0][2])
                )
            ] if len(points) == 1 else [],
            markers=[]
        )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    k = min(
        3,
        len(points) - 1
    )

    tck, u = splprep(
        [x, y, z],
        k=k,
        s=1e-6
    )

    u_fine = np.linspace(
        0,
        1,
        INTERPOLATED_POINTS
    )

    x_fine, y_fine, z_fine = splev(
        u_fine,
        tck
    )

    visualization_points = [
        VisualizationPoint(
            x=float(px),
            y=float(py),
            z=float(pz)
        )
        for px, py, pz in zip(
            x_fine,
            y_fine,
            z_fine
        )
    ]

    # These are the original smoothed samples,
    # corresponding to the numbered points shown
    # by your existing PNG visualization.
    markers = []

    last = np.array(
        [
            x[0],
            y[0],
            z[0]
        ]
    )

    markers.append(
        VisualizationMarker(
            index=0,
            x=float(x[0]),
            y=float(y[0]),
            z=float(z[0])
        )
    )

    for i in range(1, len(x)):

        current = np.array(
            [
                x[i],
                y[i],
                z[i]
            ]
        )

        if np.linalg.norm(
            current - last
        ) >= MIN_DISTANCE:

            markers.append(
                VisualizationMarker(
                    index=i,
                    x=float(x[i]),
                    y=float(y[i]),
                    z=float(z[i])
                )
            )

            last = current

    return VisualizationResponse(
        points=visualization_points,
        markers=markers
    )

