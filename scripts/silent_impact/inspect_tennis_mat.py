
from pathlib import Path
from scipy.io import loadmat
import numpy as np

root = Path("../../data/public/smartwatch_tennis/raw")
files = sorted(root.rglob("*_inertial.mat"))

print(f"Total files: {len(files)}\n")

for file_path in files[:6]:
    data = loadmat(file_path)
    x = np.asarray(data["d_iner"], dtype=float)

    repeated = np.all(x[1:] == x[:-1], axis=1).sum()
    unique_rows = np.unique(x, axis=0).shape[0]

    print(file_path.name)
    print(f"  Samples: {x.shape[0]}")
    print(f"  Unique rows: {unique_rows}")
    print(f"  Consecutive identical rows: {repeated}")
    print()

print("Read-only inspection complete.")
