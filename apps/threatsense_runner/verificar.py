import h5py
import glob
import os

folder = "output/collect_and_save/"
files = sorted(glob.glob(os.path.join(folder, "*.h5")))

total = 0
for fpath in files:
    with h5py.File(fpath, "r") as f:
        n = f["teacher_actions"].shape[0]
        print(f"{os.path.basename(fpath)}: {n} exemplos")
        total += n

print("TOTAL:", total)
