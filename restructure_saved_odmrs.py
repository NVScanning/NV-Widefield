from pathlib import Path
import re
import numpy as np


def restructure_odmr_data(source_dir):
    src = Path(source_dir)

    # Matches exactly 'cw_odmr_YYYY-MM-DD_HH-MM-SS.npy'
    pattern = re.compile(r'^cw_odmr_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.npy$')

    # Iterate only over files in the top-level directory
    for file_path in src.iterdir():
        if not file_path.is_file():
            continue

        match = pattern.match(file_path.name)
        if match:
            date_str, time_str = match.groups()

            # Define target paths
            target_subdir = src / date_str
            print("new subdir: ", target_subdir,f"new filename: cw_odmr_{time_str}.npy")
            target_file = target_subdir / f"cw_odmr_{time_str}.npy"

            # Ensure the date subdirectory exists
            target_subdir.mkdir(exist_ok=True)

            # Read, save to new location, and clean up original
            data = np.load(file_path)
            np.save(target_file, data)
            # file_path.unlink()

            print(f"Processed: {file_path.name} -> {date_str}/{target_file.name}")

restructure_odmr_data(r"C:\Users\NVCFM\Desktop\NVCFM_Data")
