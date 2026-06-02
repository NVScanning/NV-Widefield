from pathlib import Path
import re
import numpy as np
import sys
import os
# import nv_widefield.binning_exposure_vary as bev
# Get the absolute path to the directory where this script sits
current_dir = os.path.dirname(os.path.abspath(__file__))

# Point directly to the nv_widefield folder containing pco_cam_interface
nv_widefield_path = os.path.join(current_dir, 'nv_widefield')

# Inject it into Python's global search path registry
if nv_widefield_path not in sys.path:
    sys.path.insert(0, nv_widefield_path)

# NOW you can cleanly import your modules
import nv_widefield.binning_exposure_vary as bev

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

# restructure_odmr_data(r"C:\Users\NVCFM\Desktop\NVCFM_Data")



def parse_log_text(log_string):
    # Regex patterns to isolate data parameters
    # Matches: binning+analyzing 1x1 area with 1 window(s)
    setup_pattern = re.compile(
        r"binning\+analyzing\s+(\d+)x\1\s+area\s+with\s+(\d+)\s+window\(s\)"
    )
    # Matches: Overall average SNR:18.86, average contrast:8.89%
    data_pattern = re.compile(
        r"Overall\s+average\s+SNR:\s*([\d\.]+),\s*average\s+contrast:\s*([\d\.]+)%"
    )

    lines = log_string.strip().split('\n')

    # 1. First pass: Collect all unique bins and window counts seen in the log
    discovered_bins = []
    discovered_windows = []

    raw_entries = []
    current_bin = None
    current_win = None

    for line in lines:
        setup_match = setup_pattern.search(line)
        if setup_match:
            current_bin = int(setup_match.group(1))
            current_win = int(setup_match.group(2))

            if current_bin not in discovered_bins:
                discovered_bins.append(current_bin)
            if current_win not in discovered_windows:
                discovered_windows.append(current_win)
            continue

        data_match = data_pattern.search(line)
        if data_match and current_bin is not None and current_win is not None:
            snr = float(data_match.group(1))
            # Convert percentage back to decimal format (e.g., 8.89% -> 0.0889)
            contrast = float(data_match.group(2)) / 100.0

            raw_entries.append({
                'bin': current_bin,
                'window': current_win,
                'snr': snr,
                'contrast': contrast
            })
            # Reset contextual trackers
            current_bin = None
            current_win = None

    # Sort trackers to maintain strict exponential mapping consistency
    discovered_bins.sort()
    discovered_windows.sort()

    n_bins = len(discovered_bins)
    n_windows = len(discovered_windows)

    # 2. Map structural array indexing spaces
    bin_to_index = {b: idx for idx, b in enumerate(discovered_bins)}
    win_to_index = {w: idx for idx, w in enumerate(discovered_windows)}

    # Initialize zero-matrices with exact array footprint
    snr_avg = np.zeros((n_windows, n_bins))
    contr_avg = np.zeros((n_windows, n_bins))

    # Populate matrices dynamically using tracking dict indexes
    for entry in raw_entries:
        w_idx = win_to_index[entry['window']]
        b_idx = bin_to_index[entry['bin']]

        snr_avg[w_idx, b_idx] = entry['snr']
        contr_avg[w_idx, b_idx] = entry['contrast']

    return contr_avg, snr_avg, n_windows, n_bins


# =====================================================================
# EXECUTION COUPLING
# =====================================================================

log_data = """Using the following roi: (993, 993, 1056, 1064) and binning a 1x1 region
Connected to sg386
Adjusting exposure from 0.010 to 0.023s (Peak was: 1.99e+04, now will be ~4.59e+04)
Exposure time:  0.023103
Example image plotted
Frequency range from  2.82  to  2.92  GHz
sg386 ON
measuring ODMR, estimate time to completion ~ 4.87s
Iteration 0
at freq 2.82GHz; 0.0% done
at freq 2.86GHz; 20.0% done
at freq 2.9GHz; 40.0% done
at freq 2.901GHz; 60.0% done
at freq 2.861GHz; 80.0% done
at freq 2.821GHz; 100.0% done
sg386 OFF
binning+analyzing 1x1 area with 1 window(s), estimate time to completion ~20.48s
Saved as: snr_contr_cw_odmr_12-28-54.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:18.86, average contrast:8.89%
binning+analyzing 2x2 area with 1 window(s), estimate time to completion ~5.12s
Saved as: snr_contr_cw_odmr_12-29-00.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:26.24, average contrast:8.89%
binning+analyzing 4x4 area with 1 window(s), estimate time to completion ~1.28s
Saved as: snr_contr_cw_odmr_12-29-02.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:30.45, average contrast:8.89%
binning+analyzing 8x8 area with 1 window(s), estimate time to completion ~0.32s
Saved as: snr_contr_cw_odmr_12-29-02.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.65, average contrast:8.76%
binning+analyzing 16x16 area with 1 window(s), estimate time to completion ~0.08s
Saved as: snr_contr_cw_odmr_12-29-02.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.59, average contrast:8.73%
binning+analyzing 32x32 area with 1 window(s), estimate time to completion ~0.02s
Saved as: snr_contr_cw_odmr_12-29-02.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.95, average contrast:8.67%
binning+analyzing 64x64 area with 1 window(s), estimate time to completion ~0.01s
Saved as: snr_contr_cw_odmr_12-29-02.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.98, average contrast:8.67%
sg386 ON
measuring ODMR, estimate time to completion ~ 9.54s
Iteration 0
at freq 2.82GHz; 0.0% done
at freq 2.86GHz; 20.0% done
at freq 2.9GHz; 40.0% done
at freq 2.901GHz; 60.0% done
at freq 2.861GHz; 80.0% done
at freq 2.821GHz; 100.0% done
sg386 OFF
binning+analyzing 1x1 area with 2 window(s), estimate time to completion ~20.48s
Saved as: snr_contr_cw_odmr_12-30-01.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:22.98, average contrast:8.88%
binning+analyzing 2x2 area with 2 window(s), estimate time to completion ~5.12s
Saved as: snr_contr_cw_odmr_12-30-07.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:28.60, average contrast:8.88%
binning+analyzing 4x4 area with 2 window(s), estimate time to completion ~1.28s
Saved as: snr_contr_cw_odmr_12-30-08.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.05, average contrast:8.88%
binning+analyzing 8x8 area with 2 window(s), estimate time to completion ~0.32s
Saved as: snr_contr_cw_odmr_12-30-09.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.50, average contrast:8.78%
binning+analyzing 16x16 area with 2 window(s), estimate time to completion ~0.08s
Saved as: snr_contr_cw_odmr_12-30-09.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.48, average contrast:8.72%
binning+analyzing 32x32 area with 2 window(s), estimate time to completion ~0.02s
Saved as: snr_contr_cw_odmr_12-30-09.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.50, average contrast:8.70%
binning+analyzing 64x64 area with 2 window(s), estimate time to completion ~0.01s
Saved as: snr_contr_cw_odmr_12-30-09.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.48, average contrast:8.69%
sg386 ON
measuring ODMR, estimate time to completion ~ 18.87s
Iteration 0
at freq 2.82GHz; 0.0% done
at freq 2.86GHz; 20.0% done
at freq 2.9GHz; 40.0% done
at freq 2.901GHz; 60.0% done
at freq 2.861GHz; 80.0% done
at freq 2.821GHz; 100.0% done
sg386 OFF
binning+analyzing 1x1 area with 4 window(s), estimate time to completion ~20.48s
Saved as: snr_contr_cw_odmr_12-31-18.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:26.36, average contrast:8.90%
binning+analyzing 2x2 area with 4 window(s), estimate time to completion ~5.12s
Saved as: snr_contr_cw_odmr_12-31-25.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:29.97, average contrast:8.90%
binning+analyzing 4x4 area with 4 window(s), estimate time to completion ~1.28s
Saved as: snr_contr_cw_odmr_12-31-27.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.27, average contrast:8.90%
binning+analyzing 8x8 area with 4 window(s), estimate time to completion ~0.32s
Saved as: snr_contr_cw_odmr_12-31-27.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.50, average contrast:8.86%
binning+analyzing 16x16 area with 4 window(s), estimate time to completion ~0.08s
Saved as: snr_contr_cw_odmr_12-31-27.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.20, average contrast:8.74%
binning+analyzing 32x32 area with 4 window(s), estimate time to completion ~0.02s
Saved as: snr_contr_cw_odmr_12-31-27.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.13, average contrast:8.71%
binning+analyzing 64x64 area with 4 window(s), estimate time to completion ~0.01s
Saved as: snr_contr_cw_odmr_12-31-27.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.12, average contrast:8.70%
sg386 ON
measuring ODMR, estimate time to completion ~ 37.54s
Iteration 0
at freq 2.82GHz; 0.0% done
at freq 2.86GHz; 20.0% done
at freq 2.9GHz; 40.0% done
at freq 2.901GHz; 60.0% done
at freq 2.861GHz; 80.0% done
at freq 2.821GHz; 100.0% done
sg386 OFF
binning+analyzing 1x1 area with 8 window(s), estimate time to completion ~20.48s
Saved as: snr_contr_cw_odmr_12-32-55.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:28.78, average contrast:8.90%
binning+analyzing 2x2 area with 8 window(s), estimate time to completion ~5.12s
Saved as: snr_contr_cw_odmr_12-33-02.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:30.90, average contrast:8.90%
binning+analyzing 4x4 area with 8 window(s), estimate time to completion ~1.28s
Saved as: snr_contr_cw_odmr_12-33-04.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.59, average contrast:8.90%
binning+analyzing 8x8 area with 8 window(s), estimate time to completion ~0.32s
Saved as: snr_contr_cw_odmr_12-33-04.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.78, average contrast:8.89%
binning+analyzing 16x16 area with 8 window(s), estimate time to completion ~0.08s
Saved as: snr_contr_cw_odmr_12-33-04.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:30.92, average contrast:8.75%
binning+analyzing 32x32 area with 8 window(s), estimate time to completion ~0.02s
Saved as: snr_contr_cw_odmr_12-33-04.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.04, average contrast:8.67%
binning+analyzing 64x64 area with 8 window(s), estimate time to completion ~0.01s
Saved as: snr_contr_cw_odmr_12-33-04.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.03, average contrast:8.66%
sg386 ON
measuring ODMR, estimate time to completion ~ 74.87s
Iteration 0
at freq 2.82GHz; 0.0% done
at freq 2.86GHz; 20.0% done
at freq 2.9GHz; 40.0% done
at freq 2.901GHz; 60.0% done
at freq 2.861GHz; 80.0% done
at freq 2.821GHz; 100.0% done
sg386 OFF
binning+analyzing 1x1 area with 16 window(s), estimate time to completion ~20.48s
Saved as: snr_contr_cw_odmr_12-35-09.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:30.97, average contrast:8.87%
binning+analyzing 2x2 area with 16 window(s), estimate time to completion ~5.12s
Saved as: snr_contr_cw_odmr_12-35-17.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:32.03, average contrast:8.87%
binning+analyzing 4x4 area with 16 window(s), estimate time to completion ~1.28s
Saved as: snr_contr_cw_odmr_12-35-18.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:32.34, average contrast:8.87%
binning+analyzing 8x8 area with 16 window(s), estimate time to completion ~0.32s
Saved as: snr_contr_cw_odmr_12-35-19.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:32.43, average contrast:8.87%
binning+analyzing 16x16 area with 16 window(s), estimate time to completion ~0.08s
Saved as: snr_contr_cw_odmr_12-35-19.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:32.17, average contrast:8.79%
binning+analyzing 32x32 area with 16 window(s), estimate time to completion ~0.02s
Saved as: snr_contr_cw_odmr_12-35-19.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.81, average contrast:8.69%
binning+analyzing 64x64 area with 16 window(s), estimate time to completion ~0.01s
Saved as: snr_contr_cw_odmr_12-35-19.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:31.75, average contrast:8.67%
sg386 ON
measuring ODMR, estimate time to completion ~ 149.54s
Iteration 0
at freq 2.82GHz; 0.0% done
at freq 2.86GHz; 20.0% done
at freq 2.9GHz; 40.0% done
at freq 2.901GHz; 60.0% done
at freq 2.861GHz; 80.0% done
at freq 2.821GHz; 100.0% done
sg386 OFF
binning+analyzing 1x1 area with 32 window(s), estimate time to completion ~20.48s
Saved as: snr_contr_cw_odmr_12-38-37.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:34.08, average contrast:8.88%
binning+analyzing 2x2 area with 32 window(s), estimate time to completion ~5.12s
Saved as: snr_contr_cw_odmr_12-38-44.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:34.75, average contrast:8.88%
binning+analyzing 4x4 area with 32 window(s), estimate time to completion ~1.28s
Saved as: snr_contr_cw_odmr_12-38-46.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:34.93, average contrast:8.88%
binning+analyzing 8x8 area with 32 window(s), estimate time to completion ~0.32s
Saved as: snr_contr_cw_odmr_12-38-46.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:34.98, average contrast:8.88%
binning+analyzing 16x16 area with 32 window(s), estimate time to completion ~0.08s
Saved as: snr_contr_cw_odmr_12-38-46.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:35.00, average contrast:8.88%
binning+analyzing 32x32 area with 32 window(s), estimate time to completion ~0.02s
Saved as: snr_contr_cw_odmr_12-38-46.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:34.38, average contrast:8.72%
binning+analyzing 64x64 area with 32 window(s), estimate time to completion ~0.01s
Saved as: snr_contr_cw_odmr_12-38-46.npz in directory: C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-05-28
Overall average SNR:34.21, average contrast:8.67%
"""

# Call parsing script
contr_avg, snr_avg, n_windows, n_bins = parse_log_text(log_data)

# Print verification to console
print(f"Success! Captured Grid Structure: n_windows={n_windows}, n_bins={n_bins}")
# print("\nSNR Grid Matrix Data:")
# print(snr_avg)

# You can now immediately execute:
bev.plot_exposure_snr_contr_bin(contr_avg, snr_avg, n_windows, n_bins)