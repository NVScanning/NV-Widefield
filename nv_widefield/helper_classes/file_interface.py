import numpy as np
from numpy import dtype, float64, ndarray
from typing import Any
import time
import datetime
import os
import json
import numpy as np
from pathlib import Path


import nv_widefield.helper_classes.odmr_plotting as oPlot

# CONSTANTS:
desktop_dir = "C:\\Users\\NVCFM\\Desktop"



def print_params(params):
    print("")
    print("Printing saved params:")
    print("="*40)
    for key, value in params.items():
        if key == "f_center" or key == "span":
            print(f"{key}: {value/1e9} GHz")
        elif key.endswith("dwell") or key.endswith("time"):
            print(f"{key}: {value*1000:.1f} ms")
        elif "dbm" in key:
            print(f"{key}: {value} dBm")
        elif "binning" in key:
            print(f"{key}: {value}x{value}")
        else:
            print(f"{key}: {value}")
    print("="*40)
    print("")

def read_widefield_odmr(date, time) -> tuple[Any, Any, Any, Any, Any, Any]:
    filepath = os.path.join(desktop_dir, "NVCFM_Data", date, "widefield_cw_odmr_" + time)
    data = np.load(filepath + ".npz")

    x_points, y_points, freqs, counts_2D = data["x"], data["y"], data["f"], data["odmrs"]

    if are_params_saved(date, time):
        params = json.loads(str(data["params"].item()))
        print_params(params)
    else:
        params = {}

    if is_B_saved:
        B_Z_overall = data["magnet"]
        oPlot.plot_magnet_image(x_points, y_points, B_Z_overall)
    else:
        freq_deltas = data["deltas"]
        B_Z_overall = np.zeros_like(freq_deltas)
        oPlot.plot_dFreq_image(x_points, y_points, freq_deltas)
    return B_Z_overall, counts_2D, freqs, x_points, y_points, params


def read_snr_contr(date, time) -> tuple[Any, Any]:
    filepath = os.path.join(desktop_dir, "NVCFM_Data", date, "snr_contr_cw_odmr_" + time)
    data = np.load(filepath + ".npz")

    # x_points = data["x"]
    # y_points = data["y"]
    # freqs = data["f"]
    # counts_2D = data["odmrs"]
    snrs = data["snrs"]
    contrasts = data["contr"]

    if are_params_saved(date, time):
        params = json.loads(str(data["params"].item()))
        print_params(params)
    return contrasts, snrs

def read_scanned_ODMR(date, time) -> tuple[Any, Any, Any, Any]:
    filepath = os.path.join(desktop_dir, "NVCFM_Data", date, "scanned_cw_odmr_" + time)

    data = np.load(filepath + ".npz")
    x_points, y_points, freqs, counts_2D = data["x"], data["y"], data["f"], data["odmrs"]

    if are_params_saved(date, time):
        params = json.loads(str(data["params"].item()))
        print_params(params)

    B_saved = is_B_saved(date, time)

    if B_saved:
        B_Z_overall = data["magnet"]
        oPlot.plot_magnet_image(x_points, y_points, B_Z_overall)
    else:
        freq_deltas = data["deltas"]
        oPlot.plot_dFreq_image(x_points, y_points, freq_deltas)
    return counts_2D, freqs, x_points, y_points

def read_point_ODMR(date, time) -> tuple[Any, Any]:
    filepath = os.path.join(desktop_dir, "NVCFM_Data", date, "cw_odmr_" + time)
    data = np.load(filepath + ".npz")
    if is_freq_saved(date, time):
        freqs, counts = data["x"], data["y"]
        if are_params_saved(date, time):
            params = json.loads(str(data["params"].item()))
            print_params(params)
        else:
            params = {}
    else:
        counts = data
        freqs = np.linspace(2.67, 3.07, len(counts), endpoint=True)
        params = {}
        print("Old file being read, frequency data is guessed and cannot be used for numerical analysis.")
    return counts, freqs, params

def save_point_odmr_measurement(counts: ndarray[tuple[Any, ...], dtype[Any]],
                                freqs: ndarray[tuple[Any, ...], dtype[float64]], **params):
    save_path = get_newfile_dir("")
    meta = json.dumps(params, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez(save_path, x=freqs, y=counts, params=meta)



def overwrite_2D_odmr_measurement(x_points, y_points, freqs, counts_2D, prev_path, print_saving=True, Bmap=None, **params):
    time.sleep(2) # make sure new directory is different from old, so you don't overwrite then delete the new one
    new_path = get_newfile_dir("widefield_", print_saving=print_saving)
    # if Bmap is not None:
    #     np.savez(new_path, x=x_points, y=y_points, f=freqs, magnet=Bmap, odmrs=counts_2D)
    # else:
    #     np.savez(new_path, x=x_points, y=y_points, f=freqs, magnet=np.zeros((len(x_points),len(y_points))), odmrs=counts_2D)
    if Bmap is None:
        Bmap = np.zeros((len(x_points),len(y_points)))
        Bmap.fill(np.nan)
    meta = json.dumps(params, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez(new_path, x=x_points, y=y_points, f=freqs, magnet=Bmap, odmrs=counts_2D, params=meta)
    try:
        os.remove(prev_path)
        if print_saving:
            print(f"Deleted old file {prev_path}")
    except Exception as e:
        print("trying to delete old file caused an error:", e)
    return new_path

def save_2D_odmr_measurement(x_points, y_points, freqs, B_Z_overall, counts_2D, **params):
    save_path = get_newfile_dir("widefield_")
    meta = json.dumps(params, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez(save_path, x=x_points, y=y_points, f=freqs, magnet=B_Z_overall, odmrs=counts_2D, params=meta)

def save_2D_odmr_snr_contrast(x_points, y_points, freqs, SNR_overall, contrasts_overall, counts_2D, **params):
    save_path = get_newfile_dir("snr_contr_")
    meta = json.dumps(params, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez(save_path, x=x_points, y=y_points, f=freqs, snr=SNR_overall, contr=contrasts_overall, odmrs=counts_2D, params=meta)



def get_newfile_dir(prefix, print_saving=True):
    now = datetime.datetime.now()
    datestamp = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")
    project_root = "C:\\Users\\NVCFM\\Desktop"
    directory = os.path.join(project_root, "NVCFM_Data", datestamp)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, f"{prefix}cw_odmr_{timestamp}.npz")
    if print_saving:
        print(f"Saving as: {prefix}cw_odmr_{timestamp}.npz in directory: {directory}")
    return save_path



def are_params_saved(date, time):
    are_params_saved = False
    if (date > "2026-10-29"):
        are_params_saved = True
    elif (date == "2026-10-29") & (time > "15-00-00"):
        are_params_saved = True

    return are_params_saved

def is_freq_saved(date, time):
    # Because its in YY-MM-DD we can do:
    is_freq_saved = False
    if (date > "2026-05-13"):
        is_freq_saved = True
    elif (date == "2026-05-13") & (time > "15-02-05"):
        is_freq_saved = True

    return is_freq_saved

def is_B_saved(date, time):
    # Below code is to determine if magnetic field data is saved or not
    is_B_saved = False
    if (date > "2026-05-19"):
        is_B_saved = True
    elif (date == "2026-05-19") & (time > "11-00-00"):
        is_B_saved = True

    return is_B_saved