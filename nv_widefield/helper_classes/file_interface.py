import numpy as np
from numpy import dtype, float64, ndarray
from typing import Any
import time
import datetime
import os
import json
import numpy as np
from pathlib import Path


def print_params(params):
    print("")
    print("Printing saved params:")
    print("="*40)
    for key, value in params.items():
        print(f"{key}: {value} ")
    print("="*40)
    print("")


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
