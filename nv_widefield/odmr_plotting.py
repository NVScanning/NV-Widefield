import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype, float64, ndarray
from typing import Any
from pathlib import Path
import datetime
import os

def plot_odmr(freqs: np.ndarray, kcps: np.ndarray):

    plt.figure(figsize=(8, 5))
    plt.plot(freqs / 1e9, kcps, "-o", markersize=2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("kcps")
    # plt.ylim(7.4e11,7.45e11)
    plt.title("ODMR")
    plt.grid(True)
    plt.show()

def plot_magnet_image(x_points, y_points, B_Z_overall):
    plt.figure(figsize=(10, 6))
    # shading='auto' handles the coordinate mapping automatically
    mesh = plt.pcolormesh(x_points, y_points, B_Z_overall, shading='nearest', cmap='viridis')

    plt.colorbar(mesh, label='B_Z (T)')
    plt.xlabel('x space (mm)')
    plt.ylabel('y space (mm)')
    plt.title('Magnetic Field Heatmap')
    plt.show()
def plot_dFreq_image(x_points, y_points, freq_deltas):
    plt.figure(figsize=(10, 6))
    # shading='auto' handles the coordinate mapping automatically
    mesh = plt.pcolormesh(x_points, y_points, freq_deltas, shading='nearest', cmap='viridis')

    plt.colorbar(mesh, label='B_Z (T)')
    plt.xlabel('x space (mm)')
    plt.ylabel('y space (mm)')
    plt.title('Frequency delta')
    plt.show()


def save_point_odmr_measurement(counts: ndarray[tuple[Any, ...], dtype[Any]],
                                freqs: ndarray[tuple[Any, ...], dtype[float64]]):
    now = datetime.datetime.now()
    datestamp = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")
    script_path = Path(__file__).resolve()
    # project_root = script_path.parent.parent
    project_root = "C:\\Users\\NVCFM\\Desktop"
    directory = os.path.join(project_root, "NVCFM_Data", datestamp)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, f"cw_odmr_{timestamp}.npz")
    print(f"Saved as: cw_odmr_{timestamp}.npz in directory: {directory}")
    np.savez(save_path, x=freqs, y=counts)


def save_2D_odmr_measurement(x_points, y_points, freqs, B_Z_overall, counts_2D):
    now = datetime.datetime.now()
    datestamp = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")
    script_path = Path(__file__).resolve()
    # project_root = script_path.parent.parent.parent
    project_root = "C:\\Users\\NVCFM\\Desktop"
    directory = os.path.join(project_root, "NVCFM_Data", datestamp)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, f"scanned_cw_odmr_{timestamp}.npz")
    print(f"Saved as: scanned_cw_odmr_{timestamp}.npz in directory: {directory}")
    np.savez(save_path, x=x_points, y=y_points, f=freqs, magnet=B_Z_overall, odmrs=counts_2D)
