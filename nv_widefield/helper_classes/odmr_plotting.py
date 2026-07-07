import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype, float64, ndarray
from typing import Any
import datetime
import os

"""
Lots of plotting code gets used in various placed, but is all quite similar, so has been
compiled into one file to always use when plotting
"""



def plot_odmr(freqs: np.ndarray, kcps: np.ndarray):
    # freqs in Hz
    plt.figure(figsize=(8, 5))
    plt.plot(freqs / 1e9, kcps, "-o", markersize=2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("brightness/s")
    # plt.ylim(7.4e11,7.45e11)
    plt.title("ODMR")
    plt.grid(True)
    plt.show()

def plot_magnet_image(x_points, y_points, B_Z_overall):
    plt.figure(figsize=(10, 6))
    # shading='auto' handles the coordinate mapping automatically
    mesh = plt.pcolormesh(x_points, y_points, B_Z_overall, shading='nearest', cmap='inferno')

    plt.colorbar(mesh, label='B_Z (T)')
    plt.xlabel('x space (mm)')
    plt.ylabel('y space (mm)')
    plt.title('Magnetic Field Heatmap')
    plt.show()


def plot_dFreq_image(x_points, y_points, freq_deltas):
    plt.figure(figsize=(10, 6))
    # shading='auto' handles the coordinate mapping automatically
    mesh = plt.pcolormesh(x_points, y_points, freq_deltas, shading='nearest', cmap='inferno')

    plt.colorbar(mesh, label='freq delta [GHz]')
    plt.xlabel('x space (mm)')
    plt.ylabel('y space (mm)')
    plt.title('Frequency delta')
    plt.show()


def plot_fitted_data(freqs, I_norm, fit_norm):
    # Expects frequencies in GHz
    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')

    ax.plot(freqs, I_norm, '-o', ms=2, color='k')
    ax.plot(freqs, fit_norm, '-', lw=2, color='C0', alpha=0.6, label='Lorentzian fit')

    ax.axvline(
        x=2.87,
        color='red',
        linestyle='--',
        linewidth=1.2,
        alpha=0.7,
        label='2.87 GHz'
    )

    ax.set_title("ODMR")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Normalized Intensity")
    ax.legend(loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_point_odmr_measurement(counts: ndarray[tuple[Any, ...], dtype[Any]],
                                freqs: ndarray[tuple[Any, ...], dtype[float64]]):
    save_path = get_newfile_dir("")
    np.savez(save_path, x=freqs, y=counts)

def overwrite_2D_odmr_measurement(x_points, y_points, freqs, counts_2D, prev_path, print_saving=True):
    new_path = get_newfile_dir("widefield_", print_saving=print_saving)
    np.savez(new_path, x=x_points, y=y_points, f=freqs, magnet=np.zeros((len(x_points),len(y_points))), odmrs=counts_2D)
    try:
        os.remove(prev_path)
        if print_saving:
            print(f"Deleted old file {prev_path}")
    except Exception as e:
        print("trying to delete old file caused an error:", e)
    return new_path

def save_2D_odmr_measurement(x_points, y_points, freqs, B_Z_overall, counts_2D):
    save_path = get_newfile_dir("widefield_")
    np.savez(save_path, x=x_points, y=y_points, f=freqs, magnet=B_Z_overall, odmrs=counts_2D)

def save_2D_odmr_snr_contrast(x_points, y_points, freqs, SNR_overall, contrasts_overall, counts_2D):
    save_path = get_newfile_dir("snr_contr_")
    np.savez(save_path, x=x_points, y=y_points, f=freqs, snr=SNR_overall, contr=contrasts_overall, odmrs=counts_2D)



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

def plot_windows_odmrs(N_windows_steps: int, freqs: ndarray[tuple[Any, ...], dtype[float64]], windows_sweep_results):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.85, N_windows_steps))

    for idx, (n_win, counts) in enumerate(windows_sweep_results.items()):
        plt.plot(
            freqs / 1e9,
            counts/max(counts), # Removed normalization to plot raw absolute counts/intensities
            label=f"Windows = {n_win}, Mean = {np.mean(counts):.1e}",
            color=colors[idx],
            linewidth=2
        )

    plt.xlabel("Frequency [GHz]", fontsize=12)
    plt.ylabel("Binned Brightness (Raw Counts)", fontsize=12)
    plt.title("Unnormalized ODMR Comparison for Varying Number of Windows", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()