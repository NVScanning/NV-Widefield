import sys
import os
import time
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype, float64, ndarray
from pco import Camera

sys.path.append(os.path.abspath(".."))
import connection_setup as cs
import helper_classes.pco_cam_interface as pci
import helper_classes.odmr_plotting as oPlot
import nv_setup.cw_odmr.Lorentzian_fit as Lfit

"""
measure ODMRs at a variety of camera ROI widths, then plot SNR, contrast as a fn of spatial window size
"""


def measure_binned_odmr_at_roi(cam, sg, freqs, dwell, point_duration_s, n_windows, n_iter=1):
    """Executes a dual-directional frequency sweep and returns a 1D binned array of brightness per second."""
    brightnesses = np.zeros((n_iter * 2, freqs.size))

    for i in range(n_iter):
        time.sleep(dwell)
        brightnesses[i] = pci.sweep_freqs_binned(cam, sg, dwell, freqs, n_windows, n_iter * 2, i * 2)
        brightnesses[n_iter + i] = pci.sweep_freqs_binned(cam, sg, dwell, freqs[::-1], n_windows, n_iter * 2,
                                                          i * 2 + 1)[::-1]

    return np.sum(brightnesses, axis=0) / (n_iter * 2)


def main():
    binning_amount = 1
    focus_point_centre_x, focus_point_centre_y = 880, 1070

    n_windows_per_point = 1
    freq_dwell = 0.000
    settle_dwell = 0.00  # Settle interval following a camera ROI re-allocation step
    n_iter = 15

    # Frequency Sweep Space Configuration
    f_center = 2.87e9
    span = 0.3e9
    N_freqs = 151

    # Camera ROI Width Sweep Parameters
    width_min = 100  # pixels
    width_max = 1000  # pixels
    N_width_steps = 5
    roi_width_range = np.linspace(width_min, width_max, N_width_steps, dtype=int)

    # Calculate operational frequency sweep coordinates
    _, _, freqs = cs.calc_sweep_range(f_center, span, N_freqs)

    # Lock RF power to a safe constant reference level for the duration of the geometry sweep
    fixed_rf_power = -10.0

    roi, _, _ = pci.get_spacial_params(binning_amount, (width_min, focus_point_centre_x, focus_point_centre_y)) # Comment out to use previous image
    # roi = (1,1,pci.camera_resolution//binning_amount,pci.camera_resolution//binning_amount)
    # Pass functional loop array down to interface controller execution stack
    avg_contrasts, avg_snrs, evaluated_widths, roi_sweep_results = pci.run_odmr_measurement(
        (None, binning_amount, 0.02), fixed_rf_power, measure_roi_dependency,
        (freq_dwell, freqs, n_iter, n_windows_per_point, settle_dwell, roi_width_range, binning_amount,
         focus_point_centre_x, focus_point_centre_y)
    )

    plot_SNR_contr(avg_contrasts, avg_snrs, evaluated_widths)
    plot_odmrs(N_width_steps, freqs, roi_sweep_results)


def plot_odmrs(N_width_steps: int, freqs: ndarray[tuple[Any, ...], dtype[float64]], roi_sweep_results):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.9, N_width_steps))

    for idx, (width_val, counts) in enumerate(roi_sweep_results.items()):
        plt.plot(
            freqs / 1e9,
            counts / max(counts),
            label=f"ROI Width = {int(width_val)} px",
            color=colors[idx],
            linewidth=2
        )

    plt.xlabel("Frequency [GHz]", fontsize=12)
    plt.ylabel("Binned Brightness (arb units/s)", fontsize=12)
    plt.title("Binned ODMR for varying camera ROI widths", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()


def plot_SNR_contr(avg_contrasts, avg_snrs, evaluated_widths):
    fig, ax1 = plt.subplots(figsize=(8, 5), layout='constrained')
    ax2 = ax1.twinx()

    ax1.set_xlabel("Camera ROI Width [pixels]", fontsize=12)
    ax1.set_ylabel("Average SNR", color="tab:blue", fontsize=12)
    ax2.set_ylabel("Average Contrast [%]", color="tab:orange", fontsize=12)

    p1 = ax1.plot(evaluated_widths, avg_snrs, 'o-', color="tab:blue", linewidth=2, label="SNR")
    p2 = ax2.plot(evaluated_widths, avg_contrasts, 's-', color="tab:orange", linewidth=2, label="Contrast")

    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:orange")
    plt.title("SNR & Contrast dependency on Camera ROI Width", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def measure_roi_dependency(cam: Camera, sg: Any, freq_dwell: float,
                           freqs: ndarray[tuple[Any, ...], dtype[float64]], n_iter: int, n_windows: int,
                           settle_dwell: float, roi_width_range: ndarray[tuple[Any, ...], dtype[int]],
                           binning_amount: int, center_x: int, center_y: int) -> tuple[
    list[float], list[float], list[int], dict[int, float]]:
    point_duration_s = cam.exposure_time * n_windows
    print(f"Beginning spatial loop. Width parameters: {roi_width_range} pixels.")

    roi_sweep_results = {}
    evaluated_widths = []
    avg_snrs = []
    avg_contrasts = []

    t0 = time.time()
    total_steps = len(roi_width_range)

    for idx, width_val in enumerate(roi_width_range):
        # Dynamically calculate the new bounding parameters for the camera sensor array
        new_roi, _, _ = pci.get_spacial_params(binning_amount, (int(width_val), center_x, center_y))

        # Apply the new geometry allocation directly into the camera properties register
        # cam.set_roi(new_roi)
        pci.set_cam_settings(cam, cam.exposure_time, new_roi, (binning_amount, binning_amount))
        time.sleep(settle_dwell)

        # Draw updated processing status bar onto the active line row
        # percent = int((idx + 1) / total_steps * 100)
        # bar = '█' * int(20 * (idx + 1) // total_steps) + '-' * (20 - int(20 * (idx + 1) // total_steps))
        # sys.stdout.write(f"\r\033[K[{bar}] {percent}% | Acquiring ROI Width: {width_val} px...")
        # sys.stdout.flush()

        # Execute sweep collection
        counts = measure_binned_odmr_at_roi(
            cam, sg, freqs, freq_dwell, point_duration_s, n_windows=n_windows, n_iter=n_iter
        )
        oPlot.save_point_odmr_measurement(counts, freqs)
        roi_sweep_results[int(width_val)] = counts

        try:
            popt, pcov, _, _, baseline = Lfit.analyze_data(freqs, counts, max_peaks=2)
            contrasts, _, _ = Lfit.get_dip_params(popt)
            snrs = Lfit.get_SNRs(baseline, counts, freqs / 1e9, popt)

            evaluated_widths.append(int(width_val))
            if len(snrs) > 0:
                avg_snrs.append(np.mean(snrs))
                avg_contrasts.append(np.mean(contrasts) * 100.0)
            else:
                avg_snrs.append(0.0)
                avg_contrasts.append(0.0)
        except Exception:
            # Fallback values to keep data sets symmetric in case a fit breaks
            evaluated_widths.append(int(width_val))
            avg_snrs.append(0.0)
            avg_contrasts.append(0.0)

    # Clean execution wrapup: erase the progress tracking bar row completely
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

    print(f"Sweeping spatial ROI parameters finished. Total duration: {time.time() - t0:.0f}s")
    return avg_contrasts, avg_snrs, evaluated_widths, roi_sweep_results


if __name__ == "__main__":
    main()