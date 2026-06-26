import sys
import os
import time
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype, float64, ndarray
from pco import Camera

from APT.thorlabs_apt import Motor

sys.path.append(os.path.abspath(".."))
import connection_setup as cs
import helper_classes.pco_cam_interface as pci
import helper_classes.odmr_plotting as oPlot
import nv_setup.cw_odmr.Lorentzian_fit as Lfit
import pco


"""
measure ODMRs at a variety of Y values

Much of this code was combined from previously written code by Gemini, then edited
"""


roi = None


def measure_binned_odmr_at_y(cam, sg, freqs, dwell, point_duration_s, n_windows, n_iter=1):
    """Executes a dual-directional frequency sweep and returns a 1D binned array, of the kilo brightness per second"""
    brightnesses = np.zeros((n_iter * 2, freqs.size))

    for i in range(n_iter):
        # Forward Frequency Sweep
        brightness = []
        time.sleep(dwell)
        image = pci.read_image(cam, n_windows) # take a first image you ignore, maybe it removes the slightly higher num from first measurement
        for f in freqs:
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = pci.read_image(cam, n_windows)
            all_counts = pci.bin_image(image)
            brightness.append(all_counts / point_duration_s / 1000.0)
        brightnesses[i] = brightness

        brightness = []
        time.sleep(dwell)
        image = pci.read_image(cam, n_windows) # take a first image you ignore
        for f in freqs[::-1]:
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = pci.read_image(cam, n_windows)
            all_counts = pci.bin_image(image)
            brightness.append(all_counts / point_duration_s / 1000.0)
        brightnesses[n_iter + i] = brightness[::-1]

    return np.sum(brightnesses, axis=0) / (n_iter * 2)


def main():
    binning_amount = 4  # Hardware binning configuration (1, 2, or 4)
    # focus_point_size = 256  # in physical (unbinned) pixels, diameter of circle of laser point
    # focus_point_centre_x, focus_point_centre_y = 930,770 # in pixels, center point of the laser point
    focus_point_size = 400  # in physical (unbinned) pixels, diameter of circle of laser point
    focus_point_centre_x, focus_point_centre_y = 850,1000 # in pixels, center point of the laser point

    n_windows_per_point = 1
    amp_dbm = -10  # RF Generator Amplitude
    freq_dwell = 0.001  # Frequency switch recovery interval
    y_dwell = 0.1
    n_iter = 1 # Iterations at each z-step

    # Frequency Sweep Space Configuration
    f_center = 2.87e9  # Hz
    span = 0.3e9  # Hz
    N_freqs = 151  # Total frequency resolution steps

    # Y-Axis Step Parameters
    # y_center = 3.1625  # Target focus center
    # y_span = 0.005  # Distance range over sweep
    # N_y_steps = 5  # Total step divisions to evaluate
    y_center = 4.22  # Target focus center
    y_span = 0.15 # Distance range over sweep
    N_y_steps = 10  # Total step divisions to evaluate

    # Calculate operational sweep coordinates
    _, _, freqs = cs.calc_sweep_range(f_center, span, N_freqs)
    _, _, y_range = cs.calc_sweep_range(y_center, y_span, N_y_steps)
    N_y_steps = len(y_range)

    # -------------------------------------------------------------
    # HARDWARE INITIALIZATION
    # -------------------------------------------------------------
    y_motor, y_prev_position = cs.connect_motor(cs.y_mID)
    roi, _, _ = pci.get_spacial_params(binning_amount, (focus_point_size, focus_point_centre_x, focus_point_centre_y)) # Comment out to use previous image
    # roi = (1,1,pci.camera_resolution//binning_amount,pci.camera_resolution//binning_amount)
    if roi is not None:
        print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    else:
        print("Using cam's previous roi and binning settings")

    y_motor.move_to(y_center)
    time.sleep(5)


    # with pco.Camera() as cam:
    #     pci.set_cam_settings(cam, 10e-3/binning_amount**2, roi=roi, binning=(binning_amount, binning_amount))
    #     pci.auto_expose(cam, target_intensity=0.3)  # sets cameras exposure time

    # cam, sg = pci.connect_cam_RF(roi, binning_amount)
    # point_duration_s = cam.exposure_time * n_windows_per_point


    # cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    print(f"Staging motor to just below initial y-coordinate: {y_range[0]-0.003:.4f}mm...")
    y_motor.move_to(y_range[0]-0.003) # Move to a bit below the first measurement, so always on same side of backlash
    time.sleep(2)  # extra time for the first point

    avg_contrasts, avg_snrs, avg_FWHMs, y_positions, y_sweep_results = (
        pci.run_odmr_measurement((roi, binning_amount, 0.01), amp_dbm, measure_ODMRs,
                                 (freq_dwell, freqs, n_iter, n_windows_per_point, y_dwell, y_motor, y_range)))

    plot_odmrs(N_y_steps, freqs, y_sweep_results)

    plot_SNR_contr(avg_contrasts, avg_snrs, avg_FWHMs, y_positions)

    move_to_user_input(y_motor, y_prev_position)


def move_to_user_input(y_motor: Motor, y_prev_position: float):
    print("\n" + "=" * 40)
    print("ODMRs at different Ys complete")
    print("Type 'exit' to quit.")
    print("Type 'prev' to move to previous position before measurement.")
    print("Type a distance in mm [0,8] to move there")
    print("=" * 40)
    ans = input("Which Y to move to? ").strip().lower()

    if ans.lower() == "exit":
        print("Exiting")
    elif ans == "n":
        y_motor.move_to(y_prev_position)
        time.sleep(1)
        print("moved to pre-optimization position, possibly uncalibrated")
    else:
        try:
            y_motor.move_to(float(ans) - 0.005)
            time.sleep(3)
            y_motor.move_to(float(ans))
            time.sleep(3)
            print(f"Moved to y={ans}mm")
        except:
            print(f"Unable to move to y={ans}mm, exiting")


def plot_odmrs(N_y_steps: int, freqs: ndarray[tuple[Any, ...], dtype[float64]], z_sweep_results):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.85, N_y_steps))

    for idx, (y_pos, counts) in enumerate(z_sweep_results.items()):
        plt.plot(
            freqs / 1e9,
            # counts/max(counts) + 0.004*idx, # display normalized curves above each other
            counts/max(counts), # display normalized curves on top of each other
            # counts,
            label=f"y = {y_pos:.5f} mm, avg val ={np.mean(counts):.1e}",
            color=colors[idx],
            linewidth=2
        )

    plt.xlabel("Frequency [GHz]", fontsize=12)
    plt.ylabel("Normalized Brightness (arb units/s)", fontsize=12)
    plt.title("Binned-sensor ODMR for varying y positions", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()


def plot_SNR_contr(avg_contrasts, avg_snrs, avg_FWHMs, y_positions):
    fig, ax1 = plt.subplots(figsize=(8, 5), layout='constrained')
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax1.set_xlabel("Y Position [mm]", fontsize=12)
    ax1.set_ylabel("Average SNR", color="tab:blue", fontsize=12)
    ax2.set_ylabel("Average Contrast [%]", color="tab:orange", fontsize=12)
    ax3.set_ylabel("Average FWHM (GHz)", color="tab:green", fontsize=12)

    p1 = ax1.plot(y_positions, avg_snrs, 'o-', color="tab:blue", linewidth=2, label="SNR")
    p2 = ax2.plot(y_positions, avg_contrasts, 's-', color="tab:orange", linewidth=2, label="Contrast")
    p2 = ax3.plot(y_positions, avg_FWHMs, 's-', color="tab:green", linewidth=2, label="FWHM")

    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:orange")
    ax3.tick_params(axis='y', labelcolor="tab:green")
    ax3.spines['right'].set_position(('outward', 60))
    plt.title("SNR&contrast&FWHM dependency on Y position", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def measure_ODMRs(cam: Camera, sg: float, freq_dwell: float,
                  freqs: ndarray[tuple[Any, ...], dtype[float64]], n_iter: int, n_windows: int,
                  y_dwell: int, y_motor: Motor,
                  y_range: ndarray[tuple[Any, ...], dtype[float64]]) -> tuple[
    list[float], list[float], list[float], dict[float, float]]:
    point_duration_s = cam.exposure_time * n_windows
    print(f"beggining measurements, estimate time to completion: {len(y_range) * (n_iter * (len(freqs) + 1) * 2 * (point_duration_s + freq_dwell + 0.1) + y_dwell) + 50:.0f}s")

    # Setup data store dictionary: {y_position: odmr_counts_array}
    y_sweep_results = {}

    # Trackers for peak fit data vs Y position
    y_positions = []
    avg_snrs = []
    avg_contrasts = []
    avg_FWHMs = []

    # try:
    for idx, y_pos in enumerate(y_range):
        print(f"\n[{idx + 1}/{len(y_range)}] Moving to y={y_pos:.5f}mm")
        y_motor.move_to(y_pos)
        time.sleep(y_dwell)  # Allow structural mechanical settle time
        img = pci.read_image(cam, 1)
        pci.plot_image(img, title=f"Camera image at y={y_pos:.5f}")

        # Run binned measurement
        counts = measure_binned_odmr_at_y(
            cam, sg, freqs, freq_dwell, point_duration_s, n_windows=n_windows, n_iter=n_iter
        )
        oPlot.save_point_odmr_measurement(counts, freqs)
        y_sweep_results[y_pos] = counts

        # Run Lorentzian curve fitting on current data
        try:
            popt, pcov, _, _, baseline = Lfit.analyze_data(freqs, counts, max_peaks=2)

            # Fetch parameters using localized logic definitions
            contrasts, FWHM, _ = Lfit.get_dip_params(popt)
            snrs = Lfit.get_SNRs(baseline, counts, freqs / 1e9, popt)

            y_positions.append(y_pos)
            if len(snrs) > 0:
                avg_snrs.append(np.mean(snrs))
                avg_contrasts.append(np.mean(contrasts) * 100.0)
                avg_FWHMs.append(np.mean(FWHM))
            else:
                avg_snrs.append(0.0)
                avg_contrasts.append(0.0)
                avg_FWHMs.append(0.0)
            print(
                f"Fit Successful at y={y_pos:.4f}mm: Mean SNR={avg_snrs[-1]:.2f}, Mean Contrast={avg_contrasts[-1]:.2f}%")
        except Exception as fit_error:
            print(f"Data fit sequence rejected at y={y_pos:.4f} mm: {fit_error}")

    return avg_contrasts, avg_snrs, avg_FWHMs, y_positions, y_sweep_results


if __name__ == "__main__":
    main()