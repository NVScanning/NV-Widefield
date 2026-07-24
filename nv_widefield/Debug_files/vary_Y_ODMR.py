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
from scipy.optimize import curve_fit
import re

"""
measure ODMRs at a variety of Y values, then plotting snr, contrast, and FWHM as a fn of Y

Much of this code was combined from previously written code for varying Z
"""


roi = None
max_peaks = 4
I_applied = 1

new_Measurement = False
# old_Measurement_Path = "C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-07-23\\y_dep_00-32-38.txt" # overnight with 1A
# old_Measurement_Path = "C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-07-23\\y_dep_09-46-58.txt" # 0.7A
old_Measurement_Path = "C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\2026-07-23\\y_dep_10-39-51.txt" # larger stepsize 1A


def wire_b_field(y, y0, B_offset, I=1.0):
    """
    Ampere's law model for B-field vs distance from a long straight wire.
    y: physical y-position along sensor (mm)
    y0: wire position along y-axis (mm)
    B_offset: baseline ambient/bias magnetic field (T)
    I: current (A), default 1.0 A
    """
    mu_0 = 4 * np.pi * 1e-7  # T*m/A

    # Convert (y - y0) from mm to meters
    r_meters = np.abs(y - y0) * 1e-3

    # Avoid division by zero
    # r_meters = np.maximum(r_meters, 1e-9)

    return (mu_0 * I) / (2 * np.pi * r_meters) + B_offset

def measure_binned_odmr_at_y(cam, sg, freqs, dwell, point_duration_s, n_windows, n_iter=1):
    """Executes a dual-directional frequency sweep and returns a 1D binned array, of the brightness per second"""
    t0 = time.time()
    brightnesses = np.zeros((n_iter * 2, freqs.size))

    for i in range(n_iter):
        time.sleep(dwell)

        brightnesses[i] = pci.sweep_freqs_binned_ringBuf(cam, sg, dwell, freqs, n_windows, n_iter * 2, i * 2)
        brightnesses[n_iter + i] = pci.sweep_freqs_binned_ringBuf(cam, sg, dwell, freqs[::-1], n_windows, n_iter * 2,
                                                                  i * 2 + 1)[::-1]
    sys.stdout.write(f"\r\033[KODMR finished, took {time.time() - t0:.0f}s\n")  # Clear progress bar
    sys.stdout.flush()

    return np.sum(brightnesses, axis=0) / (n_iter * 2)


def read_txt(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Log text file not found at: {txt_path}")

    y_sweep_results = {}
    freqs = None

    # Regex patterns to match lines containing y-position, filename, and directory
    y_pattern = re.compile(r"Moving to y=([\d\.]+)mm")
    file_pattern = re.compile(r"Saving as:\s*([\w\.\-]+)\s*in directory:\s*(.+)")

    current_y = None

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()

            # Extract y-position
            y_match = y_pattern.search(line_str)
            if y_match:
                current_y = float(y_match.group(1))
                # print("matched y: ", current_y)
                continue

            # Extract filepath and load NPZ file
            file_match = file_pattern.search(line_str)
            if file_match and current_y is not None:
                filename = file_match.group(1)
                directory = file_match.group(2).strip()
                # print("matched file and y, filename:", filename)

                # Build full file path
                filepath = os.path.join(directory, filename)
                if not filepath.endswith(".npz"):
                    filepath += ".npz"

                if not os.path.exists(filepath):
                    print(f"[Warning] Referenced dataset file missing: {filepath}")
                    current_y = None
                    continue

                # Load raw NPZ data structure
                data = np.load(filepath)

                # Assign frequencies array from first valid file
                if freqs is None:
                    freqs = data["x"]

                counts = data["y"]
                y_sweep_results[current_y] = counts

                # Reset tracking variable
                current_y = None

    N_y_steps = len(y_sweep_results)
    return N_y_steps, freqs, y_sweep_results

def plot_B_Y_dep(freqs, y_sweep_results):
    B_vals = []
    fitted_Ys = []

    for idx, (y_pos, counts) in enumerate(y_sweep_results.items()):
        # if idx < 5:
        #     max_peaks = 4
        # elif idx < 2:
        #     max_peaks = 5
        # if idx > 10 or idx < 6:
        #     continue
        # elif idx < 6:
        #     max_peaks = 6
        # elif idx == 5:
        #     max_peaks = 5
        # else:
        #     max_peaks = 6
        delta_freq = Lfit.odmr_to_delta_freq(counts, freqs, max_peaks=max_peaks)
        if delta_freq == 0:
            print(f"Couldn't fit ODMR at y = {y_pos:.4f} mm")
        else:
            B_Z = delta_freq / (2 * cs.gamma_e)  # in T
            B_vals.append(B_Z)
            fitted_Ys.append(y_pos)

    fitted_Ys = np.array(fitted_Ys)
    B_vals = np.array(B_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(fitted_Ys, B_vals, 'o', color='crimson', label='Measured $B_z$', markersize=6)

    # Execute 1/(y - y0) fit if enough valid points exist
    if len(fitted_Ys) >= 3:
        try:
            # Initial parameter guesses:
            # y0: initial guess slightly outside data range or at min y
            # B_offset: baseline ambient field guess
            initial_y0 = 200/10**6 # guess 200um away
            initial_B_offset = 0 # np.min(B_vals)
            p0 = [initial_y0, initial_B_offset]

            popt, pcov = curve_fit(
                lambda y, y0, B_off: wire_b_field(y, y0, B_off, I=I_applied),
                fitted_Ys,
                B_vals,
                p0=p0
            )

            fit_y0, fit_B_off = popt
            perr = np.sqrt(np.diag(pcov))  # Fit parameter standard errors

            # Generate fine evaluation grid for smooth line plot
            y_fine = np.linspace(np.min(fitted_Ys), np.max(fitted_Ys), 200)
            B_fit = wire_b_field(y_fine, fit_y0, fit_B_off, I=I_applied)

            # Convert extracted standoff distance (y0) to mm display
            print(f"--- Wire Distance Fit Results ---")
            print(f"Wire y-position (y0): {fit_y0:.4f} ± {perr[0]:.4f} mm, ~ {fitted_Ys[0]-fit_y0:.4f}mm away from first measurement")
            print(f"Ambient B-offset:    {fit_B_off:.2e} ± {perr[1]:.2e} T")

            plt.plot(
                y_fine,
                B_fit,
                '--',
                color='navy',
                linewidth=2.0,
                label=f'1/r Fit: $(mu_0{I_applied})/(2pi (y-{fit_y0:.2f})) + {fit_B_off:.3e}T$)'
            )

        except Exception as fit_err:
            print(f"[Warning] 1/r Curve fit failed: {fit_err}")

    plt.xlabel("y position [mm]", fontsize=12)
    plt.ylabel(r"$\partial$B [T]", fontsize=12)
    plt.title("B field dependence on position relative to wire", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()

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


def plot_odmrs(N_y_steps: int, freqs: ndarray[tuple[Any, ...], dtype[float64]], y_sweep_results):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.85, N_y_steps))

    for idx, (y_pos, counts) in enumerate(y_sweep_results.items()):
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
    print(f"beggining measurements, estimate time to completion: {len(y_range) * (n_iter * (len(freqs) + 1) * 2 * (point_duration_s + freq_dwell) + y_dwell/2 + 0.1):.0f}s")

    # Throw out first scan, it's always fucked
    y_motor.move_to(y_range[0])
    time.sleep(y_dwell*3)  # Allow structural mechanical settle time
    pci.sweep_freqs_binned_ringBuf(cam, sg, freq_dwell, freqs, min(2, n_windows), 1, 0)
    # pci.sweep_freqs_binned_ringBuf(cam, sg, freq_dwell, freqs[::-1], n_windows, 2,  1)[::-1]
    sys.stdout.write(f"\r\033[KFirst throwaway scan complete\n")  # Clear progress bar

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
        # img = pci.read_image(cam, 1)
        # pci.plot_image(img, title=f"Camera image at y={y_pos:.5f}")

        # Run binned measurement
        counts = measure_binned_odmr_at_y(
            cam, sg, freqs, freq_dwell, point_duration_s, n_windows=n_windows, n_iter=n_iter
        )
        oPlot.save_point_odmr_measurement(counts, freqs)
        y_sweep_results[y_pos] = counts

        # Run Lorentzian curve fitting on current data
        try:
            popt, pcov, _, _, baseline = Lfit.analyze_data(freqs, counts, max_peaks=max_peaks)

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



def main():
    if new_Measurement:
        binning_amount = 4  # Hardware binning configuration (1, 2, or 4)
        # focus_point_size = 256  # in physical (unbinned) pixels, diameter of circle of laser point
        # focus_point_centre_x, focus_point_centre_y = 930,770 # in pixels, center point of the laser point
        focus_point_size = 128  # in physical (unbinned) pixels, diameter of circle of laser point
        focus_point_centre_x, focus_point_centre_y = 1110,1050 # in pixels, center point of the laser point

        n_windows_per_point = 10
        amp_dbm = -10  # RF Generator Amplitude
        freq_dwell = 0.01  # Frequency switch recovery interval
        y_dwell = 1
        n_iter = 2 # Iterations at each z-step

        # Frequency Sweep Space Configuration
        f_center = 2.87e9  # Hz
        span = 0.25e9  # Hz
        N_freqs = 501  # Total frequency resolution steps

        # Y-Axis Step Parameters
        # y_center = 3.1625  # Target focus center
        # y_span = 0.005  # Distance range over sweep
        # N_y_steps = 5  # Total step divisions to evaluate
        # y_center = 4.22  # Target focus center
        # y_span = 0.15 # Distance range over sweep
        y_start = 3.92
        N_y_steps = 10  # Total step divisions to evaluate
        y_stepsize = 0.02 #

        # Calculate operational sweep coordinates
        _, _, freqs = cs.calc_sweep_range(f_center, span, N_freqs)
        # _, _, y_range = cs.calc_sweep_range(y_center, y_span, N_y_steps)
        y_range = np.arange(y_start, y_start+(N_y_steps-1) * y_stepsize, y_stepsize)
        # y_range = [3.92, 3.93, 3.94, 3.95, 4.05,4.06,4.07,4.08,4.09,4.10,4.11,4.12,4.13]
        N_y_steps = len(y_range)
        # print("going to check Y:", y_range)

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

        y_motor.move_to(y_start)
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
            pci.run_odmr_measurement((roi, binning_amount, 0.002), amp_dbm, measure_ODMRs,
                                     (freq_dwell, freqs, n_iter, n_windows_per_point, y_dwell, y_motor, y_range)))

        plot_odmrs(N_y_steps, freqs, y_sweep_results)

        plot_B_Y_dep(freqs, y_sweep_results)
        # plot_SNR_contr(avg_contrasts, avg_snrs, avg_FWHMs, y_positions)
        move_to_user_input(y_motor, y_prev_position)
    else:
        N_y_steps, freqs, y_sweep_results = read_txt(old_Measurement_Path)
        plot_odmrs(N_y_steps, freqs, y_sweep_results)

        plot_B_Y_dep(freqs, y_sweep_results)


if __name__ == "__main__":
    main()