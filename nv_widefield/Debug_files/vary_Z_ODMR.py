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
import helper_classes.optimization_plotting as optPlot
import nv_setup.cw_odmr.Lorentzian_fit as Lfit
import pco


"""
measure ODMRs at a variety of Z values, then plot contrast and SNR as a fn of Z

Much of this code was combined from previously written code by Gemini, then edited
"""


roi = None
max_peaks = 4

def measure_binned_odmr_at_z(cam, sg, freqs, dwell, point_duration_s, n_windows, n_iter=1):
    """Executes a dual-directional frequency sweep and returns a 1D binned array, of the brightness per second"""

    t0 = time.time()
    brightnesses = np.zeros((n_iter * 2, freqs.size))

    for i in range(n_iter):
        time.sleep(dwell)

        brightnesses[i] = pci.sweep_freqs_binned_ringBuf(cam, sg, dwell, freqs, n_windows, n_iter * 2, i * 2)
        brightnesses[n_iter + i] = pci.sweep_freqs_binned_ringBuf(cam, sg, dwell, freqs[::-1], n_windows, n_iter * 2, i * 2 + 1)[::-1]
    sys.stdout.write(f"\r\033[KODMR finished, took {time.time() - t0:.0f}s\n")  # Clear progress bar
    sys.stdout.flush()


    return np.sum(brightnesses, axis=0) / (n_iter * 2)


def main():
    binning_amount = 1  # Hardware binning configuration (1, 2, or 4)
    focus_point_size = 200  # in physical (unbinned) pixels, diameter of circle of laser point
    focus_point_centre_x, focus_point_centre_y = 1110,1010 # in pixels, center point of the laser point

    n_windows_per_point = 1
    amp_dbm = -10  # RF Generator Amplitude
    freq_dwell = 0.01  # Frequency switch recovery interval
    z_dwell = 1
    n_iter = 3 # Iterations at each z-step

    # Frequency Sweep Space Configuration
    f_center = 2.87e9  # Hz
    span = 0.2e9  # Hz
    N_freqs = 101  # Total frequency resolution steps
    # f_center = 5.3e9  # Hz
    # span = 0.4e9  # Hz
    # N_freqs = 51  # Total frequency resolution steps

    # Z-Axis Step Parameters
    z_center = 6.04 # Target focus center
    z_span = 0.06 # Distance range over sweep
    N_z_steps = 7     # Total step divisions to evaluate

    # Calculate operational sweep coordinates
    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N_freqs)
    z_start, z_end, z_range = cs.calc_sweep_range(z_center, z_span, N_z_steps)
    N_z_steps = len(z_range)
    # print(f"Frequency range from {f_start/1e9:.3f} to {f_end/1e9:.3f}GHz")
    print(f"Z range from {z_start:.4f} to {z_end:.4f}mm")

    # -------------------------------------------------------------
    # HARDWARE INITIALIZATION
    # -------------------------------------------------------------
    z_motor, z_prev_position = cs.connect_motor(cs.z_mID)
    roi, _, _ = pci.get_spacial_params(binning_amount, (focus_point_size, focus_point_centre_x, focus_point_centre_y)) # Comment out to use previous image
    # roi = (1,1,pci.camera_resolution//binning_amount,pci.camera_resolution//binning_amount)
    if roi is not None:
        print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    else:
        print("Using cam's previous roi and binning settings")

    z_motor.move_to(z_center)
    time.sleep(5)



    # cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    print(f"Staging motor to just below initial z-coordinate: {z_range[0]-0.003:.4f}mm...")
    z_motor.move_to(z_range[0]-0.003) # Move to a bit below the first measurement, so always on same side of backlash
    time.sleep(2)  # extra time for the first point

    avg_contrasts, avg_snrs, z_positions, z_sweep_results = (
        pci.run_odmr_measurement((roi, binning_amount,0.005), amp_dbm, measure_ODMRs,
                                 (freq_dwell, freqs, n_iter, n_windows_per_point, z_dwell, z_motor, z_range)))

    plot_odmrs(N_z_steps, freqs, z_sweep_results)

    optPlot.plot_z_SNR_contr(avg_contrasts, avg_snrs, z_positions)

    move_to_user_input(z_motor, z_prev_position)


def move_to_user_input(z_motor: Motor, z_prev_position: float):
    print("\n" + "=" * 40)
    print("ODMRs at different Zs complete")
    print("Type 'exit' to quit.")
    print("Type 'prev' to move to previous position before measurement.")
    print("Type a distance in mm [0,8] to move there")
    print("=" * 40)
    ans = input("Which Z to move to? ").strip().lower()

    if ans.lower() == "exit":
        print("Exiting")
    elif ans == "n":
        z_motor.move_to(z_prev_position)
        time.sleep(1)
        print("moved to pre-optimization position, possibly uncalibrated")
    else:
        try:
            z_motor.move_to(float(ans) - 0.005)
            time.sleep(3)
            z_motor.move_to(float(ans))
            time.sleep(3)
            print(f"Moved to z={ans}mm")
        except:
            print(f"Unable to move to z={ans}mm, exiting")


def plot_odmrs(N_z_steps: int, freqs: ndarray[tuple[Any, ...], dtype[float64]], z_sweep_results):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.85, N_z_steps))

    for idx, (z_pos, counts) in enumerate(z_sweep_results.items()):
        plt.plot(
            freqs / 1e9,
            # counts/max(counts) + 0.004*idx, # display normalized curves above each other
            counts/max(counts), # display normalized curves on top of each other
            # counts,
            label=f"z = {z_pos:.5f} mm, avg val ={np.mean(counts):.1e}",
            color=colors[idx],
            linewidth=2
        )

    plt.xlabel("Frequency [GHz]", fontsize=12)
    plt.ylabel("Normalized Brightness (arb units/s)", fontsize=12)
    plt.title("Binned-sensor ODMR for varying z positions", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()




def measure_ODMRs(cam: Camera, sg: float, freq_dwell: float,
                  freqs: ndarray[tuple[Any, ...], dtype[float64]], n_iter: int, n_windows: int,
                  z_dwell: int, z_motor: Motor,
                  z_range: ndarray[tuple[Any, ...], dtype[float64]]) -> tuple[
    list[float], list[float], list[float], dict[float, float]]:
    point_duration_s = cam.exposure_time * n_windows
    print(f"beggining measurements, estimate time to completion: "
          f"{len(z_range) * ((n_iter * (len(freqs) + 1) * 2 * (point_duration_s + freq_dwell) + 0.02) + z_dwell) + (len(freqs) + 1)*(point_duration_s + freq_dwell) +0.02 + z_dwell * 3:.0f}s")

    t0 = time.time()
    # Setup data store dictionary: {z_position: odmr_counts_array}
    z_sweep_results = {}

    # Trackers for peak fit data vs Z position
    z_positions = []
    avg_snrs = []
    avg_contrasts = []

    # Throw out first scan, it's always fucked
    z_motor.move_to(z_range[0])
    time.sleep(z_dwell*3)  # Allow structural mechanical settle time
    pci.sweep_freqs_binned_ringBuf(cam, sg, freq_dwell, freqs, min(2, n_windows), 1, 0)
    # pci.sweep_freqs_binned_ringBuf(cam, sg, freq_dwell, freqs[::-1], n_windows, 2,  1)[::-1]
    sys.stdout.write(f"\r\033[KFirst throwaway scan complete\n")  # Clear progress bar
    sys.stdout.flush()


    # try:
    for idx, z_pos in enumerate(z_range):
        print(f"\n[{idx + 1}/{len(z_range)}] Moving to z={z_pos:.5f}mm")
        z_motor.move_to(z_pos)
        time.sleep(z_dwell)  # Allow structural mechanical settle time


        # img = pci.read_image(cam, 1)
        # pci.plot_image(img, title=f"Camera image at z={z_pos:.5f}")  # if roi fills, then plot full image

        # Run binned measurement
        counts = measure_binned_odmr_at_z(
            cam, sg, freqs, freq_dwell, point_duration_s, n_windows=n_windows, n_iter=n_iter
        )
        oPlot.save_point_odmr_measurement(counts, freqs)
        z_sweep_results[z_pos] = counts

        # Run Lorentzian curve fitting on current data
        try:
            popt, pcov, _, _, baseline = Lfit.analyze_data(freqs, counts, max_peaks=max_peaks)

            # Fetch parameters using localized logic definitions
            contrasts, _, _ = Lfit.get_dip_params(popt)
            snrs = Lfit.get_SNRs(baseline, counts, freqs / 1e9, popt)

            z_positions.append(z_pos)
            if len(snrs) > 0:
                avg_snrs.append(np.mean(snrs))
                avg_contrasts.append(np.mean(contrasts) * 100.0)
            else:
                avg_snrs.append(0.0)
                avg_contrasts.append(0.0)
            print(
                f"Fit Successful at z={z_pos:.4f}mm: Mean SNR={avg_snrs[-1]:.2f}, Mean Contrast={avg_contrasts[-1]:.2f}%")
        except Exception as fit_error:
            print(f"Data fit sequence rejected at z={z_pos:.4f} mm: {fit_error}")

    print(f"Sweeping Z ODMRs took {time.time() - t0:.0f}s")
    return avg_contrasts, avg_snrs, z_positions, z_sweep_results


if __name__ == "__main__":
    main()