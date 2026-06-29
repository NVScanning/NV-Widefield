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
measure ODMRs at a variety of RF powers

Much of this code was combined from previously written code by Gemini, then edited
"""

amp_name = ""
# amp_name = "SAM100"
# amp_name = "ZHL-2W"

def measure_binned_odmr_at_power(cam, sg, freqs, dwell, point_duration_s, n_windows, n_iter=1):
    """Executes a dual-directional frequency sweep and returns a 1D binned array of brightness per second."""


    t0 = time.time()
    brightnesses = np.zeros((n_iter * 2, freqs.size))

    for i in range(n_iter):
        # Forward Frequency Sweep
        # brightness = []
        time.sleep(dwell)

        brightnesses[i] = pci.sweep_freqs_binned(cam, sg, dwell, freqs, n_windows, n_iter * 2, i * 2)
        brightnesses[n_iter + i] = pci.sweep_freqs_binned(cam, sg, dwell, freqs[::-1], n_windows, n_iter * 2, i * 2 + 1)[::-1]
    sys.stdout.write(f"\r\033[KODMR finished, took {time.time()-t0:.0f}s\n") # Clear progress bar
    sys.stdout.flush()


    return np.sum(brightnesses, axis=0) / (n_iter * 2)


def main():
    binning_amount = 1
    focus_point_size = 150
    focus_point_centre_x, focus_point_centre_y = 880,1045

    n_windows_per_point = 1
    freq_dwell = 0.000
    power_dwell = 0.01  # Settle interval following an amplitude step update
    n_iter = 1

    # Frequency Sweep Space Configuration
    f_center = 2.87e9
    span = 0.1e9
    N_freqs = 51


    # RF Power Amplitude Sweep Parameters
    amp_min = -20.0  # dBm
    amp_max = -10.0  # dBm
    N_amp_steps = 5  # Total power increments to evaluate
    amp_range = np.linspace(amp_min, amp_max, N_amp_steps)

    # Calculate operational frequency sweep coordinates
    _, _, freqs = cs.calc_sweep_range(f_center, span, N_freqs)

    # -------------------------------------------------------------
    # HARDWARE INITIALIZATION
    # -------------------------------------------------------------
    roi, _, _ = pci.get_spacial_params(binning_amount, (focus_point_size, focus_point_centre_x, focus_point_centre_y))

    if roi is not None:
        print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    else:
        print("Using cam's previous roi and binning settings")



    # Pass functional loop array down to interface controller execution stack
    avg_contrasts, avg_snrs, evaluated_powers, power_sweep_results = pci.run_odmr_measurement(
        (roi, binning_amount), -10, measure_power_dependency,
        (freq_dwell, freqs, n_iter, n_windows_per_point, power_dwell, amp_range)
    )

    plot_SNR_contr(avg_contrasts, avg_snrs, evaluated_powers)
    plot_odmrs(N_amp_steps, freqs, power_sweep_results)


def plot_odmrs(N_amp_steps: int, freqs: ndarray[tuple[Any, ...], dtype[float64]], power_sweep_results):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.9, N_amp_steps))

    for idx, (amp_val, counts) in enumerate(power_sweep_results.items()):
        plt.plot(
            freqs / 1e9,
            counts/max(counts),
            label=f"Power = {amp_val:.2f} dBm",
            color=colors[idx],
            linewidth=2
        )

    plt.xlabel("Frequency [GHz]", fontsize=12)
    plt.ylabel("Binned Brightness (arb units/s)", fontsize=12)
    plt.title("Binned ODMR for varying RF power amplitudes", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()


def plot_SNR_contr(avg_contrasts, avg_snrs, evaluated_powers):
    fig, ax1 = plt.subplots(figsize=(8, 5), layout='constrained')
    ax2 = ax1.twinx()

    ax1.set_xlabel("RF Power Amplitude [dBm]", fontsize=12)
    ax1.set_ylabel("Average SNR", color="tab:blue", fontsize=12)
    ax2.set_ylabel("Average Contrast [%]", color="tab:orange", fontsize=12)

    p1 = ax1.plot(evaluated_powers, avg_snrs, 'o-', color="tab:blue", linewidth=2, label="SNR")
    p2 = ax2.plot(evaluated_powers, avg_contrasts, 's-', color="tab:orange", linewidth=2, label="Contrast")

    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:orange")
    if amp_name != "":
        plt.title(f"SNR & Contrast dependency on RF power amplitude with {amp_name} amp", fontsize=14)
    else:
        plt.title(f"SNR & Contrast dependency on RF power", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def measure_power_dependency(cam: Camera, sg: Any, freq_dwell: float,
                             freqs: ndarray[tuple[Any, ...], dtype[float64]], n_iter: int, n_windows: int,
                             power_dwell: float, amp_range: ndarray[tuple[Any, ...], dtype[float64]]) -> tuple[
    list[float], list[float], list[float], dict[float, float]]:
    point_duration_s = cam.exposure_time * n_windows
    print(
        f"Beginning measurements, estimate time to completion: {len(amp_range) * (n_iter * (len(freqs) + 1) * 2 * 1.1 * (point_duration_s + freq_dwell + 0.1) + power_dwell) + 10:.0f}s")

    power_sweep_results = {}
    evaluated_powers = []
    avg_snrs = []
    avg_contrasts = []

    t0 = time.time()

    for idx, amp_val in enumerate(amp_range):
        print(f"\n[{idx + 1}/{len(amp_range)}] Setting RF Amplitude to power={amp_val:.2f} dBm")

        # Target step call requested for setting generator level variables dynamically
        cs.enable_sg386(sg, amp_dbm=amp_val, enable=True)
        time.sleep(power_dwell)

        # Run binned sweep across current power state
        counts = measure_binned_odmr_at_power(
            cam, sg, freqs, freq_dwell, point_duration_s, n_windows=n_windows, n_iter=n_iter
        )
        oPlot.save_point_odmr_measurement(counts, freqs)
        power_sweep_results[amp_val] = counts

        # Run Lorentzian curve fitting
        try:
            popt, pcov, _, _, baseline = Lfit.analyze_data(freqs, counts, max_peaks=2)

            contrasts, _, _ = Lfit.get_dip_params(popt)
            snrs = Lfit.get_SNRs(baseline, counts, freqs / 1e9, popt)

            evaluated_powers.append(amp_val)
            if len(snrs) > 0:
                avg_snrs.append(np.mean(snrs))
                avg_contrasts.append(np.mean(contrasts) * 100.0)
            else:
                avg_snrs.append(0.0)
                avg_contrasts.append(0.0)
            # contrasts, _, _ = Lfit.get_dip_params(popt)
            # snrs = Lfit.get_SNRs(baseline, counts, freqs / 1e9, popt)
            #
            # evaluated_powers.append(amp_val)
            # avg_snrs.append(np.mean(snrs))
            # avg_contrasts.append(np.mean(contrasts) * 100.0)

            print(
                f"Fit Successful at {amp_val:.2f} dBm: Mean SNR={avg_snrs[-1]:.2f}, Mean Contrast={avg_contrasts[-1]:.2f}%")
        except Exception as fit_error:
            print(f"Data fit sequence rejected at {amp_val:.2f} dBm: {fit_error}")

    print(f"Sweeping Z ODMRs took {time.time() - t0:.0f}s")
    return avg_contrasts, avg_snrs, evaluated_powers, power_sweep_results


if __name__ == "__main__":
    main()