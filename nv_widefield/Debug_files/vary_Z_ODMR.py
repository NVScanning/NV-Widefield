import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(".."))
import connection_setup as cs
import helper_classes.pco_cam_interface as pci
import helper_classes.odmr_plotting as oPlot
import nv_setup.cw_odmr.Lorentzian_fit as Lfit


"""
measure ODMRs at a variety of Z values

Much of this code was combined from previously written code by Gemini, then editeds
"""


def measure_binned_odmr_at_z(cam, sg, freqs, dwell, point_duration_s, n_windows, n_iter=1):
    """Executes a dual-directional frequency sweep and returns a 1D binned array."""
    brightnesses = np.zeros((n_iter * 2, freqs.size))

    for i in range(n_iter):
        # Forward Frequency Sweep
        brightness = []
        for f in freqs:
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = pci.read_image(cam, n_windows)
            all_counts = pci.bin_image(image)
            brightness.append(all_counts / point_duration_s / 1000.0)
        brightnesses[i] = brightness

        brightness = []
        for f in freqs[::-1]:
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = pci.read_image(cam, n_windows)
            all_counts = pci.bin_image(image)
            brightness.append(all_counts / point_duration_s / 1000.0)
        brightnesses[n_iter + i] = brightness[::-1]

    return np.sum(brightnesses, axis=0) / (n_iter * 2)


def main():
    # -------------------------------------------------------------
    # EXPERIMENTAL CONFIGURATION PARAMETERS
    # -------------------------------------------------------------
    binning_amount = 1  # Hardware PCO configuration (1, 2, or 4)
    focus_point_size = 2048  # Full frame window trace allocation
    focus_point_centre_x, focus_point_centre_y = 1024, 1024

    n_windows_per_point = 1
    amp_dbm = -10  # RF Generator Amplitude
    freq_dwell = 0.01  # Frequency switch recovery interval
    z_dwell = 0.2
    n_iter = 1  # Iterations per z-step

    # Frequency Sweep Space Configuration
    f_center = 2.87e9  # Hz
    span = 0.1e9  # Hz
    N_freqs = 51  # Total frequency resolution steps

    # Z-Axis Step Parameters
    z_center = 2.5  # Target focus center
    z_span = 0.05  # Distance range over sweep
    N_z_steps = 11  # Total step divisions to evaluate

    # Calculate operational sweep coordinates
    _, _, freqs = cs.calc_sweep_range(f_center, span, N_freqs)
    _, _, z_range = cs.calc_sweep_range(z_center, z_span, N_z_steps)
    # z_range = [3.211, 3.227, 3.248]
    # N_z_steps = len(z_range)

    # -------------------------------------------------------------
    # HARDWARE INITIALIZATION
    # -------------------------------------------------------------
    z_motor = cs.connect_motor(cs.z_mID)
    roi, _, _ = pci.get_spacial_params(binning_amount, (focus_point_size, focus_point_centre_x, focus_point_centre_y))

    print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    print(f"Staging motor to initial z-coordinate: {z_range[0]:.4f}mm...")
    z_motor.move_to(z_range[0])
    time.sleep(5)

    cam, exposure_time, sg = pci.connect_cam_RF(roi, binning_amount)
    # exposure_time = 0.1
    # cam.exposure_time = exposure_time
    point_duration_s = exposure_time * n_windows_per_point

    # Setup data store dictionary: {z_position: odmr_counts_array}
    z_sweep_results = {}

    # Trackers for peak fit data vs Z position
    z_positions = []
    avg_snrs = []
    avg_contrasts = []

    # -------------------------------------------------------------
    # EXECUTION RUN
    # -------------------------------------------------------------
    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)

    z_motor.move_to(z_range[0])
    time.sleep(2)  # extra time for the first point
    try:
        print(f"beggining measurements, estimate time to completion: {2 * N_z_steps * (n_iter * N_freqs * 2 * (point_duration_s + freq_dwell + focus_point_size**2/(5*10**6)) + z_dwell):.0f}s")
        for idx, z_pos in enumerate(z_range):
            print(f"\n[{idx + 1}/{N_z_steps}] Moving to z={z_pos:.5f}mm")
            z_motor.move_to(z_pos)
            time.sleep(z_dwell)  # Allow structural mechanical settle time
            img = pci.read_image(cam,1)
            pci.plot_image(img, title=f"Camera image at z={z_pos:.5f}") # if roi fills, then plot full image

            # Run binned measurement
            counts = measure_binned_odmr_at_z(
                cam, sg, freqs, freq_dwell, point_duration_s, n_windows=n_windows_per_point, n_iter=n_iter
            )
            oPlot.save_point_odmr_measurement(counts, freqs)
            z_sweep_results[z_pos] = counts

            # Run Lorentzian curve fitting on current data
            try:
                popt, pcov, _, _, baseline = Lfit.analyze_data(freqs, counts, max_peaks=2)

                # Fetch parameters using localized logic definitions
                contrasts, _, _ = Lfit.get_dip_params(popt)
                snrs = Lfit.get_SNRs(baseline, counts, freqs / 1e9, popt)

                z_positions.append(z_pos)
                avg_snrs.append(np.mean(snrs))
                # Convert fraction value back to a clean plotting percentage scale
                avg_contrasts.append(np.mean(contrasts) * 100.0)

                print(
                    f"Fit Successful at z={z_pos:.4f}mm: Mean SNR={avg_snrs[-1]:.2f}, Mean Contrast={avg_contrasts[-1]:.2f}%")
            except Exception as fit_error:
                print(f"Data fit sequence rejected at z={z_pos:.4f} mm: {fit_error}")

    finally:
        # Guarantee hardware teardown processes fire on exception failures
        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)
        # cam.stop()
        cam.close()


    # -------------------------------------------------------------
    # METRIC VS Z HEIGHT GRAPH GENERATION
    # -------------------------------------------------------------
    if len(z_positions) > 0:
        fig, ax1 = plt.subplots(figsize=(8, 5), layout='constrained')
        ax2 = ax1.twinx()

        ax1.set_xlabel("Z Position [mm]", fontsize=12)
        ax1.set_ylabel("Average SNR", color="tab:blue", fontsize=12)
        ax2.set_ylabel("Average Contrast [%]", color="tab:orange", fontsize=12)

        p1 = ax1.plot(z_positions, avg_snrs, 'o-', color="tab:blue", linewidth=2, label="SNR")
        p2 = ax2.plot(z_positions, avg_contrasts, 's-', color="tab:orange", linewidth=2, label="Contrast")

        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax2.tick_params(axis='y', labelcolor="tab:orange")

        plt.title("SNR&contrast dependency on z position", fontsize=14)
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    # -------------------------------------------------------------
    # MULTI-CURVE VISUALIZATION GENERATION
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.85, N_z_steps))

    for idx, (z_pos, counts) in enumerate(z_sweep_results.items()):
        plt.plot(
            freqs / 1e9,
            counts,
            label=f"z = {z_pos:.5f} mm",
            color=colors[idx],
            linewidth=2
        )

    plt.xlabel("Frequency [GHz]", fontsize=12)
    plt.ylabel("Binned Brightness (arb units/s)", fontsize=12)
    plt.title("full-sensor ODMR for varying z positions", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()