import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt


# Inject subfolder paths to keep references clean across execution contexts
# current_dir = os.path.dirname(os.path.abspath(__file__))
# nv_widefield_path = os.path.join(current_dir, 'nv_widefield')
# if nv_widefield_path not in sys.path:
#     sys.path.insert(0, nv_widefield_path)

sys.path.append(os.path.abspath(".."))
import connection_setup as cs
import helper_classes.pco_cam_interface as pci
import helper_classes.odmr_plotting as oPlot


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
    focus_point_size = 300  # Full frame window trace allocation
    focus_point_centre_x, focus_point_centre_y = 850, 1130

    n_windows_per_point = 1
    amp_dbm = -10  # RF Generator Amplitude
    freq_dwell = 0.01  # Frequency switch recovery interval
    z_dwell = 0.5
    n_iter = 1  # Iterations per z-step

    # Frequency Sweep Space Configuration
    f_center = 2.87e9  # Hz
    span = 0.1e9  # Hz
    N_freqs = 51  # Total frequency resolution steps

    # Z-Axis Step Parameters
    # z_center = 3.225  # Target focus center
    # z_span = 0.15  # Distance range over sweep
    # N_z_steps = 5  # Total step divisions to evaluate

    # Calculate operational sweep coordinates
    _, _, freqs = cs.calc_sweep_range(f_center, span, N_freqs)
    z_range = [3.211, 3.227, 3.248]
    N_z_steps = len(z_range)

    # -------------------------------------------------------------
    # HARDWARE INITIALIZATION
    # -------------------------------------------------------------
    z_motor = cs.connect_motor(cs.z_mID)
    roi, _, _ = pci.get_spacial_params(binning_amount, (focus_point_size, focus_point_centre_x, focus_point_centre_y))

    print(f"Staging motor to initial z-coordinate: {z_range[0]:.4f}mm...")
    z_motor.move_to(z_range[0])
    time.sleep(5)

    cam, exposure_time, sg = pci.connect_cam_RF(roi, binning_amount)
    point_duration_s = exposure_time * n_windows_per_point  # 1 window per point

    # Setup data store dictionary: {z_position: odmr_counts_array}
    z_sweep_results = {}

    # -------------------------------------------------------------
    # EXECUTION RUN
    # -------------------------------------------------------------
    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(z_dwell)

    try:
        print(f"beggining measurements, estimate time to completion: {n_iter*N_z_steps * (N_freqs * (point_duration_s + freq_dwell) + z_dwell):.0f}s")
        for idx, z_pos in enumerate(z_range):
            print(f"\n[{idx + 1}/{N_z_steps}] Moving Z-Motor to target: {z_pos:.4f}mm")
            z_motor.move_to(z_pos)
            time.sleep(z_dwell)  # Allow structural mechanical settle time
            img = pci.read_image(cam,1)
            pci.plot_image(img, title=f"Camera image at z={z_pos:.4f}") # if roi fills, then plot full image

            # Run binned measurement
            counts = measure_binned_odmr_at_z(
                cam, sg, freqs, freq_dwell, point_duration_s, n_windows=n_windows_per_point, n_iter=n_iter
            )
            oPlot.save_point_odmr_measurement(counts, freqs)
            z_sweep_results[z_pos] = counts

    finally:
        # Guarantee hardware teardown processes fire on exception failures
        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)
        cam.stop()
        # cam.close()

    # -------------------------------------------------------------
    # MULTI-CURVE VISUALIZATION GENERATION
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Use a colormap gradient to clearly distinguish sequential z-steps
    colors = plt.cm.plasma(np.linspace(0, 0.85, N_z_steps))

    for idx, (z_pos, counts) in enumerate(z_sweep_results.items()):
        plt.plot(
            freqs / 1e9,
            counts,
            label=f"z = {z_pos:.4f} mm",
            color=colors[idx],
            linewidth=2
        )

    plt.xlabel("Frequency [GHz]", fontsize=12)
    plt.ylabel("Binned Normalized Brightness (arb units)", fontsize=12)
    plt.title("Binned ODMR Resonance Profile vs. Z-Axis Positioning", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()