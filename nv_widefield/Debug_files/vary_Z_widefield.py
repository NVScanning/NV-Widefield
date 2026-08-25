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


def measure_widefield_odmr_at_z(
        cam: Camera,
        sg: Any,
        freqs: ndarray,
        dwell: float,
        n_windows: int,
        n_iter: int,
        x_space: ndarray,
        y_space: ndarray,
) -> tuple[ndarray, str]:
    """Executes a dual-directional widefield frequency sweep at current z position."""

    # Discard warm-up scans
    pci.sweep_freqs_binned_ringBuf(cam, sg, dwell, freqs, min(2, n_windows), 2, 0)
    pci.sweep_freqs_binned_ringBuf(cam, sg, dwell, freqs[::-1], min(2, n_windows), 2, 1)

    t0 = time.time()
    img_shape = (len(y_space), len(x_space))
    brightnesses = np.zeros((n_iter * 2, img_shape[0], img_shape[1], freqs.size))

    prev_path = oPlot.get_newfile_dir("temp_", print_saving=False)
    with open(prev_path, "a", encoding="utf-8") as f:
        f.write("temp file initialization")

    for i in range(n_iter):
        brightnesses[i, :, :, :] = pci.sweep_freqs_ringBuf(
            cam, sg, dwell, freqs, n_windows, n_iter * 2, i * 2, t0
        )
        brightnesses[n_iter + i, :, :, :] = pci.sweep_freqs_ringBuf(
            cam, sg, dwell, freqs[::-1], n_windows, n_iter * 2, i * 2 + 1, t0
        )[:, :, ::-1]

        current_avg = np.sum(brightnesses, axis=0) / (i * 2 + 2)
        prev_path = oPlot.overwrite_2D_odmr_measurement(
            x_space, y_space, freqs, current_avg, prev_path, print_saving=False
        )

    sys.stdout.write(f"\r\033[KODMR sweep finished in {time.time() - t0:.0f}s\n")
    sys.stdout.flush()

    final_counts = np.sum(brightnesses, axis=0) / (n_iter * 2)
    return final_counts, prev_path


def run_widefield_z_sweep(
        cam: Camera,
        sg: Any,
        freq_dwell: float,
        freqs: ndarray,
        n_iter: int,
        n_windows: int,
        z_dwell: float,
        z_motor: Motor,
        z_range: ndarray,
        x_space: ndarray,
        y_space: ndarray,
        post_processing_binning: int,
        max_peaks: int,
) -> None:
    point_duration_s = cam.exposure_time * n_windows
    est_time_per_z = (n_iter * 2 * ((len(freqs) + 1) * (freq_dwell + point_duration_s) + 0.2)) + z_dwell
    print(f"Total sweep estimation: {len(z_range) * est_time_per_z / 3600:.2f} hours")

    z_motor.move_to(z_range[0])
    time.sleep(z_dwell * 3)

    for idx, z_pos in enumerate(z_range):
        print(f"\n[{idx + 1}/{len(z_range)}] Moving to z = {z_pos:.5f} mm")
        z_motor.move_to(z_pos)
        time.sleep(z_dwell)

        # Capture single frame to display camera image and output brightness metrics
        frame = pci.read_image(cam, 1)
        avg_bright = np.mean(frame)
        peak_bright = np.amax(frame)
        print(
            f"z = {z_pos:.5f} mm | Average Brightness: {avg_bright:.2f} counts | Peak Brightness: {peak_bright:.2f} counts")

        pci.plot_image(frame, title=f"Camera Frame at z={z_pos:.5f} mm")

        # Execute widefield ODMR acquisition
        counts_2D, prev_path = measure_widefield_odmr_at_z(
            cam, sg, freqs, freq_dwell, n_windows, n_iter, x_space, y_space
        )

        # Post-processing and magnetic field conversion
        if post_processing_binning > 1:
            binned_counts, x_binned, y_binned = pci.bin_counts(
                counts_2D, post_processing_binning, x_space, y_space
            )
            prev_path = oPlot.overwrite_2D_odmr_measurement(
                x_binned, y_binned, freqs, binned_counts, prev_path, True
            )
            B_Z_binned, _ = Lfit.counts_to_B_Z(
                x_binned, y_binned, binned_counts, freqs, max_peaks=max_peaks
            )
            oPlot.save_2D_odmr_measurement(x_binned, y_binned, freqs, B_Z_binned, binned_counts)
            oPlot.plot_magnet_image(x_binned, y_binned, B_Z_binned)
        else:
            prev_path = oPlot.overwrite_2D_odmr_measurement(
                x_space, y_space, freqs, counts_2D, prev_path, True
            )
            B_Z_overall, _ = Lfit.counts_to_B_Z(
                x_space, y_space, counts_2D, freqs, max_peaks=max_peaks
            )
            oPlot.save_2D_odmr_measurement(x_space, y_space, freqs, B_Z_overall, counts_2D)
            oPlot.plot_magnet_image(x_space, y_space, B_Z_overall)



def main():
    camera_binning = 1
    post_processing_binning = 16
    focus_point_size = 256
    focus_point_centre_x, focus_point_centre_y = 1024, 1024

    n_windows_per_point = 1
    n_iter = 1

    amp_dbm = -10
    freq_dwell = 0.04
    z_dwell = 2.0

    f_center = 2.87e9
    span = 0.2e9
    N_freqs = 401

    z_center = 5.69
    z_span = 0.14
    N_z_steps = 15

    max_peaks = 8

    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N_freqs)
    z_start, z_end, z_range = cs.calc_sweep_range(z_center, z_span, N_z_steps)

    print(f"Sweeping {len(z_range)} Z steps from {z_start:.4f} to {z_end:.4f} mm")

    z_motor, z_prev_position = cs.connect_motor(cs.z_mID)
    roi, x_space, y_space = pci.get_spacial_params(
        camera_binning, (focus_point_size, focus_point_centre_x, focus_point_centre_y)
    )

    if len(x_space) % post_processing_binning != 0:
        raise ValueError("postprocessing binning is not a divisor of the focus region spatial dimension.")

    print(f"Staging motor to backlash offset: {z_range[0] - 0.003:.4f} mm...")
    z_motor.move_to(z_range[0] - 0.003)
    time.sleep(2)

    pci.run_odmr_measurement(
        (roi, camera_binning, 0.02),
        amp_dbm,
        run_widefield_z_sweep,
        (
            freq_dwell,
            freqs,
            n_iter,
            n_windows_per_point,
            z_dwell,
            z_motor,
            z_range,
            x_space,
            y_space,
            post_processing_binning,
            max_peaks,
        ),
    )


if __name__ == "__main__":
    main()