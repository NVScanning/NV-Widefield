import numpy as np
import time
import datetime
import connection_setup as cs
import os
import sys
sys.path.append(os.path.abspath(".."))
import helper_classes.pco_cam_interface as pci
import helper_classes.odmr_plotting as oPlot
import nv_setup.cw_odmr.Lorentzian_fit as Lfit

"""
Use every pixel read from the camera as it's own sensor in its own independent ODMR
structure + analysis(lorentzian fitting + conversion to B) is quite similar to scanned_cw_odmr,
called from nv_setup.cw_odmr.Lfit
"""

#   Potential features:
#   do translation + widefield, to capture larger structures, atm we're limited to laser focus point size ~15um

time0 = time.time()

def measure_odmr(cam, sg, freqs, dwell, n_windows, n_iter: int = 1) -> np.ndarray:
    point_duration_s = cam.exposure_time * n_windows
    print(f"measuring ODMR with {n_iter} iterations and {n_windows} windows, estimate time to completion ~{n_iter*2 * 1.1 * (len(freqs) + 1) * (dwell + point_duration_s + 0.01*n_windows) + 50:.0f}s")

    image = pci.read_image(cam,1) # Throw out first image, it's often too bright
    # Note: one row of pixels is ~30% brighter than the rest, can't figure out why though

    t0 = time.time()
    brightnesses = np.zeros((n_iter*2, image.shape[0], image.shape[1], freqs.size)) # should be n_iter*2 when reversing as well

    prev_path = oPlot.get_newfile_dir("temp_", print_saving=False)
    with open(prev_path, "a", encoding="utf-8") as f:
        f.write("temp file so no errors come up when deleting")

    for i in range(n_iter):
        brightnesses[i] = pci.sweep_freqs(cam, sg, dwell, freqs, n_windows, n_iter * 2, i * 2)
        brightnesses[n_iter + i] = pci.sweep_freqs(cam, sg, dwell, freqs[::-1], n_windows, n_iter * 2, i * 2 + 1)[::-1]
        prev_path = oPlot.overwrite_2D_odmr_measurement(np.arange(image.shape[0]), np.arange(image.shape[1]), freqs, np.sum(brightnesses,axis=0)/(i*2 + 2), prev_path, print_saving=False)

    sys.stdout.write(f"\r\033[KODMR finished, took {time.time()-t0:.0f}s\n") # Clear progress bar
    sys.stdout.flush()
    return np.sum(brightnesses,axis=0)/(n_iter*2)


def main():
    # params
    camera_binning = 4 # built-int pco camera binning, can only be 1,2,4
    post_processing_binning = 8
    focus_point_size = 288  # in physical (unbinned) pixels, diameter of circle of laser point
    focus_point_centre_x, focus_point_centre_y = 830,1070  # in pixels, center of the laser point
    # TODO: maybe make use of 2D-gaussian to determine centre of focus point automatically
    n_windows_per_point = 10 # n readouts to increase certainty without overexposing
    amp_dbm = -10 # from -30 to -10 work, higher gets more contrast but risks RF coupling, Amp at 28V
    dwell =  0.0 # seconds - time between setting a frequency on fn generator and reading value
    n_iter = 100 # integer >=1
    # frequency parameters
    f_center = 2.87e9 # Hz, generally near 2.87GHz
    span = 0.4e9 # Hz, range of frequencies to sample
    N = 201 # num points in the frequency space to sample

    max_peaks = 4

    roi, x_space, y_space = pci.get_spacial_params(camera_binning,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    # roi=(1,1,pci.camera_resolution//camera_binning,pci.camera_resolution//camera_binning)
    print(f"Using the following roi: {roi} and binning a {camera_binning}x{camera_binning} region")

    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")
    # point_duration_s = cam.exposure_time * n_windows_per_point

    counts_2D = pci.run_odmr_measurement((roi, camera_binning, 0.100), amp_dbm, measure_odmr, (freqs, dwell, n_windows_per_point, n_iter))

    print("Sweeping done")
    if post_processing_binning > 1:
        print(f"now binning {post_processing_binning}x{post_processing_binning} area and converting odmrs to B deltas, estimate time to completion ~{len(x_space)*len(y_space)/post_processing_binning**2}s")
        binned_counts, x_binned, y_binned = pci.bin_counts(counts_2D, post_processing_binning, x_space, y_space)
        B_Z_binned, _ = Lfit.counts_to_B_Z(x_binned, y_binned, binned_counts, freqs, max_peaks=max_peaks)
        print("Conversion done, saving and plotting")
        oPlot.save_2D_odmr_measurement(x_binned, y_binned, freqs, B_Z_binned, binned_counts)
        oPlot.plot_magnet_image(x_binned, y_binned, B_Z_binned)

    print(f"now converting raw odmrs to B deltas, estimate time to completion ~{len(x_space)*len(y_space)}s")
    B_Z_overall, _ = Lfit.counts_to_B_Z(x_space, y_space, counts_2D, freqs, max_peaks=max_peaks)
    print("Conversion done, saving and plotting")
    oPlot.save_2D_odmr_measurement(x_space, y_space, freqs, B_Z_overall, counts_2D)
    oPlot.plot_magnet_image(x_space, y_space, B_Z_overall)




if __name__ == "__main__":
    main()