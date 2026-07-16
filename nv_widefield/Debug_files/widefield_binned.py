from typing import Any

import numpy as np
import time

from numpy import dtype, float64, ndarray

import connection_setup as cs
import os
import sys
# sys.path.append(os.path.abspath("..."))
import nv_setup.cw_odmr.Lorentzian_fit as Lfit
# import nv_widefield.pco_cam_interface as pci
# import nv_widefield.odmr_plotting as oPlot
sys.path.append(os.path.abspath(".."))
import helper_classes.pco_cam_interface as pci
import helper_classes.odmr_plotting as oPlot
import helper_classes.Log as Log

"""
A step in the direction of widefield imaging, by using the camera sensor, but binning
all the relevant pixels together into one "brightness" signal, to use in place of the
SPCM used in cw_odmr.py

This was used initially when converting code from SPCM-based measurement to camera-based,
and now is used as a quick check for ODMR sensitivity
"""

roi = None

def measure_odmr(cam, sg, freqs, dwell, n_windows, n_iter: int = 1) -> np.ndarray:
    point_duration_s = cam.exposure_time * n_windows
    print(f"measuring binned ODMR with {n_iter} iterations, estimate time to completion"
          f" ~{n_iter*2 * ((len(freqs) + 1) * (dwell + point_duration_s) + 0.02):.0f}s")
    # seen=0

    # num_printouts = 5
    # printout_factor = len(freqs)*n_iter*2 // num_printouts
    t0 = time.time()
    brightnesses = np.zeros((n_iter*2, freqs.size)) # should be n_iter*2 when reversing as well
    for i in range(n_iter):
        # print("Iteration " + str(i))
        # time.sleep(dwell)
        # Log.log("Before reading image")
        # pci.read_image(cam,n_windows) # ignore = first image - it has higher intensity for some reason
        # Log.log("after reading image, before sweeping freqs")
        Log.log("Sweeping freqs:")
        brightnesses[i] = pci.sweep_freqs_binned_ringBuf(cam, sg, dwell, freqs, n_windows, n_iter * 2, i * 2)
        brightnesses[n_iter + i] = pci.sweep_freqs_binned_ringBuf(cam, sg, dwell, freqs[::-1], n_windows, n_iter * 2, i * 2 + 1)[::-1]
        Log.log("after sweeping freqs")

        # TODO: have it save partial measurements after each iteration, like widefield does
    sys.stdout.write(f"\r\033[KODMR finished, took {time.time()-t0:.0f}s\n") # Clear progress bar
    sys.stdout.flush()
    return np.sum(brightnesses,axis=0)/(n_iter*2)


def main():
    # Log.start()

    # params
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 200  # in pixels, approximate width of image taken, must be >=32 after binning
    focus_point_centre_x, focus_point_centre_y = 1020,1010  # in pixels, center of the laser point

    n_windows_per_point = 1 # n readouts to increase certainty without overexposing
    amp_dbm = -10 #anything bigger than -10 does nothing (Hayden)
    # Always use with 28V on the amplifier, amp_dbm ~30 is the lowest you can set while still seeing the zero-field dips
    # Larger amp means dips are more visible, but also get wider so you lose frequency resolution

    dwell =  0.0 # seconds - time between setting a frequency on fn generator and reading value
    n_iter = 2
    # frequency parameters
    f_center = 2.87e9 # Hz, generally near 2.87GHz
    span = 2e9 # Hz, range of frequencies to sample
    N = 601 # num points in the frequency space to sample
    # f_center = 3.335e9 # Hz, generally near 2.87GHz
    # span = 0.5e9 # Hz, range of frequencies to sample
    # N = 201 # num points in the frequency space to sample

    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    # roi=(1,1,pci.camera_resolution//binning_amount,pci.camera_resolution//binning_amount)
    print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")


    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    print(f"Frequency range from {f_start/1e9:.3f} to {f_end/1e9:.3f}GHz")

    counts = pci.run_odmr_measurement((roi, binning_amount, 0.01), amp_dbm, measure_odmr, (freqs, dwell, n_windows_per_point, n_iter))

    oPlot.plot_odmr(freqs, counts)

    oPlot.save_point_odmr_measurement(counts, freqs)

    max_peaks = 12
    popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
    # Lfit.print_dip_params(popt)
    contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)


    try:
        snrs = Lfit.get_SNRs(baseline, counts, freqs/10**9, popt)
        # Lfit.print_SNR(snrs, dip_Freqs)
        # Lfit.print_contrast_snr(contrasts, snrs, dip_Freqs)
        Lfit.print_contrast_snr_FWHM(contrasts, snrs, FWHMs, dip_Freqs)
    except ValueError as e:
        # do nothing cuz printing snr didnt work
        print("getting SNR failed: " + str(e))
    oPlot.plot_fitted_data(freqs/10**9, counts_norm, fitted_norm)



if __name__ == "__main__":
    main()