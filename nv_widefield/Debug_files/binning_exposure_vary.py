
import time
from typing import Any

import numpy as np
import scipy.ndimage as ndimage
import os
import sys

from numpy import dtype, float64, ndarray

sys.path.append(os.path.abspath(".."))
import nv_setup.connection_setup as cs
import helper_classes.pco_cam_interface as pci
import widefield_odmr as wODMR
import nv_setup.cw_odmr.Lorentzian_fit as Lfit
import helper_classes.odmr_plotting as oPlot
import helper_classes.optimization_plotting as optPlot


from uncertainties import ufloat

"""
This file compares binning and exposure time's effect on contrast and SNR for ODMR dips

binning can be done two ways: on-camera and post-processed
on-camera is limited to 1x1, 2x2, 4x4, but has the benefit of reducing required data transfer
and (albeit minimial) reduction in processing
post-processed can (in theory) be any number (by deleting leftovers), but here I
limit it to powers of 2. It's done by summing the pixel signals from the nxn area, and managing the new array 
as a separate measurement for 2D odmr analysis
"""
####  GLOBAL PARAMS


binning_amount = 1  # built-int pco camera binning, can only be 1,2,4
focus_point_centre_x, focus_point_centre_y = 810, 1110 # in pixels, center point of the laser point
amp_dbm = -10  # anything bigger than -10 does nothing (Hayden)
dwell = 0.001  # seconds - time between setting a frequency on fn generator and reading value
n_iter = 1
# frequency parameters
f_center = 2.87e9  # Hz, generally near 2.87GHz
span = 0.1e9  # Hz, range of frequencies to sample
N = 101  # num points in the frequency space to sample
max_peaks = 2 # TODO: make it able to detect the 4 peaks

def find_best_z(cam, motor, z_range, dwell, point_duration_s, n_windows):
    seen = 0
    num_printouts = 5
    score = np.zeros(z_range.size)
    printout_factor = len(z_range) // num_printouts
    motor.move_to(z_range[0])
    time.sleep(5) # move to starting position, before connecting cam
    for z in z_range:
        if (seen % printout_factor == 0):
            print(f"at z={z:.4f}mm {seen/(printout_factor*num_printouts)*100:.0f}% done")
        motor.move_to(z)
        time.sleep(dwell)
        image = pci.read_image(cam,n_windows)

        laplacian = ndimage.laplace(image.astype(float))
        lScore = laplacian.var()
        all_counts = pci.bin_image(image)
        # score[seen] = (all_counts / point_duration_s/1000)
        score[seen] = (lScore / point_duration_s) # use variance of laplacian to better score focus
        pci.plot_image(image, title=f"z={z:.4f}mm, brightness={all_counts:.2e},  l-score={lScore:.2e}")
        seen += 1
    max_idx = np.argmax(score)
    optimized_z_pos = z_range[max_idx]
    return score, optimized_z_pos, max(score)

def optimize_z():
    print("Optimizing Z")
    z_motor,z_previous_position = cs.connect_motor(cs.z_mID)
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 300  # in pixels, diameter of circle of laser point, must be a multiple of 16
    n_windows_per_point = 1
    dwell = 0.01 # s
    # position z parameters
    z_center = 3.229
    span = 0.005
    N = 21

    z_start, z_end, z_range = cs.calc_sweep_range(z_center, span, N)
    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    z_motor.move_to(z_center)
    time.sleep(5) # move to rough middle, to have good auto exposure time
    cam = pci.connect_cam(roi, binning_amount)


    point_duration_s = cam.exposure_time * n_windows_per_point
    print(f"Optimizing Z, estimate time to completion ~ {N * (dwell + point_duration_s) + 5:.0f}s")
    # here exact exposure time doesn't matter, as long as its constant throughout the range
    try:
        brightnesses, optimized_z_pos, max_brightness = find_best_z(cam, z_motor, z_range, dwell, point_duration_s, n_windows_per_point)
        print(f"optimal z was found at z={optimized_z_pos:.4f}mm, with a brightness of {max_brightness:.2f}, compared to previous position at z={z_previous_position}mm")
        optPlot.plot_Z_dep_graph(z_range, brightnesses)

        ans = input("Move to best position? (Y/N): ").strip().lower()

        if ans == "y":
            z_motor.move_to(optimized_z_pos)
            time.sleep(1)
            print("Moved to optimized position")
        elif ans == "n":
            z_motor.move_to(z_previous_position)
            time.sleep(1)
            print("moved to pre-optimization position, possibly uncalibrated")
        else:
            ans = input("Input height to move to (in mm)").strip().lower()
            try:
                z_motor.move_to(float(ans))
            except:
                print(f"Unable to move to z={ans}")

        image = pci.read_image(cam, n_windows_per_point)
        pci.plot_image(image, title=f"Image at z={z_motor.position:.4f}mm")
    finally:
        # cam.stop()
        cam.close()
    return


def vary_binning():
    # Do one image-ODMR, then do a variety of binnings
    # if I do exponential binning, then my overall image must be a power of 2 wide

    # params
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    n_bins = 8 # to bin 0,1,...n_bins-1 # note: n_bins must be at least 6
    focus_point_size = 2**(n_bins-1)  # in physical (unbinned) pixels, diameter of circle of laser point
    n_windows_per_point = 10 # n readouts to increase certainty without overexposing


    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    # roi=(1,1,pci.camera_resolution,pci.camera_resolution)
    print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    # cam, sg = pci.connect_cam_RF(roi, binning_amount)
    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")
    # point_duration_s = cam.exposure_time * n_windows_per_point

    # cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    # time.sleep(0.1) # why sleep for a whole second? (previous was 1)
    counts_2D = pci.run_odmr_measurement((roi, binning_amount),amp_dbm, wODMR.measure_odmr, (freqs, dwell, n_windows_per_point, n_iter))

    print("")

    binned_contrast_avg, binned_snr_avg, ubinned_contrast_avg, ubinned_snr_avg = bin_full_measurement(counts_2D, freqs,
                                                                                                      n_bins, x_space,
                                                                                                      y_space)

    optPlot.plot_binned_snr_contr(binned_contrast_avg,ubinned_contrast_avg, binned_snr_avg, ubinned_snr_avg, n_bins)

    return


def bin_full_measurement(counts_2D, freqs: ndarray[tuple[Any, ...], dtype[float64]], n_bins: int, x_space: float,
                         y_space: float) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    binned_snr_avg, ubinned_snr_avg = [], []
    binned_contrast_avg, ubinned_contrast_avg = [], []
    for bin in range(n_bins):
        print(f"binning+analyzing {2 ** bin}x{2 ** bin} area:")
        binned_counts, x_binned, y_binned = pci.bin_counts(counts_2D, 2 ** bin, x_space, y_space)
        snrs, contrasts = Lfit.counts_to_SNR_contrast(x_binned, y_binned, binned_counts, freqs, max_peaks)

        overall_avg_snr = ufloat(np.mean(snrs), np.std(snrs))
        overall_avg_contrast = ufloat(np.mean(contrasts), np.std(contrasts))
        binned_snr_avg.append(overall_avg_snr.n)
        binned_contrast_avg.append(overall_avg_contrast.n)
        ubinned_snr_avg.append(overall_avg_snr.s)
        ubinned_contrast_avg.append(overall_avg_contrast.s)

        print(f"Overall average SNR:{overall_avg_snr:.2u}, average contrast:{overall_avg_contrast * 100:.2u}%")
    return binned_contrast_avg, binned_snr_avg, ubinned_contrast_avg, ubinned_snr_avg


def vary_exposure_time():
    # do a series of image-ODMRs, but all with the same camera settings EXCEPT for num windows per point
    # num windows should range from 1-100, maybe logarithmic scale again?

    # params
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 256  # in physical (unbinned) pixels, diameter of circle of laser point
    n_windows = 7 # range exposure time from 2^(0,1,... n_windows-1)


    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    # roi=(1,1,pci.camera_resolution,pci.camera_resolution)
    print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    # cam, sg = pci.connect_cam_RF(roi, binning_amount, 0.1)
    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")

    snr_avg,usnr_avg = [], []
    contr_avg,ucontr_avg = [], []
    for window_exp in range(n_windows):
        n_windows_per_point = 2**window_exp
        # point_duration_s = cam.exposure_time * n_windows_per_point

        # cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
        # time.sleep(0.1) # why sleep for a whole second? (previous was 1)
        counts_2D = pci.run_odmr_measurement((roi, binning_amount, 0.1), amp_dbm, wODMR.measure_odmr, (freqs, dwell, n_windows_per_point, n_iter))

        print(f"\nAnalyzing SNR&contrast from ODMR for {2**window_exp} window(s), estimate time to completion ~{focus_point_size**2/200:.2f}s")
        snrs, contrasts = Lfit.counts_to_SNR_contrast(x_space, y_space, counts_2D, freqs, max_peaks)

        oPlot.save_2D_odmr_snr_contrast(x_space, y_space, freqs, snrs, contrasts, counts_2D)

        overall_avg_snr = ufloat(np.mean(snrs), np.std(snrs))
        overall_avg_contrast = ufloat(np.mean(contrasts), np.std(contrasts))
        print(f"Overall average SNR:{overall_avg_snr:.2u}, average contrast:{overall_avg_contrast * 100:.2u}%")
        snr_avg.append(overall_avg_snr.n)
        contr_avg.append(overall_avg_contrast.n)
        usnr_avg.append(overall_avg_snr.s)
        ucontr_avg.append(overall_avg_contrast.s)

    optPlot.plot_exposure_snr_contr(contr_avg, ucontr_avg, snr_avg, usnr_avg, n_windows)
    return

def vary_exposure_binning():
    # params
    n_bins = 8 # to bin 0,1,...n_bins-1 # note: n_bins must be at least 6
    n_windows = 4 # range exposure time from 2^(0,1,... n_windows-1)
    focus_point_size = 2**(n_bins-1)  # in physical (unbinned) pixels, diameter of circle of laser point
    # binning_amount = 4



    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    # roi=(1,1,pci.camera_resolution,pci.camera_resolution)
    print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    # cam, sg = pci.connect_cam_RF(roi, binning_amount, 0.1)
    # exposure_time = 0.1 # force 100ms exposure
    # cam.exposure_time = exposure_time
    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")

    snr_avg = np.zeros((n_windows, n_bins))
    contr_avg = np.zeros((n_windows, n_bins))
    for window_exp in range(n_windows):
        n_windows_per_point = 2**window_exp
        # point_duration_s = cam.exposure_time * n_windows_per_point

        # cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
        # time.sleep(0.1) # why sleep for a whole second? (previous was 1)
        counts_2D = pci.run_odmr_measurement((roi, binning_amount, 0.1), amp_dbm, wODMR.measure_odmr, (freqs, dwell, n_windows_per_point, n_iter))

        binned_contrast_avg, binned_snr_avg, _, _ = bin_full_measurement(counts_2D, freqs,
                                                                            n_bins, x_space, y_space)
        snr_avg[window_exp, :] = binned_snr_avg
        contr_avg[window_exp, :] = binned_contrast_avg


    optPlot.plot_exposure_snr_contr_bin(contr_avg, snr_avg, n_windows, n_bins)



def main():
    # do things, perhaps only one at a time or all at once idk:
    # 1: vary z, record the whole binned image at each step (with very small steps, <10um)
    # 2: keep exposure time constant, and vary the post-processed binning (probably in an exponential stepsize)
    #   for each pixel, do the ODMR and record SNR&contrast, then average them between all pixels, and record these numbers
    # 3: keep binning to 1x1, and vary total exposure time (change n_windows_per_point, use constant exposure time)
    # 4: keep camera binning to 1x1, vary total exposure time, and vary post-processed binning

    # plot_binned_snr_contr([1,2],[0.1,0.15],[0.5,1.2],[0.07,0.105], 2)

    # 1: optimizing z:
    # optimize_z()

    # 2:
    # vary_binning()

    # 3:
    # vary_exposure_time()

    # 4:
    vary_exposure_binning()


if __name__ == "__main__":
    main()