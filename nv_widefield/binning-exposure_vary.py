
import time
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(".."))
import nv_setup.connection_setup as cs
import pco_cam_interface as pci
import widefield_odmr as wODMR
import nv_setup.cw_odmr.Lorentzian_fit as Lfit
import odmr_plotting as oPlot
from uncertainties import ufloat

"""
This file compares binning and exposure time's effect on contrast and SNR for ODMR dips
atm it can only vary binning, but will soon

binning can be done two ways: on-camera and post-processed
on-camera is limited to 1x1, 2x2, 4x4, but has the benefit of reducing required data transfer
and (albeit minimial) reduction in processing
post-processed can (in theory) be any number (by deleting leftovers), but here I
limit it to powers of 2. It's done by summing the ODMRs from the nxn area, and managing the new array 
as a separate measurement for 2D odmr analysis
"""


def plot_Z_dep_graph(z_range: np.ndarray, kcps: np.ndarray):
    plt.figure(figsize=(8, 5))
    plt.plot(z_range, kcps, "-o", markersize=2)
    plt.xlabel("Z Position (mm)")
    plt.ylabel("kcps")
    plt.title("Counts as a fn of Z")
    plt.grid(True)
    plt.show()

def plot_binned_snr_contr(binned_contrast_avg,ubinned_contrast_avg, binned_snr_avg,ubinned_snr_avg, n_bins: int):
    bins = np.power(2,range(n_bins))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('binning axis size(pixels)')
    ax1.set_ylabel('Average SNR', color=color)
    ax1.errorbar(bins, binned_snr_avg, yerr=ubinned_snr_avg, markersize=3, capsize=5, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Average contrast(%)', color=color)  # we already handled the x-label with ax1
    ax2.errorbar(bins, np.array(binned_contrast_avg)*100, yerr=np.array(ubinned_contrast_avg)*100, markersize=3, capsize=5, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(True)
    plt.show()
def plot_exposure_snr_contr(contr_avg, ucontr_avg, snr_avg, usnr_avg, n_windows: int):
    bins = np.power(2,range(n_windows))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('exposure time (num windows per point)')
    ax1.set_ylabel('Average SNR', color=color)
    ax1.errorbar(bins, snr_avg, yerr=usnr_avg, markersize=3, capsize=5, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Average contrast(%)', color=color)  # we already handled the x-label with ax1
    ax2.errorbar(bins, np.array(contr_avg) * 100, yerr=np.array(ucontr_avg) * 100, markersize=3, capsize=5, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(True)
    plt.show()

def find_best_z(cam, motor, z_range, dwell, point_duration_s, n_windows):
    seen = 0
    num_printouts = 5
    brightnesses = np.zeros(z_range.size)
    printout_factor = len(z_range) // num_printouts
    motor.move_to(z_range[0])
    time.sleep(5) # move to starting position, before connecting cam
    for z in z_range:
        if (seen % 10 == 0):
            print(f"at z={z:.4f}mm {seen/(printout_factor*num_printouts)*100:.1f}% done")
        motor.move_to(z)
        time.sleep(dwell)
        # print('count seen')
        image = pci.read_image(cam,n_windows)
        all_counts = pci.bin_image(image)
        brightnesses[seen] = (all_counts / point_duration_s/1000)
        seen += 1
    max_idx = np.argmax(brightnesses)
    optimized_z_pos = z_range[max_idx]
    return brightnesses, optimized_z_pos, max(brightnesses)

def optimize_z():
    print("Optimizing Z")
    z_motor = cs.connect_motor(cs.z_mID)
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 256  # in pixels, diameter of circle of laser point, must be a multiple of either 4 or 16
    focus_point_centre_x, focus_point_centre_y = 1024, 1024  # in pixels, center point of the laser point
    n_windows_per_point = 1
    dwell = 0.2 # s
    # position z parameters
    z_center = 2.6
    span = 0.1
    N = 51

    z_start, z_end, z_range = cs.calc_sweep_range(z_center, span, N)
    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    z_motor.move_to(z_center)
    time.sleep(5) # move to rough middle, to have good auto exposure time
    cam, exposure_time, sg = pci.connect_cam_RF(roi, binning_amount)
    point_duration_s = exposure_time * n_windows_per_point
    print(f"Optimizing Z, estimate time to completion ~ {N * (dwell + point_duration_s) + 10:.2f}s")
    # here exact exposure time doesn't matter, as long as its constant throughout the range
    brightnesses, optimized_z_pos, max_brightness = find_best_z(cam, z_motor, z_range, dwell, point_duration_s, n_windows_per_point)
    cam.close()
    print(f"optimal z was found at z={optimized_z_pos:.4f}mm, with a brightness of {max_brightness:.2f}")
    plot_Z_dep_graph(z_range, brightnesses)

    ans = input("Move to best position? (Y/N): ").strip().lower()

    if ans == "y":
        z_motor.move_to(optimized_z_pos)
        time.sleep(0.5)
        print("Moved to optimized position and state saved.")
    else:
        print("Stayed at current position.")
    return

def bin_counts(counts_2D,binning_amount, x_space, y_space):
    # assume counts_2D has dims (x_width, y_with, freqs_len)
    # x,y width are equal and both a power of 2 times binning_amount

    # counts_binned = np.zeros((counts_2D.shape[0]//binning_amount,counts_2D.shape[1]//binning_amount,counts_2D.shape[2]))

    # counts_reshaped is of the shape: counts_x/bin, bin, counts_y/bin, bin, freq
    counts_reshaped = np.reshape(counts_2D,(counts_2D.shape[0]//binning_amount,binning_amount,counts_2D.shape[1]//binning_amount,binning_amount,counts_2D.shape[2]))
    # counts_binned is of the shape: counts_x/bin, counts_y/bin, freq, meaning we have to sum over the two bin axes
    counts_binned = counts_reshaped.sum(axis=1).sum(axis=2) # sum over the

    return counts_binned, x_space[0::binning_amount], y_space[0::binning_amount]


def vary_binning():
    # Do one image-ODMR, then do a variety of binnings
    # if I do exponential binning, then my overall image must be a power of 2 wide

    # params
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    n_bins = 6 # to bin 0,1,...n_bins-1 # note: n_bins must be at least 6
    focus_point_centre_x, focus_point_centre_y = 1024, 1024  # in pixels, center point of the laser point
    n_windows_per_point = 10 # n readouts to increase certainty without overexposing
    amp_dbm = -10 #anything bigger than -10 does nothing (Hayden)
    dwell =  0.001 # seconds - time between setting a frequency on fn generator and reading value
    n_iter = 1
    # frequency parameters
    f_center = 2.87e9 # Hz, generally near 2.87GHz
    span = 0.1e9 # Hz, range of frequencies to sample
    N = 101 # num points in the frequency space to sample
    max_peaks = 2 # TODO: make it able to detect the 4 peaks


    focus_point_size = 2**(n_bins-1)  # in physical (unbinned) pixels, diameter of circle of laser point
    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    # roi=(1,1,pci.camera_resolution,pci.camera_resolution)
    print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    cam, exposure_time, sg = pci.connect_cam_RF(roi, binning_amount)
    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")
    point_duration_s = exposure_time * n_windows_per_point

    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(0.1) # why sleep for a whole second? (previous was 1)
    try:
        counts_2D = wODMR.measure_odmr(cam, sg, freqs, dwell, point_duration_s, n_windows_per_point, n_iter)
    finally:
        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)
        cam.stop()

    print("")

    binned_snr_avg,ubinned_snr_avg = [], []
    binned_contrast_avg,ubinned_contrast_avg = [], []
    for bin in range(n_bins):
        print(f"binning+analyzing {2**bin}x{2**bin} area:")
        binned_counts, x_binned, y_binned = bin_counts(counts_2D, 2**bin, x_space, y_space)
        # print("Sweep done, now converting odmrs to B deltas")
        # B_Z_overall, problem_points = Lfit.counts_to_B_Z(x_binned, y_binned, binned_counts, freqs)
        # print("Conversion done, saving and plotting")
        # oPlot.save_2D_odmr_measurement(x_binned, y_binned, freqs, B_Z_overall, binned_counts)
        # oPlot.plot_dFreq_image(x_binned, y_binned, B_Z_overall)

        # print("Binning done, getting fit quality")
        all_snrs = np.zeros((x_binned.shape[0],y_binned.shape[0],max_peaks))
        all_contrasts = np.zeros((x_binned.shape[0],y_binned.shape[0],max_peaks))
        for x in range(x_binned.shape[0]):
            for y in range(y_binned.shape[0]):
                popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, binned_counts[x,y,:], max_peaks)
                snrs = Lfit.get_SNRs(baseline, binned_counts[x,y,:], freqs/10**9, popt)
                # Lfit.print_SNR(snrs, freqs)
                contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
                all_snrs[x,y,:] = snrs
                all_contrasts[x,y,:] = contrasts
        overall_avg_snr = ufloat(np.mean(snrs), np.std(snrs))
        overall_avg_contrast = ufloat(np.mean(contrasts),np.std(contrasts))
        binned_snr_avg.append(overall_avg_snr.n)
        binned_contrast_avg.append(overall_avg_contrast.n)
        ubinned_snr_avg.append(overall_avg_snr.s)
        ubinned_contrast_avg.append(overall_avg_contrast.s)

        print(f"Overall average SNR:{overall_avg_snr:.2u}, average contrast:{overall_avg_contrast*100:.2u}%")
    # TODO: save these values, and add a portion of read_ODMR which can read+display them
        # inner_dip_snr = snrs.mean(axis=0).mean(axis=0)[1:2]
        # inner_dip_contrast = contrasts.mean(axis=0).mean(axis=0)[1:2]

        # binned_snr_avg.append(inner_dip_snr)
        # binned_contrast_avg.append(inner_dip_contrast)
        # print(f"Overall average SNR:{inner_dip_snr:.4f}, average contrast:{inner_dip_contrast:.4f}")

    plot_binned_snr_contr(binned_contrast_avg,ubinned_contrast_avg, binned_snr_avg, ubinned_snr_avg, n_bins)

    return

def vary_exposure_time():
    # do a series of image-ODMRs, but all with the same camera settings EXCEPT for num windows per point
    # num windows should range from 1-100, maybe logarithmic scale again?

    # params
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 256  # in physical (unbinned) pixels, diameter of circle of laser point
    focus_point_centre_x, focus_point_centre_y = 1024, 1024  # in pixels, center point of the laser point
    amp_dbm = -10 #anything bigger than -10 does nothing (Hayden)
    dwell =  0.001 # seconds - time between setting a frequency on fn generator and reading value
    n_iter = 1
    n_windows = 8 # range exposure time from 2^(0,1,... n_windows-1)
    # frequency parameters
    f_center = 2.87e9 # Hz, generally near 2.87GHz
    span = 0.1e9 # Hz, range of frequencies to sample
    N = 201 # num points in the frequency space to sample
    max_peaks = 2 # TODO: make it able to detect the 4 peaks


    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    # roi=(1,1,pci.camera_resolution,pci.camera_resolution)
    print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    cam, exposure_time, sg = pci.connect_cam_RF(roi, binning_amount)
    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")

    snr_avg,usnr_avg = [], []
    contr_avg,ucontr_avg = [], []
    for window_exp in range(n_windows):
        n_windows_per_point = 2**window_exp
        point_duration_s = exposure_time * n_windows_per_point

        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
        time.sleep(0.1) # why sleep for a whole second? (previous was 1)
        try:
            counts_2D = wODMR.measure_odmr(cam, sg, freqs, dwell, point_duration_s, n_windows_per_point, n_iter)
        finally:
            cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)
            cam.stop()

        # B_Z_overall, problem_points = Lfit.counts_to_B_Z(x_space, y_space, counts_2D, freqs)
        # oPlot.save_2D_odmr_measurement(x_space, y_space, freqs, B_Z_overall, counts_2D)

        print(f"\nAnalyzing SNR&contrast from ODMR, esimate ~{focus_point_size**2/250:.2f}s")
        snrs, contrasts = Lfit.counts_to_SNR_contrast(x_space, y_space, counts_2D, freqs, max_peaks)
        # all_snrs = np.zeros((x_space.shape[0], y_space.shape[0], max_peaks))
        # all_contrasts = np.zeros((x_space.shape[0], y_space.shape[0], max_peaks))
        # for x in range(x_space.shape[0]):
        #     for y in range(y_space.shape[0]):
        #         popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts_2D[x, y, :],max_peaks)
        #         try:
        #             snrs = Lfit.get_SNRs(baseline, counts_2D[x, y, :], freqs / 10 ** 9, popt)
        #         except Exception as e:
        #             # oPlot.plot_odmr(freqs/10**9, counts_2D[x, y, :])
        #             oPlot.plot_fitted_data(freqs/10**9, counts_norm, fitted_norm)
        #             print("caught exception when getting SNRs: ", e, f"Plotted odmr of problematic pixel, indices: {x},{y}")
        #
        #         # Lfit.print_SNR(snrs, freqs)
        #         contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
        #         all_snrs[x, y, :] = snrs
        #         all_contrasts[x, y, :] = contrasts

        oPlot.save_2D_odmr_snr_contrast(x_space, y_space, freqs, snrs, contrasts, counts_2D)

        overall_avg_snr = ufloat(np.mean(snrs), np.std(snrs))
        overall_avg_contrast = ufloat(np.mean(contrasts), np.std(contrasts))
        print(f"Overall average SNR:{overall_avg_snr:.2u}, average contrast:{overall_avg_contrast * 100:.2u}%")
        snr_avg.append(overall_avg_snr.n)
        contr_avg.append(overall_avg_contrast.n)
        usnr_avg.append(overall_avg_snr.s)
        ucontr_avg.append(overall_avg_contrast.s)

    plot_exposure_snr_contr(contr_avg, ucontr_avg, snr_avg, usnr_avg, n_windows)
    return



def main():
    # do things, perhaps only one at a time or all at once idk:
    # 1: vary z, record the whole binned image at each step (with very small steps, <10um)
    # 2: keep exposure time constant, and vary the post-processed binning (probably in an exponential stepsize)
    #   for each pixel, do the ODMR and record SNR&contrast, then average them between all pixels, and record these numbers
    # 3: keep binning to 1x1, and vary total exposure time (change n_windows_per_point, force exposure time to a constant ~

    # plot_binned_snr_contr([1,2],[0.1,0.15],[0.5,1.2],[0.07,0.105], 2)

    # 1: optimizing z:
    # optimize_z()

    # 2:
    # vary_binning()

    # 3:
    vary_exposure_time()

    # TODO: create the combo exposure + binning variation, and see how things change



if __name__ == "__main__":
    main()