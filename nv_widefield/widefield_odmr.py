import pco
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time

from pco import Camera
from pyvisa import Resource

import connection_setup as cs


import os
import sys
sys.path.append(os.path.abspath(".."))
import nv_setup.cw_odmr.Lorentzian_fit as Lfit
import pco_cam_interface as pci

"""
Two parts: 
first use a single pixel of the camera sensor as a point ODMR sensor
then use the whole camera sensor, and do all the ODMRS for all the pixels
"""

# TODO:
#   Potential measurements:
#   vary the binning size from 1,2,4,8...2048 and compare contrasts/SNRs
#   vary the exposure time and compare contrasts/SNRs
#   vary both in a 2D array, and plot the contrasts/SNRs in a heatmap
#   Potential features:
#   do translation + widefield, to capture larger structures, atm we're limited to laser focus point size ~15um





def measure_odmr(cam, sg, freqs, dwell, point_duration_s, n_windows, n_iter: int = 1) -> np.ndarray:

    seen=0
    image = pci.read_image(cam,1)

    num_printouts = 10
    printout_factor = len(freqs)*n_iter*2 // num_printouts

    brightnesses = np.zeros((n_iter*2, image.shape[0], image.shape[1], freqs.size)) # should be n_iter*2 when reversing as well
    for i in range(n_iter):
        print("Iteration " + str(i))

        for j,f in enumerate(freqs):
            if (seen % printout_factor == 0):
                # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
                print(f"at freq {str(f / 10 ** 9)}GHz; {seen/(printout_factor*num_printouts)*100:.1f}% done")
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = pci.read_image(cam,n_windows)
            brightnesses[i,:,:,j]=image / point_duration_s/1000
            # pci.plot_image(image)
            seen+=1
        for j,f in enumerate(freqs[::-1]):
            if (seen % printout_factor == 0):
                # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
                print(f"at freq {str(f / 10 ** 9)}GHz; {seen/(printout_factor*num_printouts)*100:.1f}% done")
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = pci.read_image(cam,n_windows)
            # pci.plot_image(image)
            brightnesses[n_iter+i,:,:,j]=image / point_duration_s/1000
            seen+=1

    return np.sum(brightnesses,axis=0)/(n_iter*2)


def main():
    # params
    binning_amount = 4 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 128  # in physical (unbinned) pixels, diameter of circle of laser point
    focus_point_centre_x, focus_point_centre_y = 1000, 1110  # in pixels, center point of the laser point
    # TODO: maybe make use of 2D-gaussian to determine centre of focus point
    n_windows_per_point = 10 # n readouts to increase certainty without overexposing
    amp_dbm = -10 #anything bigger than -10 does nothing (Hayden)
    # Always use with 28V on the amplifier, amp_dbm ~30 is the lowest you can set while still seeing the zero-field dips
    # Larger amp means dips are more visible, but also get wider so you lose frequency resolution
    dwell =  0.001 # seconds - time between setting a frequency on fn generator and reading value
    n_iter = 1
    # frequency parameters
    f_center = 2.87e9 # Hz, generally near 2.87GHz
    span = 0.4e9 # Hz, range of frequencies to sample
    N = 101 # num points in the frequency space to sample


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
        counts_2D = measure_odmr(cam, sg, freqs, dwell, point_duration_s, n_windows_per_point, n_iter)
    finally:
        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)

    print("Sweep done, now converting odmrs to B deltas")
    B_Z_overall, problem_points = Lfit.counts_to_B_Z(x_space, y_space, counts_2D, freqs)
    # cs.plot_odmr(freqs, counts)
    # Save data in folder with its date
    # cs.save_point_odmr_measurement(counts, freqs)
    print("Conversion done, saving and plotting")
    cs.save_2D_odmr_measurement(x_space, y_space, freqs, B_Z_overall, counts_2D)
    cs.plot_image(x_space, y_space, B_Z_overall)



    # max_peaks = 2
    # popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
    # c0, c1 = popt[-2], popt[-1]
    # print(f"baseline: {c0} + {c1}f[GHz]")
    # Lfit.print_dip_params(popt)


    # try:
    #     Lfit.print_SNR(baseline, counts, freqs/10**9, popt)
    # except ValueError as e:
    #     # do nothing cuz printing snr didn't work
    #     print("getting SNR failed: " + str(e))
    # Lfit.plot_fitted_data(freqs/10**9, counts_norm, fitted_norm)



if __name__ == "__main__":
    main()