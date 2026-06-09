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
structure + analysis(lorentzian fitting + conversion to B) is quite similar to scanned_cw_odmr
"""

#   Potential features:
#   do translation + widefield, to capture larger structures, atm we're limited to laser focus point size ~15um


time0 = 0


def measure_odmr(cam, sg, freqs, dwell, n_windows, n_iter: int = 1) -> np.ndarray:
    point_duration_s = cam.exposure_time * n_windows
    print(f"measuring ODMR with {n_iter} iterations, estimate time to completion ~{datetime.timedelta(n_iter*2 * 1.1 * len(freqs) * (dwell + point_duration_s)):.2f}")

    seen=0
    image = pci.read_image(cam,1) # Throw out first image, it's often too bright
    # Note: one row of pixels is ~30% brighter than the rest, can't figure out why though

    num_printouts = 10
    printout_factor = len(freqs)*n_iter*2 // num_printouts
    brightnesses = np.zeros((n_iter*2, image.shape[0], image.shape[1], freqs.size)) # should be n_iter*2 when reversing as well
    for i in range(n_iter):
        # print("Iteration " + str(i))

        for j,f in enumerate(freqs):
            if (seen % printout_factor == 0):
                print(f"at t={time.time()-time0}s, iteration {i} and freq {str(f / 10 ** 9)}GHz; {seen/(printout_factor*num_printouts)*100:.0f}% done")
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = pci.read_image(cam,n_windows)
            brightnesses[i,:,:,j]=image / point_duration_s/1000
            # pci.plot_image(image)
            seen+=1
        for j,f in enumerate(freqs[::-1]):
            if (seen % printout_factor == 0):
                # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
                print(f"at t={time.time()-time0}s, iteration {i} and freq {str(f / 10 ** 9)}GHz; {seen/(printout_factor*num_printouts)*100:.0f}% done")
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
    focus_point_centre_x, focus_point_centre_y = 990,675 # in pixels, center point of the laser point
    # TODO: maybe make use of 2D-gaussian to determine centre of focus point automatically
    n_windows_per_point = 1 # n readouts to increase certainty without overexposing
    amp_dbm = -10 # from -30 to -10 work, higher gets more contrast but risks RF coupling, Amp at 28V
    dwell =  0.01 # seconds - time between setting a frequency on fn generator and reading value
    n_iter = 10
    # frequency parameters
    f_center = 2.87e9 # Hz, generally near 2.87GHz
    span = 0.3e9 # Hz, range of frequencies to sample
    N = 75 # num points in the frequency space to sample

    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    # roi=(1,1,pci.camera_resolution,pci.camera_resolution)
    print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")

    # cam, sg = pci.connect_cam_RF(roi, binning_amount) # can force a set exposure time if you want
    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")
    # point_duration_s = cam.exposure_time * n_windows_per_point

    # cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    # time.sleep(0.1) # why sleep for a whole second? (previous was 1)
    time0 = time.time()
    counts_2D = pci.run_odmr_measurement((roi, binning_amount), amp_dbm, measure_odmr, (freqs, dwell, n_windows_per_point, n_iter))



    print(f"Sweep done, now converting odmrs to B deltas, estimate time to completion ~{len(x_space)*len(y_space)}s")
    B_Z_overall, problem_points = Lfit.counts_to_B_Z(x_space, y_space, counts_2D, freqs)
    print("Conversion done, saving and plotting")
    oPlot.save_2D_odmr_measurement(x_space, y_space, freqs, B_Z_overall, counts_2D)
    oPlot.plot_dFreq_image(x_space, y_space, B_Z_overall)



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
    # oPlot.plot_fitted_data(freqs/10**9, counts_norm, fitted_norm)



if __name__ == "__main__":
    main()