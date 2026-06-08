import numpy as np
import time
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

"""
A step in the direction of widefield imaging, by using the camera sensor, but binning
all the relevant pixels together into one "brightness" signal, to use in place of the
SPCM used in cw_odmr.py
"""


def measure_odmr(cam, sg, freqs, dwell, point_duration_s, n_windows, n_iter: int = 1) -> np.ndarray:
    # below is off by a factor of ~2??
    print(f"measuring binned ODMR with {n_iter} iterations, estimate time to completion ~{n_iter*2 * len(freqs) * (dwell + point_duration_s + 0.1):.0f}s")
    seen=0

    num_printouts = 10
    printout_factor = len(freqs)*n_iter*2 // num_printouts

    brightnesses = np.zeros((n_iter*2, freqs.size)) # should be n_iter*2 when reversing as well
    for i in range(n_iter):
        # print("Iteration " + str(i))

        brightness=[]
        for j,f in enumerate(freqs):
            if (seen % printout_factor == 0):
                print(f"at iteration {i} and freq {str(f / 10 ** 9)}GHz; {seen/(printout_factor*num_printouts)*100:.1f}% done")
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = pci.read_image(cam,n_windows)
            all_counts = pci.bin_image(image)
            # pci.plot_image(image)
            brightness.append(all_counts / point_duration_s/1000)
            seen+=1
        brightnesses[i]=brightness
        brightness=[]
        for j,f in enumerate(freqs[::-1]):
            if (seen % printout_factor == 0):
                # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
                print(f"at iteration {i} and freq {str(f / 10 ** 9)}GHz; {seen/(printout_factor*num_printouts)*100:.1f}% done")
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = pci.read_image(cam,n_windows)
            all_counts = pci.bin_image(image)
            # pci.plot_image(image)
            brightness.append(all_counts / point_duration_s/1000)
            seen+=1
        brightnesses[n_iter+i]=brightness[::-1]

    return np.sum(brightnesses,axis=0)/(n_iter*2)


def main():
    # params
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 200  # in pixels, width of image taken, must be a multiple of 16
    focus_point_centre_x, focus_point_centre_y = 2050, 860  # in pixels, center of the laser point

    n_windows_per_point = 1 # n readouts to increase certainty without overexposing
    amp_dbm = -15 #anything bigger than -10 does nothing (Hayden)
    # Always use with 28V on the amplifier, amp_dbm ~30 is the lowest you can set while still seeing the zero-field dips
    # Larger amp means dips are more visible, but also get wider so you lose frequency resolution

    dwell =  0.001 # seconds - time between setting a frequency on fn generator and reading value
    n_iter = 1
    # frequency parameters
    f_center = 2.87e9 # Hz, generally near 2.87GHz
    span = 0.15e9 # Hz, range of frequencies to sample
    N = 101 # num points in the frequency space to sample

    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    # roi=(1,1,pci.camera_resolution,pci.camera_resolution)
    print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")

    cam, sg = pci.connect_cam_RF(roi, binning_amount, 0.1)

    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")
    point_duration_s = cam.exposure_time * n_windows_per_point

    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(0.1) # why sleep for a whole second? (previous was 1)
    counts = pci.run_odmr_measurement(cam, sg, measure_odmr, (freqs, dwell, point_duration_s, n_windows_per_point, n_iter))

    oPlot.plot_odmr(freqs, counts)

    oPlot.save_point_odmr_measurement(counts, freqs)

    max_peaks = 2
    popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
    Lfit.print_dip_params(popt)


    try:
        snrs = Lfit.get_SNRs(baseline, counts, freqs/10**9, popt)
        Lfit.print_SNR(snrs, freqs/10**9)
    except ValueError as e:
        # do nothing cuz printing snr didnt work
        print("getting SNR failed: " + str(e))
    oPlot.plot_fitted_data(freqs/10**9, counts_norm, fitted_norm)



if __name__ == "__main__":
    main()