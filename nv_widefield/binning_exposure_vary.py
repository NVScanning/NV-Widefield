
import time
import numpy as np
import scipy.ndimage as ndimage
import os
import sys
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
focus_point_centre_x, focus_point_centre_y = 850, 1130 # in pixels, center point of the laser point
amp_dbm = -10  # anything bigger than -10 does nothing (Hayden)
dwell = 0.001  # seconds - time between setting a frequency on fn generator and reading value
n_iter = 1
# frequency parameters
f_center = 2.87e9  # Hz, generally near 2.87GHz
span = 0.1e9  # Hz, range of frequencies to sample
N = 101  # num points in the frequency space to sample
max_peaks = 2

def find_best_z(cam, motor, z_range, dwell, point_duration_s, n_windows):
    seen = 0
    num_printouts = 5
    score = np.zeros(z_range.size)
    printout_factor = len(z_range) // num_printouts
    motor.move_to(z_range[0])
    time.sleep(5) # move to starting position, before connecting cam
    for z in z_range:
        if (seen % printout_factor == 0):
            print(f"at z={z:.4f}mm {seen/(printout_factor*num_printouts)*100:.1f}% done")
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
    z_previous_position = cs.get_motor_position(cs.z_mID)
    z_motor = cs.connect_motor(cs.z_mID)
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
    cam, exposure_time = pci.connect_cam(roi, binning_amount)


    point_duration_s = exposure_time * n_windows_per_point
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
        cam.stop()
        # cam.close()
        # time.sleep(5) # possibly cam needs time to stop itself properly
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
    focus_point_size = 2**(n_bins-1)  # in physical (unbinned) pixels, diameter of circle of laser point
    n_windows_per_point = 10 # n readouts to increase certainty without overexposing

    # max_peaks = 2 # TODO: make it able to detect the 4 peaks


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
        snrs, contrasts = Lfit.counts_to_SNR_contrast(x_binned, y_binned, binned_counts, freqs, max_peaks)

        overall_avg_snr = ufloat(np.mean(snrs), np.std(snrs))
        overall_avg_contrast = ufloat(np.mean(contrasts),np.std(contrasts))
        binned_snr_avg.append(overall_avg_snr.n)
        binned_contrast_avg.append(overall_avg_contrast.n)
        ubinned_snr_avg.append(overall_avg_snr.s)
        ubinned_contrast_avg.append(overall_avg_contrast.s)

        print(f"Overall average SNR:{overall_avg_snr:.2u}, average contrast:{overall_avg_contrast*100:.2u}%")
    # TODO: save these values, and add a portion of read_ODMR which can read+display them

    optPlot.plot_binned_snr_contr(binned_contrast_avg,ubinned_contrast_avg, binned_snr_avg, ubinned_snr_avg, n_bins)

    return

def vary_exposure_time():
    # do a series of image-ODMRs, but all with the same camera settings EXCEPT for num windows per point
    # num windows should range from 1-100, maybe logarithmic scale again?

    # params
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 256  # in physical (unbinned) pixels, diameter of circle of laser point
    n_windows = 7 # range exposure time from 2^(0,1,... n_windows-1)

    # max_peaks = 2 # TODO: make it able to detect the 4 peaks when a magnet is nearby


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

        # print("plotting the magnet image as a sanity check for poor fitting")
        # B_Z_overall, problem_points = Lfit.counts_to_B_Z(x_space, y_space, counts_2D, freqs)
        # oPlot.plot_magnet_image(x_space, y_space, B_Z_overall)

        # oPlot.save_2D_odmr_measurement(x_space, y_space, freqs, B_Z_overall, counts_2D)

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
    n_bins = 7 # to bin 0,1,...n_bins-1 # note: n_bins must be at least 6
    n_windows = 6 # range exposure time from 2^(0,1,... n_windows-1)
    focus_point_size = 2**(n_bins-1)  # in physical (unbinned) pixels, diameter of circle of laser point




    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    # roi=(1,1,pci.camera_resolution,pci.camera_resolution)
    print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    cam, exposure_time, sg = pci.connect_cam_RF(roi, binning_amount)
    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")

    snr_avg = np.zeros((n_windows, n_bins))
    contr_avg = np.zeros((n_windows, n_bins))
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

        for bin in range(n_bins):
            print(f"binning+analyzing {2 ** bin}x{2 ** bin} area with {2**window_exp} window(s), estimate time to completion ~{focus_point_size**2/200/(2**(bin*2)):.2f}s")

            binned_counts, x_binned, y_binned = bin_counts(counts_2D, 2**bin, x_space, y_space)
            snrs, contrasts = Lfit.counts_to_SNR_contrast(x_binned, y_binned, binned_counts, freqs, max_peaks)


            oPlot.save_2D_odmr_snr_contrast(x_binned, y_binned, freqs, snrs, contrasts, binned_counts)

            overall_avg_snr = np.mean(snrs)
            overall_avg_contrast = np.mean(contrasts)
            print(f"Overall average SNR:{overall_avg_snr:.2f}, average contrast:{overall_avg_contrast * 100:.2f}%")
            snr_avg[window_exp, bin] = overall_avg_snr
            contr_avg[window_exp, bin] = overall_avg_contrast

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
    optimize_z()

    # 2:
    # vary_binning()

    # 3:
    # vary_exposure_time()

    # 4:
    # vary_exposure_binning()


if __name__ == "__main__":
    main()