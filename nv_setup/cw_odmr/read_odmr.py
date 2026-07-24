import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from uncertainties import ufloat

import re

import Lorentzian_fit as Lfit
import helper_classes.pco_cam_interface as pci
import sys
sys.path.append(os.path.abspath("..."))
import nv_widefield.helper_classes.odmr_plotting as oPlot


"""
this file is to read npy/z files created by cw_odmr

Simply paste in the date and time of the file as it's stored in /NVCFM_Data/date and it will display it
change max_peaks for the lorentzian fitting for any ODMRs to be fitted


note: measurements previous to 2026-05-13 at 15:02 only have cps data, not frequency stored
^ this means you don't have any x data, and filetype is different (npy)
note: scanned odmrs from may 15, 2026 don't have the odmrs saved only the frequency deltas 
note: scanned odmrs previous to 2026-05-19 at 10:47 have freq delta data not B data

Point of all the above notes is that despite hiding a lot of these issues by checking for location manually,
it's possible there were mistakes in the code, so be aware of where things are stored
"""

# Params to change
date = "2026-07-23" # YYYY-MM-DD
time = "10-36-21"   # hh-mm_ss
max_peaks = 6


desktop_dir = "C:\\Users\\NVCFM\\Desktop"
directory = os.path.join(desktop_dir, "NVCFM_Data", date)
onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]  # list of strings of filenames
match = [item for item in onlyfiles if "cw_odmr_" + time in item] # filter to only have files with '"cw_odmr" + date'

if (len(match)==0):
    print("No matching files found, exiting")
    exit()

# assume only one file has the exact same time to the second
if match[0].startswith("cw_odmr"):
    # single point measurement
    # Because its in YY-MM-DD we can do:
    is_freq_saved = False
    if (date > "2026-05-13"):
        is_freq_saved = True
    elif (date == "2026-05-13") & (time > "15-02-05"):
        is_freq_saved = True


    filepath = os.path.join(desktop_dir, "NVCFM_Data", date, "cw_odmr_" + time)
    if (is_freq_saved):
        data = np.load(filepath + ".npz")
        freqs, counts = data["x"], data["y"]
        oPlot.plot_odmr(freqs, counts)

        popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)

        # Lfit.print_dip_params(popt)
        snrs = Lfit.get_SNRs(baseline, counts, freqs/10**9, popt)

        contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)

        # Lfit.print_SNR(snrs, dip_Freqs)
        Lfit.print_contrast_snr_FWHM(contrasts, snrs, FWHMs, dip_Freqs)
        oPlot.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)
    else :
        data = np.load(filepath + ".npy")
        counts = data
        freqs = np.linspace(2.67, 3.07, len(counts) ,endpoint=True)
        print("Old file being read, frequency data is guessed and cannot be used for numerical analysis.")
        oPlot.plot_odmr(freqs, data)

elif match[0].startswith("scanned"):
    # is widefield

    # Below code is to determine if magnetic field data is saved or not
    is_B_saved = False
    if (date > "2026-05-19"):
        is_B_saved = True
    elif (date == "2026-05-19") & (time > "11-00-00"):
        is_B_saved = True

    filepath = os.path.join(desktop_dir, "NVCFM_Data", date, "scanned_cw_odmr_" + time)
    plt.figure(figsize=(8, 5))

    data = np.load(filepath + ".npz")


    x_points = data["x"]
    y_points = data["y"]
    freqs = data["f"]
    counts_2D = data["odmrs"]
    if is_B_saved:
        B_Z_overall = data["magnet"]
        oPlot.plot_magnet_image(x_points, y_points, B_Z_overall)
    else:

        fig, ax = plt.subplots(figsize=(8, 6))
        ext = [x_points.min(), x_points.max(), y_points.min(), y_points.max()]
        ax.set_xlabel("x space (mm)")
        ax.set_ylabel("x space (mm)")

        freq_deltas = data["deltas"]
        oPlot.plot_dFreq_image(x_points,y_points,freq_deltas)


    # --- User Interaction Loop ---
    print("\n" + "=" * 40)
    print("Spatial scan complete. Select a pixel to view its 1D ODMR spectrum.")
    print(f"Available X indices: 0 to {len(x_points) - 1}")
    print(f"Available Y indices: 0 to {len(y_points) - 1}")
    print("Type 'exit' to quit.")
    print("Type 'reanalyze' to re-analyze image from saved odmrs")
    print("=" * 40)

    while True:
        user_input = input("\nEnter pixel indices as 'X, Y' (e.g.: '5, 12'): ").strip()

        if user_input.lower() == "exit":
            print("Exiting interaction loop.")
            break
        if user_input.lower() == "reanalyze":
            print("Reanalyzing, will write new file")

            B_Z_overall, problem_points = Lfit.counts_to_B_Z(x_points, y_points, counts_2D, freqs, max_peaks=max_peaks)
            if (len(problem_points) > 0):
                print("following indices didn't autofit >=2 dips:")
                print(problem_points)
            oPlot.plot_magnet_image(x_points, y_points, B_Z_overall)

            oPlot.save_2D_odmr_measurement(x_points, y_points, freqs, B_Z_overall, counts_2D)

        try:
            # Parse the input string into integers
            x_str, y_str = user_input.split(",")
            x_ind = int(x_str.strip())
            y_ind = int(y_str.strip())

            # Boundary validation
            if (not ((0 <= x_ind < len(x_points))
                and (0 <= y_ind < len(y_points)))):
                print(f"Indices out of bounds! Keep X within [0, {len(x_points) - 1}] "
                      f"and Y within [0, {len(y_points) - 1}].")
                continue

            counts = counts_2D[x_ind,y_ind]
            popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
            Lfit.print_dip_params(popt)
            # contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
            # Lfit.print_SNR(baseline, counts, freqs / 10 ** 9, popt)
            oPlot.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)
        except ValueError:
            print("Invalid format. Please enter two integers separated by a comma (e.g., '2, 4').")


elif match[0].startswith("snr_contr_"):

    filepath = os.path.join(desktop_dir, "NVCFM_Data", date, "snr_contr_cw_odmr_" + time)
    plt.figure(figsize=(8, 5))

    data = np.load(filepath + ".npz")


    x_points = data["x"]
    y_points = data["y"]
    freqs = data["f"]
    counts_2D = data["odmrs"]
    snrs = data["snrs"]
    contrasts = data["contr"]

    overall_avg_snr = ufloat(np.mean(snrs), np.std(snrs))
    overall_avg_contrast = ufloat(np.mean(contrasts), np.std(contrasts))
    print(f"Overall average SNR:{overall_avg_snr:.2u}, average contrast:{overall_avg_contrast * 100:.2u}%")
elif match[0].startswith("widefield"):
    # is widefield

    # Below code is to determine if magnetic field data is saved or not
    is_B_saved = False
    if (date > "2026-05-19"):
        is_B_saved = True
    elif (date == "2026-05-19") & (time > "11-00-00"):
        is_B_saved = True

    filepath = os.path.join(desktop_dir, "NVCFM_Data", date, "widefield_cw_odmr_" + time)
    plt.figure(figsize=(8, 5))

    data = np.load(filepath + ".npz")


    x_points = data["x"]
    y_points = data["y"]
    freqs = data["f"]
    counts_2D = data["odmrs"]
    if is_B_saved:
        B_Z_overall = data["magnet"]
        # freq_deltas = np.zeros_like(B_Z_overall)
        oPlot.plot_magnet_image(x_points, y_points, B_Z_overall)
    else:

        fig, ax = plt.subplots(figsize=(8, 6))
        ext = [x_points.min(), x_points.max(), y_points.min(), y_points.max()]
        ax.set_xlabel("x space (mm)")
        ax.set_ylabel("x space (mm)")

        freq_deltas = data["deltas"]
        B_Z_overall = np.zeros_like(freq_deltas)
        oPlot.plot_dFreq_image(x_points,y_points,freq_deltas)


    # --- User Interaction Loop ---
    print("\n" + "=" * 40)
    print("Spatial scan complete. Select a pixel to view its 1D ODMR spectrum.")
    print(f"Available X indices: 0 to {len(x_points) - 1}")
    print(f"Available Y indices: 0 to {len(y_points) - 1}")
    print("Type 'exit' to quit.")
    print("Type 'reanalyze' to re-analyze image from saved odmrs")
    print("Type 'reanalyze bin _' to re-analyze image from saved odmrs with initial params derived from a binned version of the image")
    print("Type 'bin _' to re-analyze a binned version of the image (must be an even divisor of the image dimensions)")
    print("Type 'FFT x,y' to look at the fourier transform of an ODMR at pixel x,y")
    print("Type 'image' to sum over all the frequencies and get the brightness image ")
    print("=" * 40)

    while True:
        user_input = input("\nEnter pixel indices as 'X, Y' (e.g.: '5, 12'): ").strip()

        if user_input.lower() == "exit":
            print("Exiting interaction loop.")
            break
        elif user_input.lower().startswith("image"):
            image = counts_2D.sum(axis=2)

            pci.plot_image(image, x_points, y_points,"Brightness from summing over all frequencies")
        elif user_input.lower().startswith("fft"):
            try:
                # Parse the input string into integers
                x_str, y_str = user_input[4:].split(",")
                x_ind = int(x_str.strip())
                y_ind = int(y_str.strip())

                # Boundary validation
                if (not ((0 <= x_ind < len(x_points))
                    and (0 <= y_ind < len(y_points)))):
                    print(f"Indices out of bounds! Keep X within [0, {len(x_points) - 1}] "
                          f"and Y within [0, {len(y_points) - 1}].")
                    continue

                df = abs(freqs[1] - freqs[0])
                fft = scipy.fft.fft(counts_2D[x_ind,y_ind])
                fft_freqs = scipy.fft.fftfreq(len(freqs), d=df)
                fft_shifted = scipy.fft.fftshift(fft)
                freqs_shifted = scipy.fft.fftshift(fft_freqs)

                magnitude = np.abs(fft_shifted)
                pos_mask = freqs_shifted > 0
                plot_freqs = freqs_shifted[pos_mask]
                plot_mag = magnitude[pos_mask]

                oPlot.plot_odmr_fft(plot_freqs, plot_mag)


            except ValueError as e:
                print("threw error", e)
        elif user_input.lower().startswith("reanalyze bin"):
            try:
                numbers = re.findall(r'\d+', user_input)
                numbers = [int(num) for num in numbers]

                print(f"Reanalyzing with initial guesses from a  {numbers[-1]}x{numbers[-1]} bin, will write new file")

                B_Z_overall, problem_points = Lfit.counts_to_B_Z_bin_init_params(x_points, y_points, counts_2D, freqs,
                                                                 max_peaks=max_peaks, binning_num=numbers[-1])
                if (len(problem_points) > 0):
                    print("following indices didn't autofit >=2 dips:")
                    print(problem_points)
                oPlot.plot_magnet_image(x_points, y_points, B_Z_overall)

                oPlot.save_2D_odmr_measurement(x_points, y_points, freqs, B_Z_overall, counts_2D)
                # print(f"Trying to bin a {numbers[-1]}x{numbers[-1]} region")
                # binned_counts, x_binned, y_binned = pci.bin_counts(counts_2D, numbers[-1], x_points, y_points)
                # B_Z_overall, problem_points = Lfit.counts_to_B_Z(x_binned, y_binned, binned_counts, freqs,
                #                                                  max_peaks=max_peaks)
                # if (len(problem_points) > 0):
                #     print("following indices didn't autofit >=2 dips:")
                #     print(problem_points)
                # oPlot.plot_magnet_image(x_binned, y_binned, B_Z_overall)
                #
                # oPlot.save_2D_odmr_measurement(x_binned, y_binned, freqs, B_Z_overall, binned_counts)
                #
                # counts_2D = binned_counts
                # x_points, y_points = x_binned, y_binned

            except Exception as e:
                print("Trying to parse fit with binningl caused an error:", e)
                break
        elif user_input.lower() == "reanalyze":
            print("Reanalyzing, will write new file")

            B_Z_overall, problem_points = Lfit.counts_to_B_Z(x_points, y_points, counts_2D, freqs, max_peaks=max_peaks)
            if (len(problem_points) > 0):
                print("following indices didn't autofit >=2 dips:")
                print(problem_points)
            oPlot.plot_magnet_image(x_points, y_points, B_Z_overall)

            oPlot.save_2D_odmr_measurement(x_points, y_points, freqs, B_Z_overall, counts_2D)
        elif user_input.lower().startswith("bin"):
            try:
                numbers = re.findall(r'\d+', user_input)
                numbers = [int(num) for num in numbers]
                print(f"Trying to bin a {numbers[-1]}x{numbers[-1]} region")
                binned_counts, x_binned, y_binned = pci.bin_counts(counts_2D, numbers[-1], x_points, y_points)
                B_Z_overall, problem_points = Lfit.counts_to_B_Z(x_binned, y_binned, binned_counts, freqs,
                                                                 max_peaks=max_peaks)
                if (len(problem_points) > 0):
                    print("following indices didn't autofit >=2 dips:")
                    print(problem_points)
                oPlot.plot_magnet_image(x_binned, y_binned, B_Z_overall)

                oPlot.save_2D_odmr_measurement(x_binned, y_binned, freqs, B_Z_overall, binned_counts)

                counts_2D = binned_counts
                x_points, y_points = x_binned, y_binned

            except Exception as e:
                print("Trying to parse binning amount caused an error:", e)
                break
        else: # Look at the ODMR for one pixel
            try:
                # Parse the input string into integers
                x_str, y_str = user_input.split(",")
                x_ind = int(x_str.strip())
                y_ind = int(y_str.strip())

                # Boundary validation
                if (not ((0 <= x_ind < len(x_points))
                    and (0 <= y_ind < len(y_points)))):
                    print(f"Indices out of bounds! Keep X within [0, {len(x_points) - 1}] "
                          f"and Y within [0, {len(y_points) - 1}].")
                    continue

                B_Z_saved = B_Z_overall[y_ind, x_ind]
                counts = counts_2D[x_ind,y_ind]
                popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
                # print(f"B_Z was saved in the file as {B_Z_saved:.3}")
                # Lfit.print_dip_params(popt)
                # contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
                # Lfit.print_SNR(baseline, counts, freqs / 10 ** 9, popt)

                # contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
                # snrs = Lfit.get_SNRs(baseline, counts, freqs, popt)
                # Lfit.print_contrast_snr_FWHM(contrasts, snrs, FWHMs, dip_Freqs)

                oPlot.plot_odmr(freqs, counts, "raw ODMR data")
                oPlot.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)
            except ValueError:
                print("Invalid format. Please enter two integers separated by a comma (e.g., '2, 4').")


else:
    print("file has wrong prefix", match[0])
