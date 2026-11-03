from typing import Any

import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import json

from numpy import dtype, ndarray, float64
from uncertainties import ufloat
import scipy.ndimage as ndi

import re

import Lorentzian_fit as Lfit
import helper_classes.pco_cam_interface as pci
import sys
sys.path.append(os.path.abspath("..."))
import nv_widefield.helper_classes.odmr_plotting as oPlot
import nv_widefield.helper_classes.file_interface as iFile


"""
this file is to read npy/z files created from each ODMR measurement 

Simply paste in the date and time of the file as it's stored in /NVCFM_Data/date and it will display it
change max_peaks for the lorentzian fitting for any ODMRs to be fitted


note: measurements previous to 2026-05-13 at 15:02 only have cps data, not frequency stored
^ this means you don't have any x data, and filetype is different (npy)
note: scanned odmrs previous to may 15, 2026 don't have the odmrs saved only the frequency deltas 
note: scanned odmrs previous to 2026-05-19 at 10:47 have freq delta data not B data

Point of all the above notes is that despite hiding a lot of these issues by checking for location manually,
it's possible there were mistakes in the code, so be aware of where things are stored
"""

# Params to change
date = "2026-11-03" # YYYY-MM-DD
time = "16-07-03"   # hh-mm_ss
max_peaks = 8
fft_point_measurement = False # when reading a point ODMR, if True then perform an FFT of it
subtract_blurred_ODMR = False # when reading a point ODMR, if True then subtract a gaussian blurred
subtract_gaussian_sigma = None # when subtracting a gaussian ODMR, this is the sigma to use (None or num samples)

def main():
    directory = os.path.join(iFile.desktop_dir, "NVCFM_Data", date)
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]  # list of strings of filenames
    match = [item for item in onlyfiles if "cw_odmr_" + time in item] # filter to only have files with '"cw_odmr" + date'

    if (len(match)==0):
        print("No matching files found, exiting")
        exit()

    # assume only one file has the exact same time to the second
    if match[0].startswith("cw_odmr"):
        handle_point_ODMR(date, time)

    elif match[0].startswith("scanned"):
        # is widefield
        handle_scanned_ODMR(date, time)

    elif match[0].startswith("snr_contr_"):
        handle_snr_contr(date, time)

    elif match[0].startswith("widefield"):
        # is widefield
        handle_widefield_ODMR(date, time)

    else:
        print("file has wrong prefix", match[0])


def handle_widefield_ODMR(date, time):
    B_Z_overall, counts_2D, freqs, x_points, y_points, params = iFile.read_widefield_odmr(date, time)
    widfield_user_interaction_loop(B_Z_overall, counts_2D, freqs, x_points, y_points, params=params)


def handle_snr_contr(date, time):

    contrasts, snrs = iFile.read_snr_contr(date, time)

    overall_avg_snr = ufloat(np.mean(snrs), np.std(snrs))
    overall_avg_contrast = ufloat(np.mean(contrasts), np.std(contrasts))
    print(f"Overall average SNR:{overall_avg_snr:.2u}, average contrast:{overall_avg_contrast * 100:.2u}%")




def handle_scanned_ODMR(date, time):



    counts_2D, freqs, x_points, y_points = iFile.read_scanned_ODMR(date, time)

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
            # oPlot.overwrite_2D_odmr_measurement(x_points, y_points, freqs, counts_2D, filepath + ".npz",
            #                                     print_saving=True, Bmap=B_Z_overall)

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

            counts = counts_2D[y_ind, x_ind]
            try:
                popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
                Lfit.print_dip_params(popt)
                # contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
                # Lfit.print_SNR(baseline, counts, freqs / 10 ** 9, popt)
                oPlot.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)
            except Exception as e:
                print("analyzing ODMR raised error:", e)
        except ValueError:
            print("Invalid format. Please enter two integers separated by a comma (e.g., '2, 4').")




def handle_point_ODMR(date, time):
    # single point measurement

    # filepath = os.path.join(desktop_dir, "NVCFM_Data", date, "cw_odmr_" + time)
    counts, freqs, params = iFile.read_point_ODMR()
    oPlot.plot_odmr(freqs, counts)

    try:
        popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks,Gaussian_subtraction=subtract_blurred_ODMR)

        # Lfit.print_dip_params(popt)
        snrs = Lfit.get_SNRs(baseline, counts, freqs / 10 ** 9, popt)

        contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)

        # Lfit.print_SNR(snrs, dip_Freqs)
        Lfit.print_contrast_snr_FWHM(contrasts, snrs, FWHMs, dip_Freqs)
        oPlot.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)
    except Exception as e:
        print("there was an error reading ODMR", e)

    if fft_point_measurement:
        fft, fft_freqs = perform_FFT(counts, freqs)

        plot_freq_filtered_ODMR(fft, fft_freqs, freqs)

    # if subtract_blurred_ODMR:
    #     oPlot.plot_odmr(freqs, Lfit.subtract_blurred_ODMR(freqs, counts, Gaussian_sigma=subtract_gaussian_sigma), title="ODMR with subtracted gaussian blurred")




def widfield_user_interaction_loop(B_Z_overall: ndarray[tuple[int, int], dtype[Any]] | Any, counts_2D, freqs,
                                   x_points, y_points, params={}):
    # --- User Interaction Loop ---
    print("\n" + "=" * 40)
    print("Spatial scan complete. Select a pixel to view its 1D ODMR spectrum.")
    print(f"Available X indices: 0 to {len(x_points) - 1}")
    print(f"Available Y indices: 0 to {len(y_points) - 1}")
    print("Type 'exit' to quit.")
    print("Type 'reanalyze' to re-analyze image from saved odmrs")
    print(
        "Type 'reanalyze bin _' to re-analyze image from saved odmrs with initial params derived from a binned version of the image")
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

            pci.plot_image(image, x_points, y_points, "Brightness from summing over all frequencies")
        elif user_input.lower().startswith("fft"):
            try:
                # Parse the input string into integers
                x_str, y_str = user_input[4:].split(",")
                x_ind = int(x_str.strip())
                y_ind = int(y_str.strip())

                # Boundary validation
                if (not ((0 <= x_ind < len(x_points)) and (0 <= y_ind < len(y_points)))):
                    print(f"Indices out of bounds! Keep X within [0, {len(x_points) - 1}] "
                          f"and Y within [0, {len(y_points) - 1}].")
                    continue

                fft, fft_freqs = perform_FFT(counts_2D[y_ind, x_ind], freqs)

            except ValueError as e:
                print("threw error", e)
        elif user_input.lower().startswith("reanalyze bin"):
            try:
                numbers = re.findall(r'\d+', user_input)
                numbers = [int(num) for num in numbers]

                print(f"Reanalyzing with initial guesses from a  {numbers[-1]}x{numbers[-1]} bin, will write new file")

                B_Z_overall, problem_points = Lfit.counts_to_B_Z_bin_init_params(x_points, y_points, counts_2D, freqs,
                                                                                 max_peaks=max_peaks,
                                                                                 binning_num=numbers[-1])
                if (len(problem_points) > 0):
                    print("following indices didn't autofit >=2 dips:")
                    print(problem_points)
                oPlot.plot_magnet_image(x_points, y_points, B_Z_overall)

                iFile.save_2D_odmr_measurement(x_points, y_points, freqs, B_Z_overall, counts_2D, **params)
                # iFile.overwrite_2D_odmr_measurement(x_points, y_points, freqs, counts_2D, filepath + ".npz",
                #                                     print_saving=True, Bmap=B_Z_overall)

            except Exception as e:
                print("Trying to parse fit with binning caused an error:", e)
                break
        elif user_input.lower() == "reanalyze":
            print("Reanalyzing, will write new file")

            B_Z_overall, problem_points = Lfit.counts_to_B_Z(x_points, y_points, counts_2D, freqs, max_peaks=max_peaks)
            if (len(problem_points) > 0):
                print("following indices didn't autofit >=2 dips:")
                print(problem_points)
            oPlot.plot_magnet_image(x_points, y_points, B_Z_overall)

            iFile.save_2D_odmr_measurement(x_points, y_points, freqs, B_Z_overall, counts_2D, **params)
            # iFile.overwrite_2D_odmr_measurement(x_points, y_points, freqs, counts_2D, filepath + ".npz",
            #                                     print_saving=True, Bmap=B_Z_overall)
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

                if  iFile.are_params_saved(date, time):
                    if params.get("post_processing_binning") is not None:
                        params["post_processing_binning"] = params["post_processing_binning"] * numbers[-1]
                    else:
                        params["post_processing_binning"] = numbers[-1]


                iFile.save_2D_odmr_measurement(x_binned, y_binned, freqs, B_Z_overall, binned_counts, **params)

                counts_2D = binned_counts
                x_points, y_points = x_binned, y_binned
            except KeyError as e:
                print("Trying to parse binning amount caused an error:", e)

            except Exception as e:
                print("Trying to parse binning amount caused an error:", e)
                break
        else:  # Look at the ODMR for one pixel
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
                counts = counts_2D[y_ind, x_ind]
                try:
                    popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks,
                                                                                   Gaussian_subtraction=True, Gaussian_sigma=subtract_gaussian_sigma)
                    print(f"B_Z was saved in the file as {B_Z_saved*1000:.1f} mT")
                    contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
                    snrs = Lfit.get_SNRs(baseline, counts, freqs, popt)
                    Lfit.print_contrast_snr_FWHM(contrasts, snrs, FWHMs, dip_Freqs)

                    oPlot.plot_odmr(freqs, counts, "raw ODMR data for (x,y)={},{}".format(x_ind, y_ind))
                    oPlot.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm, popt)
                except Exception as e:
                    print("analyzing data raised error:", e)
                    oPlot.plot_odmr(freqs, counts, "raw ODMR data for (x,y)={},{}".format(x_ind, y_ind))
            except ValueError as e:
                print("Invalid format. Please enter two integers separated by a comma (e.g., '2, 4'). Threw:", e)




def perform_FFT(counts, freqs: ndarray[tuple[Any, ...], dtype[float64]] | Any) -> tuple[Any, Any]:
    df = abs(freqs[1] - freqs[0])
    fft = scipy.fft.fft(counts)
    fft_freqs = scipy.fft.fftfreq(len(freqs), d=df)
    fft_shifted = scipy.fft.fftshift(fft)
    freqs_shifted = scipy.fft.fftshift(fft_freqs)

    magnitude = np.abs(fft_shifted)
    pos_mask = freqs_shifted > 0
    plot_freqs = freqs_shifted[pos_mask]
    plot_mag = magnitude[pos_mask]

    oPlot.plot_odmr_fft(plot_freqs, plot_mag)
    return fft, fft_freqs



def plot_freq_filtered_ODMR(fft, fft_freqs, freqs: ndarray[tuple[Any, ...], dtype[float64]] | Any):
    filter_mask = np.abs(fft_freqs) > 9e-9
    filter_mask2 = np.abs(fft_freqs) < 11e-9

    fft_filtered = fft.copy()
    fft_filtered[filter_mask & filter_mask2] *= 5

    counts_reconstructed = np.real(scipy.fft.ifft(fft_filtered))
    oPlot.plot_odmr(freqs, counts_reconstructed)


if __name__ == "__main__":
    main()