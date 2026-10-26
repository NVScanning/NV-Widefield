# from json.decoder import NaN
from typing import Any

import numpy as np
import scipy
import scipy.ndimage as ndi
from numpy import dtype, float64, ndarray, signedinteger
from numpy._typing import _32Bit, _64Bit
from scipy.optimize import curve_fit
import connection_setup as cs
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
import sys
import time

import helper_classes.odmr_plotting as oPlot
import helper_classes.pco_cam_interface as pci


# from nv_widefield.helper_classes.odmr_plotting import plot_fitted_data

"""
Has the lorentzian function definition, and various helper functions to fit lorentizans to an ODMR
Also has printout methods which format the params in an easily legible way for debugging

part of this fitting is to limit the frequency range where dips are expected to help reduce fits to noise, with the 
dip_range field, which is the min and max frequencies in GHz that will get a dip fitted

An additional component is to ignore dark pixels, below some threshold brightness, done with the brightnes_factor field
"""

dip_range = (2.75,2.99) # GHz
brightness_factor = 0.05 # unitless, 0<_<1, proportion of max brightness, below which pixels are automatically ignored
fine_sigma_GHz = 0.005 # sigma for gaussian blur in GHz
dip_separation_Ghz = 0.006 # minimal separation between autofitted dips in GHz
d2_prominence_factor = 0.15 # factor for the minimal prominence of a dip of ODMR 2nd derivative
background_sigma_GHz = 0.05 # sigma for gaussian blur of background in Hz
contrast_cutoff_ratio = 0.05

# ============================
# Fitting helper functions
# ============================

def multi_lorentzian(f, *params):
    """
    f: frequency array
    params: [A1, f1, g1, A2, f2, g2, ..., c0, c1]
        Ai: amplitude (negative for dip)
        fi: center frequency
        gi: HWHM (gamma)
        c0: baseline offset
        c1: baseline slope  (c0 + c1 * f)
    """
    n_peaks = (len(params) - 2) // 3
    c0 = params[-2]
    c1 = params[-1]

    y = c0 + c1 * f

    for i in range(n_peaks):
        A  = params[3*i]
        f0 = params[3*i + 1]
        g  = params[3*i + 2]
        y += A * g**2 / ((f - f0)**2 + g**2)

    return y


def guess_initial_params(freqs, vals, max_peaks=None, dip_range=(2.6, 3.2)):
    df = abs(freqs[1] - freqs[0])  # in GHz

    # 1. Use a strict, narrow smoothing window (~3 MHz) to preserve doublet inflections
    fine_sigma = max(1.0, fine_sigma_GHz / df) # sigma is in num samples, this is 3MHz converted to num samples
    filtered_vals = ndi.gaussian_filter(vals, sigma=fine_sigma)

    # Calculate second derivative for peak detection
    d1 = np.gradient(filtered_vals, df)
    filtered_d1 = ndi.gaussian_filter(d1, sigma=fine_sigma)
    d2 = np.gradient(filtered_d1, df)
    filtered_d2 = ndi.gaussian_filter(d2, sigma=fine_sigma)

    # oPlot.plot_odmr(freqs*10**9, filtered_vals, f"gaussian filtered ODMR with sigma={fine_sigma:.2f}") # print the filtered ODMR
    # oPlot.plot_odmr(freqs*10**9, filtered_d1, f"gaussian filtered ODMR derivative with sigma={fine_sigma:.2f}") # print the 1st derivative filtered ODMR
    # oPlot.plot_odmr(freqs*10**9, filtered_d2, f"gaussian filtered ODMR 2nd derivative with sigma={fine_sigma:.2f}") # print the 2nd derivative filtered ODMR

    max_d2 = np.max(filtered_d2)
    min_dist_pts = max(1, int(dip_separation_Ghz / df))  # ~6 MHz minimum separation between seeds

    peaks_d2, props_d2 = find_peaks(
        filtered_d2,
        prominence=d2_prominence_factor * max_d2,
        distance=min_dist_pts
    )

    freqs_peaks = freqs[peaks_d2]

    # Mask valid frequency range
    valid_mask = (freqs[peaks_d2] >= dip_range[0]) & (freqs[peaks_d2] <= dip_range[1])
    peaks = peaks_d2[valid_mask]



    peak_prominences,_,_ = scipy.signal.peak_prominences(-filtered_vals, peaks)
    peak_widths,_,_,_ = scipy.signal.peak_widths(-filtered_vals, peaks, rel_height=0.5)
    new_peaks = []

    for (peak, prominence, width) in zip(peaks, peak_prominences, peak_widths):
        fwhm_GHZ = width * df
        if fwhm_GHZ > 0.05:
            # TODO: change this threshold, because for high B its wide, but for low B it should be closer to 0.01
            split_offset_pts = max(2, int(fwhm_GHZ / df))  # fwhm splitting seed
            p1 = max(0, peak - split_offset_pts)
            p2 = min(len(freqs) - 1, peak + split_offset_pts)
            new_peaks.extend([p1, p2])
        else:
            new_peaks.append(peak)

    peaks = new_peaks


    # Fallback if no peaks detected
    if len(peaks) == 0:
        center_idx = len(freqs) // 2
        split_offset_pts = max(2, int(0.008 / df))
        peaks = np.array([center_idx - split_offset_pts, center_idx + split_offset_pts])

    # Baseline linear fit setup
    c1 = (filtered_vals[-10] - filtered_vals[10]) / (freqs[-10] - freqs[10])
    c0 = np.mean(vals) - c1 * freqs[len(freqs) // 2]

    # TODO: improve this baseline linear guess, because if the ends of the ODMRs go down in brightness
    #  (which they often do), then the estimates get all shit
    init_params = []
    peak_prominences,_,_ = scipy.signal.peak_prominences(-filtered_vals, peaks)
    peak_widths,_,_,_ = scipy.signal.peak_widths(-filtered_vals, peaks, rel_height=0.5)
    for (p, prominence, width) in zip(peaks, peak_prominences, peak_widths):
        f0 = freqs[p]
        if prominence == 0:
            amp = min(0,vals[p] - (c0 + c1 * f0)) # estimate from linear
        else:
            amp =-prominence
        if width == 0:
            gamma = 0.01
        else:
            # gamma = width * df/2
            gamma = 0.01
        init_params.extend([amp, f0, gamma])

    init_params.extend([c0, c1])

    init_params, peaks = prune_weak_dips(init_params, peaks,0.1)
    init_params, peaks = prune_asymmetric_dip_pairs(init_params, peaks, f_center=2.87)


    # # Sort and cull by prominence if now exceeding max_peaks
    # if max_peaks is not None and len(peaks) > max_peaks:
    #     # Calculate depth at peak indices relative to baseline
    #     depths = [abs(vals[p] - np.median(vals)) for p in peaks]
    #     top_idx = np.argsort(depths)[-max_peaks:]
    #     peaks = np.sort(peaks[top_idx])

    return np.array(init_params), peaks


def move_dips_to_local_min(vals, seed_idx, search_radius_pts):
    """
    Finds the index of the minimum value in `vals` within ±search_radius_pts of seed_idx.
    """
    start_idx = max(0, seed_idx - search_radius_pts)
    end_idx = min(len(vals), seed_idx + search_radius_pts + 1)

    # Argmin relative to the slice, converted back to global index
    refined_idx = start_idx + np.argmin(vals[start_idx:end_idx])
    return refined_idx

def prune_weak_dips(popt, peaks, contr_ratio=0.10):
    # Filters out dips with too low of a contrast, relative to the deepest dip

    # popt is the multi_lorentzian parameters
    # peaks is the indices of the guessed peaks
    # contr_ratio is the minimal contrast factor relative to the deepest dip that will be allowed

    n_peaks = len(peaks)
    contrasts, _, _ = get_dip_params(popt)
    contrasts = np.asarray(contrasts)
    max_contr = np.max(contrasts)

    minimal_contrast = contr_ratio * max_contr

    valid_mask = contrasts >= minimal_contrast
    valid_indices = np.where(valid_mask)[0]

    pruned_popt = prune_popt(popt, valid_indices)
    pruned_peaks = np.array(peaks)[valid_mask]

    return np.array(pruned_popt), pruned_peaks

def prune_wide_dips(popt, peaks, dip_range):
    # Filters out Lorentzian dips that are too far from the allowed range

    # popt is the multi_lorentzian parameters
    # peaks is the indices of the guessed peaks
    # freqs is the frequency array in GHz
    # dip_range is the lower and upper bound of allowed freqs, a tuple in GHz

    _, _, dip_Freqs = get_dip_params(popt)
    dip_Freqs = np.asarray(dip_Freqs)

    valid_mask = (dip_Freqs >= dip_range[0]) & (dip_Freqs <= dip_range[1])
    valid_indices = np.where(valid_mask)[0]

    pruned_popt = prune_popt(popt, valid_indices)
    pruned_peaks = np.array(peaks)[valid_mask]

    return np.array(pruned_popt), pruned_peaks


def prune_popt(popt, valid_indices: ndarray[tuple[Any, ...], dtype[signedinteger[_32Bit | _64Bit]]]) -> list[Any]:
    pruned_popt = []
    for idx in valid_indices:
        pruned_popt.extend(popt[3 * idx: 3 * idx + 3])
    # Append linear baseline parameters (slope, offset)
    pruned_popt.extend(popt[-2:])
    return pruned_popt


def prune_asymmetric_dip_pairs(popt, peaks, f_center=2.87):
    # Filters out fitted Lorentzian components that do not form symmetric pairs around f_center

    # popt is the multi_lorentzian parameters
    # peaks is the indices of the guessed peaks
    # f_center is the center frequency (2.87GHz) in GHz

    n_peaks = len(peaks)
    if n_peaks == 0:
        return np.asarray(popt, dtype=float), np.asarray(peaks)

    _, FWHMs, dip_Freqs = get_dip_params(popt)
    FWHMs = np.asarray(FWHMs)
    dip_Freqs = np.asarray(dip_Freqs)

    freq_margin = 0.01 # 10MHz margin to the center
    # Separate into lower and upper Zeeman transitions relative to 2.87 GHz
    lower_indices = np.where(dip_Freqs < f_center + FWHMs)[0]
    upper_indices = np.where(dip_Freqs > f_center)[0]

    valid_peak_indices = set()

    # Attempt to find a symmetric partner for each lower dip
    for l_idx in lower_indices:
        f_l = dip_Freqs[l_idx]
        fwhm_l = FWHMs[l_idx]

        best_partner = None
        min_asym = float("inf")

        for u_idx in upper_indices:
            if u_idx in valid_peak_indices:
                continue

            f_u = dip_Freqs[u_idx]
            fwhm_u = FWHMs[u_idx]

            # Measure center point deviation from 2.87 GHz
            center_freq = (f_l + f_u) / 2.0
            asym_dev = abs(center_freq - f_center)

            # Max allowed asymmetry is the avg FWHM of the pair
            allowed_threshold = (fwhm_l + fwhm_u)/2

            if asym_dev <= allowed_threshold and asym_dev < min_asym:
                min_asym = asym_dev
                best_partner = u_idx

        # If a valid symmetric partner was found, or its at 2.87GHz, retain them
        if best_partner is not None or (f_center - f_l) < fwhm_l:
            valid_peak_indices.add(l_idx)
            valid_peak_indices.add(best_partner)

    # Reconstruct popt and peaks maintaining original order
    valid_mask = np.array([i in valid_peak_indices for i in range(n_peaks)])
    valid_indices = np.where(valid_mask)[0]

    pruned_popt = prune_popt(popt, valid_indices)

    pruned_peaks = np.array(peaks)[valid_mask]

    return np.array(pruned_popt, dtype=float), pruned_peaks

def subtract_blurred_ODMR(freqs, R_vals, Gaussian_sigma=None):
    # Blur the R_vals array with a set gaussian blur, and subtract this blurred value from the original array
    # To ensure positive numbers and approximate relative contrast, the averaged value gets added back
    # returns an ODMR with similar brightness scale, but without background trends

    # freqs is array of freqs in Hz
    # R_vals is the one-dimensional ODMR array as a fn of frequency
    # Gaussian_sigma is the filtering sigma in num samples


    df = freqs[1] - freqs[0]
    freq_range = freqs[-1] - freqs[0]
    if Gaussian_sigma is not None:
        filtering_sigma = Gaussian_sigma
    else:
        # filtering_sigma = 0.02 * freq_range / df  # this is num samples, not freq space, sigma is 2% of frequency range
        filtering_sigma = background_sigma_GHz * 1e9 / df  # this is num samples, not freq space
        #

    filtered_counts = ndi.gaussian_filter(R_vals, sigma=filtering_sigma)

    return R_vals - filtered_counts + np.mean(filtered_counts)

def fit_odmr_multi_lorentzian(freqs, R_vals:np.ndarray, max_peaks=None, default_fit = None):
    # takes an ODMR, and returns the curvefitted params, covariance, and dip indices

    # freqs array of frequencies in GHz
    # R_vals is the one-dimensional ODMR array as a fn of frequency
    # max_peaks is the max number of peaks that will be fitted to teh ODMR
    # default_fit is combined p0 and peaks that will be used instead of the initial param guessing
    #   usually came from the analysis of a binned version of the measurement

    if default_fit is None:
        p0, peaks = guess_initial_params(freqs, R_vals, max_peaks=max_peaks)
    else:
        p0, peaks = default_fit

    if len(p0) == 2:
        raise Exception("No dips were fitted")

    n_peaks = (len(p0) - 2) // 3
    # if max_peaks is not None and (n_peaks < max_peaks):
    #     print(f"Guessed only {n_peaks} peaks out of {max_peaks} peaks at frequencies ", freqs[peaks])
    lower = []
    upper = []
    params = []
    for i in range(n_peaks):
        A0, f0, g0 = p0[3*i:3*i+3]
        lower += [-abs(A0*3), f0 - 0.005, 0.002]
        upper += [0, f0 + 0.005, min(3 * g0,0.025)]
        params.extend([f"A{i}",f"f{i}",f"g{i}"])


    # offset must be within 15% of maxcounts
    # slope may be +/- 10%
    lower += [p0[-2]-0.15*max(R_vals), p0[-1]-0.1*max(R_vals)]
    upper += [p0[-2]+0.15*max(R_vals), p0[-1]+0.1*max(R_vals)]
    params.extend(["offset","slope"])

    try:
        popt, pcov = curve_fit(
            multi_lorentzian, freqs, R_vals,
            p0=p0, bounds=(lower, upper), maxfev=3000
            ,ftol=0.001, xtol=0.001 # Note: these make convergence quicker, but lose accuracy
        )
        # check_bound_proximity(popt, (lower, upper), threshold=0.05, param_names=params) # useful to know if fitting code is flexible enough, gets annoying with non-ODMR sweeps

        popt, peaks = prune_wide_dips(popt, peaks, dip_range)
        popt, peaks = prune_weak_dips(popt, peaks, contrast_cutoff_ratio)
        popt, peaks = prune_asymmetric_dip_pairs(popt, peaks, f_center=2.87)


        popt, pcov = curve_fit(
            multi_lorentzian, freqs, R_vals,
            p0=popt, maxfev=3000
            # ,ftol=0.001, xtol=0.001 # Note: these make convergence quicker, but lose accuracy
        )

        popt, peaks = prune_wide_dips(popt, peaks, dip_range)
        # popt, peaks = prune_weak_dips(popt, peaks, contrast_cutoff_ratio)
        # popt, peaks = prune_asymmetric_dip_pairs(popt, peaks, f_center=2.87)
        return popt, pcov, peaks
    except Exception as e:
        # do smth to tell abt the error maybe idk
        #
        # print("Couldn't curve_fit, threw:", e, "Plotting ODMR")
        # print("\nCouldn't curve_fit, threw:", e, ". trying to fit peaks at frequencies", freqs[peaks],
        #       ". p0=",p0,")")
        # print("lower bounds:", lower)
        # print("upper bounds:", upper)
        # oPlot.plot_odmr(freqs*10**9, R_vals)
        return p0, np.zeros_like(p0), peaks


def check_bound_proximity(popt, bounds, threshold=0.05, param_names=None):
    # checks if any params are near the bounds, and prints out if so
    # returns which params are close to bound and how close in relative position

    # popt is the fitted params from curvefit
    # bounds is bounds passed into curvefit
    # threshold is how close to the bounds a param must be to get flagged
    # param_names is a list of strings to have clearer printouts of the params

    lb, ub = np.asarray(bounds[0]), np.asarray(bounds[1])
    popt = np.asarray(popt)

    # Compute relative position within bounded interval [0, 1]
    span = ub - lb
    rel_pos = (popt - lb) / span

    near_lower = rel_pos <= threshold
    near_upper = rel_pos >= (1 - threshold)
    flagged = near_lower | near_upper

    if np.any(flagged):
        for idx in np.where(flagged)[0]:
            name = param_names[idx] if param_names else f"popt[{idx}]"
            side = "LOWER" if near_lower[idx] else "UPPER"
            print(
                f"[Warning] {name} = {popt[idx]:.4e} is near {side} bound ({lb[idx]} to {ub[idx]}) | rel_pos = {rel_pos[idx]:.2%}")

    return flagged, rel_pos

def get_dip_params(popt):
    # converts the fitted params from curvefit into clearly defined contrast, FWHM, and dip center(dip_freq) for each dip
    popt = np.asarray(popt)
    c0, c1 = popt[-2], popt[-1]

    contrasts = []
    dip_Freqs = []
    FWHMs = []

    n_dips = (len(popt) - 2) // 3
    dip_params = popt[:3 * n_dips].reshape(n_dips, 3)

    for A, f0, gamma in dip_params:
        baseline_at_f0 = c0 + c1 * f0

        # Contrast (fraction & percent)
        C_frac = abs(A) / baseline_at_f0  # normalized dip depth
        # C_percent = C_frac * 100.0

        FWHM = 2.0 * gamma  # same unit as freqs (GHz)

        contrasts.append(C_frac)
        FWHMs.append(FWHM)
        dip_Freqs.append(f0)
    return contrasts, FWHMs, dip_Freqs # prop, GHz, GHz


def get_dip_uncertainties(pcov):
    # converts fitted covariance matrix from curvefit to the error in FWHMs and dip center(dip_freq) for each dip
    perr = np.sqrt(np.diag(pcov))
    n_dips = (len(perr) - 2) // 3

    err_FWHMs = []
    err_dip_Freqs = []

    for i in range(n_dips):
        idx_f0 = 3 * i + 1
        idx_gamma = 3 * i + 2

        err_dip_Freqs.append(perr[idx_f0])
        err_FWHMs.append(2.0 * perr[idx_gamma])

    return err_FWHMs, err_dip_Freqs


def print_SNR(snrs, freqs):
    # prints the SNR for each dip
    # freqs in GHz

    snr = np.mean(snrs)
    for (freq, snr_val) in zip(freqs, snrs):
        print(f"At frequency {freq:.4f} GHz: snr = {snr_val:.3}")

    print(f"SNR avg: {snr:.3}")

def print_contrast_snr(contrasts, snrs, dip_Freqs):
    # prints the contrasts, snrs, and dip center(dip_freq) in a clearly formatted way
    for (freq, snr_val, contr_val) in zip(dip_Freqs, snrs, contrasts):
        print(f"At frequency {freq:.4f} GHz: Contrast = {contr_val * 100:.3f}%, snr = {snr_val:.3}")
    print(f"SNR avg: {np.mean(snrs):.3}, contrast avg: {np.mean(contrasts)*100:.3}%")

def print_contrast_snr_FWHM(contrasts, snrs, FWHMs, dip_Freqs):
    # prints the contrasts, snrs, FWHMs, and dip center(dip_freq) in a clearly formatted way
    for (freq, snr_val, contr_val, FWHM) in zip(dip_Freqs, snrs, contrasts, FWHMs):
        print(f"At frequency {freq:.4f} GHz: FWHM = {FWHM * 1000:.1f} MHz, Contrast = {contr_val * 100:.3f}%, snr = {snr_val:.3}")
    print(f"SNR avg: {np.mean(snrs):.3f}, contrast avg: {np.mean(contrasts)*100:.3}%")

def print_dip_params(popt):
    # Prints FWHM, contrast for each dip
    # prints frequency delta for each adjacent pair of frequencies, and its equivalent B field
    # also returns the contrasts, FWHMs, and dip centers(dip_freqs)

    contrasts, FWHMs, dip_Freqs = get_dip_params(popt)

    # Print summary lines
    for (C, FWHM, freq) in zip(contrasts, FWHMs, dip_Freqs):
        # this tells us about T2 time (dephasing rate)
        print(f"At frequency {freq:.4f} GHz: FWHM = {FWHM * 1e3:.1f} MHz, Contrast = {C * 100:.3f}%")
    for i in range(len(dip_Freqs)-1):
        # this tells us abt magnetic field
        print(f"Frequency delta is {((dip_Freqs[i+1]-dip_Freqs[i])*1000):.3}MHz, equivalent to {(dip_Freqs[i+1]-dip_Freqs[i])/(2*cs.gamma_e):.3}T")
    return contrasts, FWHMs, dip_Freqs

def get_SNRs(baseline, counts, freqs, popt):
    # convert the fitted ODMR parameters to an SNR per-frequency

    # baseline is the linear fitted background term
    # counts is the brightness/coutns dependent on freqs
    # freqs is the frequency array in Hz (I think)
    # popt is the fitted parameters from curvefit

    noise_signal = counts - baseline
    c0, c1 = popt[-2], popt[-1]

    n_dips = (len(popt) - 2) // 3
    dip_params = popt[:3 * n_dips].reshape(n_dips, 3)
    off_mask = np.ones_like(freqs, dtype=bool)

    k_exclude = 1  # was previously 3.0
    for A, f0, gamma in dip_params:
        FWHM = 2.0 * gamma
        off_mask &= (np.abs(freqs - f0) > (k_exclude * FWHM))

    if np.count_nonzero(off_mask) < max(5, 0.1 * len(freqs)):

        raise ValueError("Off-resonance region too small. Decrease k_exclude or widen sweep range.")

    sigma = np.std(noise_signal[off_mask], ddof=1)
    # print(f"sigma of background : {sigma:.3} kcps, which is ~{sigma / c0 * 100:.3}%")

    snrs = []
    for i, (A, f0, gamma) in enumerate(dip_params, start=1):
        signal = abs(A)

        snrs.append(signal / sigma)
    return snrs


# ============================
# Lorentzian fitting (with baseline normalization)
# ============================
def analyze_data(freqs, counts, max_peaks,Gaussian_subtraction=False, Gaussian_sigma=None):
    # fits a multi-lorentzian to an ODMR, taking into account the max allowed dips, as well as any gaussian subtraction
    # returns the fitted params and covariance matrix, normalized counts, lorentzian fitted to the norm, and the baseline array

    # freqs is the frequency array in Hz
    # counts is the frequency-dependent brightness array
    # max_peaks is an integer describing the max number of dips allowed to be fitted
    # Gaussian_subtraction=True will subtract a gaussian blurred version of the ODMR from the raw signal for
    #   (hopefully) easier analysis
    # Gaussian_sigma is the sigma used in the gaussian blur, in number of samples, if None then a 2% of freq
    #   range sigma is used


    # expects freqs in Hz

    if Gaussian_subtraction:
        counts = subtract_blurred_ODMR(freqs, counts, Gaussian_sigma)
        # oPlot.plot_odmr(freqs, counts, "gaussian subtracted ODMR signal")

    popt, pcov, peaks = fit_odmr_multi_lorentzian(freqs / 10 ** 9, counts, max_peaks=max_peaks)
    if len(popt) == 2:
        raise Exception("No dips were fitted")

    fitted_counts = multi_lorentzian(freqs / 10 ** 9, *popt)

    # ---- Baseline from fit (off-resonance level) ----
    c0, c1 = popt[-2], popt[-1]
    baseline = c0 + c1 * freqs / 10 ** 9

    # ---- Normalized intensity (baseline = 1) ----
    # if there's a linear term in the counts, then this also removes that

    # if Gaussian_subtraction:
    #     # gaussian subtraction normalizes any background
    #     counts_norm = (counts-np.min(counts))/np.max(counts-np.min(counts))
    #     fitted_norm = (fitted_counts-np.min(fitted_counts))/np.max(fitted_counts-np.min(fitted_counts))
    # else:
    #     counts_norm = counts / baseline
    #     fitted_norm = fitted_counts / baseline

    counts_norm = counts / baseline
    fitted_norm = fitted_counts / baseline

    # print_SNR(baseline, counts, freqs, popt)
    return popt, pcov, counts_norm, fitted_norm, baseline



def params_to_B(popt):
    # convert the fitted lorentzian params to a magnetic field scalar
    delta_freq = 0
    _, _, dip_Freqs = get_dip_params(popt)
    # print_SNR(baseline, counts, freqs / 10 ** 9, popt)
    # oPlot.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)
    if (len(dip_Freqs) >= 2):
        # need exactly at least 2 dips to get the difference between the two
        # if >2 dips, assume the additional ones are the middle dips (irrelevant i think)
        delta_freq = dip_Freqs[-1] - dip_Freqs[0]
    # if you didn't get >=2 dips there's no delta, so return 0
    return delta_freq / (2 * cs.gamma_e)  # in T

def odmr_to_delta_freq(counts, freqs, max_peaks=4):
    # convert a counts array to a frequency delta between dips by fitting a lorentzian

    # counts is a 2D array of freq-dependent counts/brightness
    # freqs is the array of frequencies in Hz
    # max_peaks is the max number of dips that will be fitted to the ODMR

    delta_freq = 0
    # max_peaks = 2 # nominally 4, try 2 if the middle dips r too small to see
    try:
        popt, pcov, counts_norm, fitted_norm, baseline = analyze_data(freqs, counts, max_peaks, Gaussian_subtraction=True)
        _, _, dip_Freqs = get_dip_params(popt)
        # print_SNR(baseline, counts, freqs / 10 ** 9, popt)
        # oPlot.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)
        if (len(dip_Freqs) >= 2):
            # need exactly at least 2 dips to get the difference between the two
            # if >2 dips, assume the additional ones are the middle dips (irrelevant i think)
            delta_freq = dip_Freqs[-1] - dip_Freqs[0]
        if (len(dip_Freqs) == 0):
            delta_freq = np.nan
            # print("no fitted dips, B is NaN")
    except:
        delta_freq = np.nan
        # print("no fitted dips, B is NaN")

    return delta_freq

def counts_to_B_Z(x_points, y_points, counts_2D, freqs, max_peaks=4):
    # Convert a 3D array of freq-dependent counts over 2D space to a B scalar at each point in space
    # pixels which are too dark or couldn't fit get added to a problem_points array which is also returned

    # x_ and y_points are arrays for the the x and y axis of the FOV in mm
    # counts_2D is the 3D freq-dependent array of counts in 2D space
    # freqs is the frequencies in Hz
    # max_peaks is the max number of dips that are allowed to be fitted to an ODMR

    # TODO: convert this fitting to be on the GPU with JAXFit
    # TODO: find a way to have a background term that's shared between nearby pixels
    # TODO: if fitting problem, then save NaN rather than 0
    B_Z_overall = np.zeros((len(y_points), len(x_points)), dtype=float)
    problem_points = []

    t0 = time.time()
    # num_printouts = 10
    # printout_factor = len(x_points) * len(y_points) // num_printouts

    image = counts_2D.sum(axis=2)
    brightest_pixel = np.amax(image)

    for x_ind in range(len(x_points)):
        for y_ind in range(len(y_points)):
            cs.print_analysis_progress((x_ind * len(y_points) + y_ind), len(y_points) * len(x_points),time.time()-t0)
            # if ((x_ind*len(y_points) + y_ind) % printout_factor == 0):
            #     # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
            #     print(f"at position (x,y)=({x_ind},{y_ind}); {(x_ind*len(y_points) + y_ind)/(printout_factor*num_printouts)*100}% done")
            if image[y_ind, x_ind] < brightness_factor * brightest_pixel:
                problem_points.append((x_ind, y_ind))
                B_Z_overall[y_ind, x_ind] = np.nan
                # print(f"pixel ({x_ind},{y_ind}) has too low of signal, ignoring)")
            else:
                delta_freq = odmr_to_delta_freq(counts_2D[y_ind, x_ind], freqs, max_peaks=max_peaks)


                if delta_freq == np.nan:
                    # had problem fitting
                    problem_points.append((x_ind, y_ind))
                    B_Z_overall[y_ind, x_ind] = np.nan
                else:
                    B_Z = delta_freq / (2 * cs.gamma_e)  # in T
                    B_Z_overall[y_ind, x_ind] = B_Z

    sys.stdout.write(f"\r\033[KConverting to B_Z finished, took {time.time()-t0:.0f}s, "
                     f"and {len(problem_points)} points with a problem fitting\n")  # Clear progress bar
    sys.stdout.flush()
    return B_Z_overall, problem_points


def counts_to_B_Z_bin_init_params(x_points, y_points, counts_2D, freqs, max_peaks=4, binning_num=1):
    # Convert a 3D array of freq-dependent counts over 2D space to a B scalar at each point in space, but use
    # analysis of a binned measurement as the initial guess to (hopefully) improve analysis time
    # pixels which are too dark or couldn't fit get added to a problem_points array which is also returned

    # x_ and y_points are arrays for the the x and y axis of the FOV in mm
    # counts_2D is the 3D freq-dependent array of counts in 2D space
    # freqs is the frequencies in Hz
    # max_peaks is the max number of dips that are allowed to be fitted to an ODMR
    # binning_num is the sqrt(num pixels) to be binned together for the initial guesses
    #   binning_num x binning_num area gets binned together

    if len(x_points) % binning_num != 0 or len(y_points) % binning_num != 0:
        raise ValueError("binning_num is not an even divisior of axis sizes")

    B_Z_overall = np.zeros((len(x_points), len(y_points)), dtype=float)
    problem_points = []

    binned_counts, x_binned, y_binned = pci.bin_counts(counts_2D, binning_num, x_points, y_points)
    for x_ind in range(len(x_binned)):
        for y_ind in range(len(y_binned)):
            try:
                popt_bin, _, peaks_bin = fit_odmr_multi_lorentzian(freqs / 10 ** 9, binned_counts[x_ind, y_ind], max_peaks=max_peaks)
                # Use popt as initial params for fitment for each of the
                for x_in_bin in range(binning_num):
                    for y_in_bin in range(binning_num):
                        # Fit with popt
                        counts = counts_2D[x_ind*binning_num + x_in_bin, y_ind*binning_num + y_in_bin]/binning_num**2
                        popt, pcov, peaks = fit_odmr_multi_lorentzian(freqs / 10 ** 9, counts, default_fit=(popt_bin, peaks_bin))
                        B_Z = params_to_B(popt)
                        B_Z_overall[x_ind * binning_num + x_in_bin, y_ind * binning_num + y_in_bin] = B_Z
                        if B_Z == 0:
                            # had problem fitting
                            problem_points.append((x_ind * binning_num + x_in_bin, y_ind * binning_num + y_in_bin))
            except:
                for x_in_bin in range(binning_num):
                    for y_in_bin in range(binning_num):
                        # Fit without initial guess
                        counts = counts_2D[x_ind*binning_num + x_in_bin, y_ind*binning_num + y_in_bin]/binning_num**2
                        popt, pcov, peaks = fit_odmr_multi_lorentzian(freqs / 10 ** 9, counts)
                        B_Z = params_to_B(popt)
                        B_Z_overall[x_ind * binning_num + x_in_bin, y_ind * binning_num + y_in_bin] = B_Z
                        if B_Z == 0:
                            # had problem fitting
                            problem_points.append((x_ind * binning_num + x_in_bin, y_ind * binning_num + y_in_bin))
    return B_Z_overall, problem_points

def counts_to_SNR_contrast(x_points, y_points, counts_2D, freqs,max_peaks):
    # fit lorentzians to a 3D array of freq-dependent counts over 2D space and return the SNRs and contrasts for each pixel

    # x_ and y_points are arrays for the the x and y axis of the FOV in mm
    # counts_2D is the 3D freq-dependent array of counts in 2D space
    # freqs is the frequencies in Hz
    # max_peaks is the max number of dips that are allowed to be fitted to an ODMR

    all_snrs = np.zeros((x_points.shape[0], y_points.shape[0], max_peaks))
    all_contrasts = np.zeros((x_points.shape[0], y_points.shape[0], max_peaks))
    for x in range(x_points.shape[0]):
        for y in range(y_points.shape[0]):
            snrs, contrasts = ODMR_to_SNR_contr(counts_2D[y, x, :], freqs, max_peaks)
            all_snrs[x, y, :] = snrs
            all_contrasts[x, y, :] = contrasts

    return all_snrs, all_contrasts


def ODMR_to_SNR_contr(counts, freqs, max_peaks):
    # fit a lorentzian to freq-dependent counts and return the SNRs and contrasts of each dip

    # counts is thefreq-dependent array of counts/brightness
    # freqs is the frequencies in Hz
    # max_peaks is the max number of dips that are allowed to be fitted to an ODMR

    popt, pcov, counts_norm, fitted_norm, baseline = analyze_data(freqs, counts, max_peaks)
    contrasts, _, _ = get_dip_params(popt)
    snrs = get_SNRs(baseline, counts, freqs / 10 ** 9, popt)
    # print_SNR(snrs, dip_Freqs)

    return snrs, contrasts


def counts_to_SNR_contrast_init_params(x_points, y_points, counts_2D, freqs, max_peaks=4, binning_num=1):
    # fit lorentzians to a 3D array of freq-dependent counts over 2D space and return the SNRs and contrasts for
    # each pixel, but use analysis of a binned measurement for initial params to (hopefully) improve analysis time

    # x_ and y_points are arrays for the the x and y axis of the FOV in mm
    # counts_2D is the 3D freq-dependent array of counts in 2D space
    # freqs is the frequencies in Hz
    # max_peaks is the max number of dips that are allowed to be fitted to an ODMR
    # binning_num is the sqrt(num pixels) to be binned together for the initial guesses
    #   binning_num x binning_num area gets binned together


    if len(x_points) % binning_num != 0 or len(y_points) % binning_num != 0:
        raise ValueError("binning_num is not an even divisior of axis sizes")

    all_snrs = np.zeros((x_points.shape[0], y_points.shape[0], max_peaks))
    all_contrasts = np.zeros((x_points.shape[0], y_points.shape[0], max_peaks))

    binned_counts, x_binned, y_binned = pci.bin_counts(counts_2D, binning_num, x_points, y_points)
    for x_ind in range(len(x_binned)):
        for y_ind in range(len(y_binned)):
            try:
                popt_bin, _, peaks_bin = fit_odmr_multi_lorentzian(freqs / 10 ** 9, binned_counts[x_ind, y_ind], max_peaks=max_peaks)
                # Use popt as initial params for fitment for each of the
                for x_in_bin in range(binning_num):
                    for y_in_bin in range(binning_num):
                        # Fit with popt, somehow have to feed this into the
                        counts = counts_2D[x_ind*binning_num + x_in_bin, y_ind*binning_num + y_in_bin]/binning_num**2
                        popt, pcov, peaks = fit_odmr_multi_lorentzian(freqs / 10 ** 9, counts, default_fit=(popt_bin, peaks_bin))


                        fitted_counts = multi_lorentzian(freqs / 10 ** 9, *popt)
                        c0, c1 = popt[-2], popt[-1]
                        baseline = c0 + c1 * freqs / 10 ** 9
                        # counts_norm = counts / baseline
                        # fitted_norm = fitted_counts / baseline

                        contrasts, _, _ = get_dip_params(popt)
                        try:
                            snrs = get_SNRs(baseline, counts, freqs / 10 ** 9, popt)
                            # print_SNR(snrs, dip_Freqs)
                        except Exception as e:
                            print("getting SNRs returned an error", e)
                            snrs = np.zeros(max_peaks)
                            # print("Fitted dip frequencies at ", dip_Freqs, "GHz, with FWHMs ", FWHMs)
                            # oPlot.plot_fitted_data(freqs/10**9, counts_norm, fitted_norm)

                        all_snrs[x_ind * binning_num + x_in_bin, y_ind * binning_num + y_in_bin, :] = snrs
                        all_contrasts[x_ind * binning_num + x_in_bin, y_ind * binning_num + y_in_bin, :] = contrasts
            except:
                for x_in_bin in range(binning_num):
                    for y_in_bin in range(binning_num):
                        # Fit without initial params
                        counts = counts_2D[
                                     x_ind * binning_num + x_in_bin, y_ind * binning_num + y_in_bin] / binning_num ** 2
                        popt, pcov, peaks = fit_odmr_multi_lorentzian(freqs / 10 ** 9, counts)

                        fitted_counts = multi_lorentzian(freqs / 10 ** 9, *popt)
                        c0, c1 = popt[-2], popt[-1]
                        baseline = c0 + c1 * freqs / 10 ** 9
                        # counts_norm = counts / baseline
                        # fitted_norm = fitted_counts / baseline

                        contrasts, _, _ = get_dip_params(popt)
                        try:
                            snrs = get_SNRs(baseline, counts, freqs / 10 ** 9, popt)
                            # print_SNR(snrs, dip_Freqs)
                        except Exception as e:
                            print("getting SNRs returned an error", e)
                            snrs = np.zeros(max_peaks)
                            # print("Fitted dip frequencies at ", dip_Freqs, "GHz, with FWHMs ", FWHMs)
                            # oPlot.plot_fitted_data(freqs/10**9, counts_norm, fitted_norm)

                        all_snrs[x_ind * binning_num + x_in_bin, y_ind * binning_num + y_in_bin, :] = snrs
                        all_contrasts[x_ind * binning_num + x_in_bin, y_ind * binning_num + y_in_bin, :] = contrasts


    return all_snrs, all_contrasts
