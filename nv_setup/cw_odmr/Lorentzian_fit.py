# from json.decoder import NaN

import numpy as np
import scipy.ndimage as ndi
from numpy import dtype, float64, ndarray
from scipy.optimize import curve_fit
import connection_setup as cs
from scipy.signal import find_peaks, peak_widths
import sys
import time

import helper_classes.odmr_plotting as oPlot
import helper_classes.pco_cam_interface as pci


# from nv_widefield.helper_classes.odmr_plotting import plot_fitted_data


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

# def guess_initial_params(freqs, vals, max_peaks=None):
#     # freqs in GHz
#     max_val = max(vals)
#     df = abs(freqs[1]-freqs[0])
#     # peaks, props = find_peaks(-vals,prominence=0.005*max_val, distance=max(1,0.005//(freqs[1]-freqs[0])))
#     peaks, props = find_peaks(-vals, prominence=0.0004*max_val, distance=max(1,0.005//df))
#     # peaks, props = find_peaks(-vals, prominence=0.0002*max_val, distance=max(1,0.005//df))
#
#     # filtered_vals = ndi.gaussian_filter(vals, sigma=1.5)
#     # oPlot.plot_odmr(freqs*10**9, filtered_vals) # print the filtered values to know what kind of prominence to expect
#     # peaks, props = find_peaks(-filtered_vals, prominence=0.0003*max_val, distance=max(1,0.005//df))
#
#
#     # print(f"\nFind_peaks found {len(peaks)} peaks, at frequencies: {freqs[peaks]}")
#
#
#     max_prominence = (max(vals)-min(vals))/max(vals) # use maybe 1/5-1/10th of this has the minimal prominence
#
#     if max_peaks is not None and len(peaks) > max_peaks:
#         # Sort by prominences
#         prominences = props["prominences"]
#         top_idx = np.argsort(prominences)[-max_peaks:]
#         peaks = peaks[top_idx]
#         peaks = peaks[np.argsort(peaks)]
#
#     if max_peaks == 2 and len(peaks) == 1:
#         # print("Warning: Unresolved overlapping doublet detected. Manually splitting peak seeds")
#         # sole_peak_idx = peaks[0]
#         sole_peak_idx = len(freqs) // 2 # assume centered around 2.87GHz
#         # Calculate index offset corresponding to roughly 8 MHz splitting
#         df = freqs[1]-freqs[0]
#         split_offset_idx = max(3, int(0.005 / df))
#
#         # Seed two distinct peaks symmetrically around the center trough
#         peak1 = max(0, sole_peak_idx - split_offset_idx)
#         peak2 = min(len(freqs) - 1, sole_peak_idx + split_offset_idx)
#         peaks = np.array([peak1, peak2])
#
#
#         # detected_peak_idx = peaks[0]
#         # # Seed two distinct peaks symmetrically around the detected dip center (approx 6-8 MHz split)
#         # split_offset_idx = max(2, int(0.004 / df))
#         #
#         # peak1 = max(0, detected_peak_idx - split_offset_idx)
#         # peak2 = min(len(freqs) - 1, detected_peak_idx + split_offset_idx)
#         # peaks = np.array([peak1, peak2])
#         print(f"Manually split 1 dip into {max_peaks} at freqs {freqs[peaks]} GHz")
#     elif len(peaks) == 0:
#         center_idx = len(freqs) // 2
#         split_offset_idx = max(2, int(0.006 / df))
#         peaks = np.array([center_idx - split_offset_idx, center_idx + split_offset_idx])
#         print(f"No peaks resolved. Defaulting to blind center seeds: {freqs[peaks]} GHz")
#
#     # print(f"After culling lowest peaks, using {len(peaks)} initial peaks, at frequencies: {freqs[peaks]}GHz")
#     filtered_vals = ndi.gaussian_filter(vals, sigma=10)
#     c1 = (filtered_vals[-1]-filtered_vals[0])/(freqs[-1]-freqs[0])                # slope value
#     c0 = filtered_vals[0] - c1*freqs[0]                                           # offset
#
#     init_params = []
#     # results_half = peak_widths(-vals, peaks, rel_height=0.5)
#     # widths = results_half[0]
#     #
#     #
#     # for i, p in enumerate(peaks):
#     #     f0 = freqs[p]
#     #     amp = -abs(vals[p] - c0)
#     #     w_idx = widths[i]
#     #     fwhm = w_idx * df
#     #     # gamma = df*3
#     #     gamma = fwhm/2
#     #     init_params.extend([amp, f0, gamma])
#     #
#     # init_params.extend([c0, c1])
#     # return np.array(init_params), peaks
#
#     try:
#         results_half = peak_widths(-vals, peaks, rel_height=0.5) # this could break for peaks I create manually
#         widths = results_half[0]
#     except Exception:
#         # Fallback if peak_widths fails on highly overlapping, noisy regions
#         widths = [max(3, int(0.015 / df))] * len(peaks)
#
#     for i, p in enumerate(peaks):
#         f0 = freqs[p]
#         amp = -abs(vals[p] - (c0 + c1 * f0))
#         fwhm = widths[i] * df
#
#         # Constrain initial gamma guess to prevent broad overlapping seeds from starting too wide
#         gamma = np.clip(fwhm / 2, 0.003, 0.015)
#         init_params.extend([amp, f0, gamma])
#
#     init_params.extend([c0, c1])
#     return np.array(init_params), peaks


def guess_initial_params(freqs, vals, max_peaks=None):
    # freqs in GHz
    df = abs(freqs[1]-freqs[0])

    filtering_sigma = 1


    filtered_vals = ndi.gaussian_filter(vals, sigma=filtering_sigma)
    # oPlot.plot_odmr(freqs*10**9, filtered_vals, f"gaussian filtered ODMR with sigma={filtering_sigma}") # print the 2nd derivative values to know what kind of prominence to expect


    # find noise of background
    edge_pts = max(5, int(len(vals) * 0.05))

    d1 = np.gradient(filtered_vals, df)
    # print(d1)
    # oPlot.plot_odmr(freqs*10**9, d1) # print the 1st derivative values to know what kind of prominence to expect
    filtered_d1 = ndi.gaussian_filter(d1, sigma=filtering_sigma)
    # oPlot.plot_odmr(freqs*10**9, filtered_d1, "gaussian filtered ODMR derivative") # print the 2nd derivative values to know what kind of prominence to expect
    d2 = np.gradient(filtered_d1, df) # why a negative sign?
    # oPlot.plot_odmr(freqs*10**9, d2) # print the 2nd derivative values to know what kind of prominence to expect

    filtered_d2 = ndi.gaussian_filter(d2, sigma=filtering_sigma)
    # oPlot.plot_odmr(freqs*10**9, filtered_d2, "gaussian filtered ODMR 2nd derivative") # print the 2nd derivative values to know what kind of prominence to expect
    max_d2 = max(d2)
    max_prominence = (max_d2-min(d2)) # use maybe 1/5-1/10th of this has the minimal prominence

    noise_std = np.std(d2[:edge_pts] - filtered_d2[:edge_pts])
    min_distance_pts = max(1, int(0.005 / df))  # ~4 MHz minimum peak separation
    peaks_d2, props_d2 = find_peaks(
        filtered_d2,
        # prominence=max(0.1*max_prominence, 1.5 * noise_std),
        prominence=0.2*max_prominence,
        distance=min_distance_pts
    )

    peaks, props = peaks_d2, props_d2
    print(f"\nFind_peaks on second derivative found {len(peaks)} peaks, at frequencies: {freqs[peaks]}")



    if max_peaks is not None and len(peaks) > max_peaks:
        # Sort by prominences
        prominences = props["prominences"]
        top_idx = np.argsort(prominences)[-max_peaks:]
        peaks = peaks[top_idx]
        peaks = peaks[np.argsort(peaks)]

    if max_peaks == 2 and len(peaks) == 1:
        # print("Warning: Unresolved overlapping doublet detected. Manually splitting peak seeds")
        # sole_peak_idx = peaks[0]
        sole_peak_idx = len(freqs) // 2 # assume centered around 2.87GHz
        # Calculate index offset corresponding to roughly 8 MHz splitting
        df = freqs[1]-freqs[0]
        split_offset_idx = max(3, int(0.005 / df))

        # Seed two distinct peaks symmetrically around the center trough
        peak1 = max(0, sole_peak_idx - split_offset_idx)
        peak2 = min(len(freqs) - 1, sole_peak_idx + split_offset_idx)
        peaks = np.array([peak1, peak2])


        print(f"Manually split 1 dip into {max_peaks} at freqs {freqs[peaks]} GHz")
    elif len(peaks) == 0:
        center_idx = len(freqs) // 2
        split_offset_idx = max(2, int(0.006 / df))
        peaks = np.array([center_idx - split_offset_idx, center_idx + split_offset_idx])
        print(f"No peaks resolved. Defaulting to blind center seeds: {freqs[peaks]} GHz")

    # print(f"After culling lowest peaks, using {len(peaks)} initial peaks, at frequencies: {freqs[peaks]}GHz")
    filtered_vals = ndi.gaussian_filter(vals, sigma=2)
    c1 = (filtered_vals[-1]-filtered_vals[0])/(freqs[-1]-freqs[0])                # slope value
    c0 = filtered_vals[0] - c1*freqs[0]                                           # offset

    init_params = []
    try:
        results_half = peak_widths(-vals, peaks, rel_height=0.5) # this could break for peaks I create manually
        widths = results_half[0]
    except Exception:
        # Fallback if peak_widths fails on highly overlapping, noisy regions
        widths = [max(5, int(0.015 / df))] * len(peaks)

    for i, p in enumerate(peaks):
        f0 = freqs[p]
        amp = -abs(vals[p] - (c0 + c1 * f0))
        fwhm = widths[i] * df

        # Constrain initial gamma guess to prevent broad overlapping seeds from starting too wide
        gamma = np.clip(fwhm / 2, 0.005, 0.015) # limit to 5-15MHz
        init_params.extend([amp, f0, gamma])
        # print(f"Dip {i}: amp={amp:.0f}, f0={f0:.4f}Ghz, gamma={gamma}")

    init_params.extend([c0, c1])
    return np.array(init_params), peaks

def fit_odmr_multi_lorentzian(freqs, R_vals:np.ndarray, max_peaks=None, default_fit = None):
    # freqs in GHz
    if default_fit is None:
        p0, peaks = guess_initial_params(freqs, R_vals, max_peaks=max_peaks)
    else:
        p0, peaks = default_fit
    n_peaks = (len(p0) - 2) // 3
    # if max_peaks is not None and (n_peaks < max_peaks):
    #     print(f"Guessed only {n_peaks} peaks out of {max_peaks} peaks at frequencies ", freqs[peaks])
    lower = []
    upper = []
    for i in range(n_peaks):
        A0, f0, g0 = p0[3*i:3*i+3]
        lower += [-abs(A0*1.5), f0 - 0.005, 0.001]
        upper += [-abs(A0*0.5), f0 + 0.005, 3 * g0] # max HWHF is 40MHz

    # lower += [R_vals.min() - 1, -0.05*max(R_vals)/(freqs[-1]-freqs[0])]
    # upper += [R_vals.max()*1.1,  0.05*max(R_vals)/(freqs[-1]-freqs[0])]
    # lower += [p0[-2]*0.8, -0.05*max(R_vals)/(freqs[-1]-freqs[0])]
    # upper += [p0[-2]*1.2,  0.05*max(R_vals)/(freqs[-1]-freqs[0])]
    lower += [p0[-2]-0.15*max(R_vals), p0[-1]-0.1*max(R_vals)]
    upper += [p0[-2]+0.15*max(R_vals), p0[-1]+0.1*max(R_vals)]

    try:
        popt, pcov = curve_fit(
            multi_lorentzian, freqs, R_vals,
            p0=p0, bounds=(lower, upper), maxfev=3000
            # ,ftol=0.001, xtol=0.001) # Note: these make convergence quicker, but lose accuracy
        )
        return popt, pcov, peaks
    except Exception as e:
        # print("Couldn't curve_fit, threw:", e, "Plotting ODMR")
        print("\nCouldn't curve_fit, threw:", e, ". trying to fit peaks at frequencies", freqs[peaks],
              ". p0=",p0,")")
        print("lower bounds:", lower)
        print("upper bounds:", upper)
        oPlot.plot_odmr(freqs*10**9, R_vals)
        return p0, np.zeros_like(p0), peaks


def get_dip_params(popt):
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
    return contrasts, FWHMs, dip_Freqs # %, GHz, GHz

def print_contrast_snr(contrasts, snrs, dip_Freqs):
    for (freq, snr_val, contr_val) in zip(dip_Freqs, snrs, contrasts):
        print(f"At frequency {freq:.4f} GHz: Contrast = {contr_val * 100:.3f}%, snr = {snr_val:.3}")
    print(f"SNR avg: {np.mean(snrs):.3}, contrast avg: {np.mean(contrasts)*100:.3}%")
def print_contrast_snr_FWHM(contrasts, snrs, FWHMs, dip_Freqs):
    for (freq, snr_val, contr_val, FWHM) in zip(dip_Freqs, snrs, contrasts, FWHMs):
        print(f"At frequency {freq:.4f} GHz: FWHM = {FWHM * 1000:.1f} MHz, Contrast = {contr_val * 100:.3f}%, snr = {snr_val:.3}")
    print(f"SNR avg: {np.mean(snrs):.3f}, contrast avg: {np.mean(contrasts)*100:.3}%")

def print_dip_params(popt):
    # Prints FWHM, contrast, and frequency delta
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

def print_SNR(snrs, freqs):
    # expect freqs of the dips in GHz

    snr = np.mean(snrs)
    for (freq, snr_val) in zip(freqs, snrs):
        print(f"At frequency {freq:.4f} GHz: snr = {snr_val:.3}")

    print(f"SNR avg: {snr:.3}")

# ============================
# Lorentzian fitting (with baseline normalization)
# ============================
def analyze_data(freqs, counts, max_peaks):
    # expects freqs in Hz
    popt, pcov, peaks = fit_odmr_multi_lorentzian(freqs / 10 ** 9, counts, max_peaks=max_peaks)
    fitted_counts = multi_lorentzian(freqs / 10 ** 9, *popt)

    # ---- Baseline from fit (off-resonance level) ----
    c0, c1 = popt[-2], popt[-1]
    baseline = c0 + c1 * freqs / 10 ** 9

    # ---- Normalized intensity (baseline = 1) ----
    # if there's a linear term in the counts, then this also removes that
    counts_norm = counts / baseline
    fitted_norm = fitted_counts / baseline

    # print_SNR(baseline, counts, freqs, popt)
    return popt, pcov, counts_norm, fitted_norm, baseline



def params_to_B(popt):
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
    delta_freq = 0
    # max_peaks = 2 # nominally 4, try 2 if the middle dips r too small to see
    popt, pcov, counts_norm, fitted_norm, baseline = analyze_data(freqs, counts, max_peaks)
    _, _, dip_Freqs = get_dip_params(popt)
    # print_SNR(baseline, counts, freqs / 10 ** 9, popt)
    # oPlot.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)
    if (len(dip_Freqs) >= 2):
        # need exactly at least 2 dips to get the difference between the two
        # if >2 dips, assume the additional ones are the middle dips (irrelevant i think)
        delta_freq = dip_Freqs[-1] - dip_Freqs[0]
    # if you didn't get >=2 dips there's no delta, so return 0
    return delta_freq

def counts_to_B_Z(x_points, y_points, counts_2D, freqs, max_peaks=4):

    # TODO: convert this fitting to be on the GPU with JAXFit
    # TODO: find a way to have a background term that's shared between nearby pixels
    # TODO: if fitting problem, then save NaN rather than 0
    B_Z_overall = np.zeros((len(x_points), len(y_points)), dtype=float)
    problem_points = []

    t0 = time.time()
    # num_printouts = 10
    # printout_factor = len(x_points) * len(y_points) // num_printouts

    for x_ind in range(len(x_points)):
        for y_ind in range(len(y_points)):
            cs.print_analysis_progress((x_ind * len(y_points) + y_ind), len(y_points) * len(x_points))
            # if ((x_ind*len(y_points) + y_ind) % printout_factor == 0):
            #     # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
            #     print(f"at position (x,y)=({x_ind},{y_ind}); {(x_ind*len(y_points) + y_ind)/(printout_factor*num_printouts)*100}% done")
            delta_freq = odmr_to_delta_freq(counts_2D[x_ind,y_ind], freqs, max_peaks=max_peaks)
            if delta_freq == 0:
                # had problem fitting
                problem_points.append((x_ind, y_ind))
                B_Z_overall[x_ind,y_ind]=np.nan
            else:
                B_Z = delta_freq / (2 * cs.gamma_e)  # in T
                B_Z_overall[x_ind,y_ind]=B_Z

    sys.stdout.write(f"\r\033[KConverting to B_Z finished, took {time.time()-t0:.0f}s\n")  # Clear progress bar
    sys.stdout.flush()
    return B_Z_overall, problem_points


def counts_to_B_Z_bin_init_params(x_points, y_points, counts_2D, freqs, max_peaks=4, binning_num=1):
    if len(x_points) % binning_num != 0 or len(y_points) % binning_num != 0:
        raise ValueError("binning_num is not an even divisior of axis sizes")

    B_Z_overall = np.zeros((len(x_points), len(y_points)), dtype=float)
    problem_points = []

    binned_counts, x_binned, y_binned = pci.bin_counts(counts_2D, binning_num, x_points, y_points)
    for x_ind in range(len(x_binned)):
        for y_ind in range(len(y_binned)):
            popt_bin, _, peaks_bin = fit_odmr_multi_lorentzian(freqs / 10 ** 9, binned_counts[x_ind, y_ind], max_peaks=max_peaks)
            # Use popt as initial params for fitment for each of the
            for x_in_bin in range(binning_num):
                for y_in_bin in range(binning_num):
                    # Fit with popt, somehow have to feed this into the
                    counts = counts_2D[x_ind*binning_num + x_in_bin, y_ind*binning_num + y_in_bin]/binning_num**2
                    popt, pcov, peaks = fit_odmr_multi_lorentzian(freqs / 10 ** 9, counts, default_fit=(popt_bin, peaks_bin))
                    B_Z = params_to_B(popt)
                    B_Z_overall[x_ind * binning_num + x_in_bin, y_ind * binning_num + y_in_bin] = B_Z
                    if B_Z == 0:
                        # had problem fitting
                        problem_points.append((x_ind * binning_num + x_in_bin, y_ind * binning_num + y_in_bin))
    return B_Z_overall, problem_points

def counts_to_SNR_contrast(x_points, y_points, counts_2D, freqs,max_peaks):

    # B_Z_overall = np.zeros((len(x_points), len(y_points)), dtype=float)
    #
    # num_printouts = 10
    # printout_factor = len(x_points) * len(y_points) // num_printouts
    #
    # for x_ind in range(len(x_points)):
    #     for y_ind in range(len(y_points)):
    #         if ((x_ind*len(y_points) + y_ind) % printout_factor == 0):
    #             # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
    #             print(f"at position (x,y)=({x_ind},{y_ind}); {(x_ind*len(y_points) + y_ind)/(printout_factor*num_printouts)*100}% done")
    #         popt, pcov, counts_norm, fitted_norm, baseline = analyze_data(freqs, counts_2D[x_ind,y_ind], max_peaks)
    #         contrasts, FWHMs, dip_Freqs = get_dip_params(popt)

    all_snrs = np.zeros((x_points.shape[0], y_points.shape[0], max_peaks))
    all_contrasts = np.zeros((x_points.shape[0], y_points.shape[0], max_peaks))
    for x in range(x_points.shape[0]):
        for y in range(y_points.shape[0]):
            snrs, contrasts = ODMR_to_SNR_contr(counts_2D[x, y, :], freqs, max_peaks)
            all_snrs[x, y, :] = snrs
            all_contrasts[x, y, :] = contrasts

    return all_snrs, all_contrasts


def ODMR_to_SNR_contr(counts, freqs, max_peaks):
    popt, pcov, counts_norm, fitted_norm, baseline = analyze_data(freqs, counts, max_peaks)
    contrasts, _, _ = get_dip_params(popt)
    try:
        snrs = get_SNRs(baseline, counts, freqs / 10 ** 9, popt)
        # print_SNR(snrs, dip_Freqs)
    except Exception as e:
        print("getting SNRs returned an error", e)
        snrs = np.zeros(max_peaks)
        # print("Fitted dip frequencies at ", dip_Freqs, "GHz, with FWHMs ", FWHMs)
        # oPlot.plot_fitted_data(freqs/10**9, counts_norm, fitted_norm)

    return snrs, contrasts


def counts_to_SNR_contrast_init_params(x_points, y_points, counts_2D, freqs, max_peaks=4, binning_num=1):
    if len(x_points) % binning_num != 0 or len(y_points) % binning_num != 0:
        raise ValueError("binning_num is not an even divisior of axis sizes")

    all_snrs = np.zeros((x_points.shape[0], y_points.shape[0], max_peaks))
    all_contrasts = np.zeros((x_points.shape[0], y_points.shape[0], max_peaks))

    binned_counts, x_binned, y_binned = pci.bin_counts(counts_2D, binning_num, x_points, y_points)
    for x_ind in range(len(x_binned)):
        for y_ind in range(len(y_binned)):
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

    return all_snrs, all_contrasts
