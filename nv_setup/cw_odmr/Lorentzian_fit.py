import numpy as np
from scipy.optimize import curve_fit
import connection_setup as cs
from scipy.signal import find_peaks, peak_widths

import helper_classes.odmr_plotting as oPlot
from nv_widefield.helper_classes.odmr_plotting import plot_fitted_data


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

def guess_initial_params(freqs, vals, max_peaks=None):
    # freqs in GHz
    max_val = max(vals)
    peaks, props = find_peaks(-vals,prominence=0.002*max_val, distance=max(1,0.005//(freqs[1]-freqs[0])))

    if max_peaks is not None and len(peaks) > max_peaks:
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
        print("Manually split 1 peak into", max_peaks, "at freqs", freqs[peaks], "GHz")

    c0 = vals[0]                                                # offset - used to be np.mean
    c1 = (vals[-1]-vals[0])/(freqs[-1]-freqs[0])                # slope value

    init_params = []
    # results_half = peak_widths(-vals, peaks, rel_height=0.5)
    # widths = results_half[0]

    df = abs(freqs[1]-freqs[0])

    for i, p in enumerate(peaks):
        f0 = freqs[p]
        amp = -abs(vals[p] - c0)
        # w_idx = widths[i]
        # fwhm = w_idx * df
        gamma = df*3
        init_params.extend([amp, f0, gamma])

    init_params.extend([c0, c1])
    return np.array(init_params), peaks

def fit_odmr_multi_lorentzian(freqs, R_vals, max_peaks=None):
    # freqs in GHz
    p0, peaks = guess_initial_params(freqs, R_vals, max_peaks=max_peaks)
    n_peaks = (len(p0) - 2) // 3
    if max_peaks is not None and (n_peaks < max_peaks):
        print(f"Guessed only {n_peaks} peaks out of {max_peaks} peaks at frequencies ", freqs[peaks])
    lower = []
    upper = []
    for i in range(n_peaks):
        A0, f0, g0 = p0[3*i:3*i+3]
        lower += [-abs(A0*1.5), f0 - 0.005, 0]
        upper += [-abs(A0*0.5) , f0 + 0.005, g0*2] # force HWHM to be at most 5 MHz

    lower += [R_vals.min() - 1, -0.05*max(R_vals)/(freqs[-1]-freqs[0])]
    upper += [R_vals.max()*1.1,  0.05*max(R_vals)/(freqs[-1]-freqs[0])]

    try:
        popt, pcov = curve_fit(
            multi_lorentzian, freqs, R_vals,
            p0=p0, bounds=(lower, upper), maxfev=1000
            # ,ftol=0.001, xtol=0.001) # Note: these make convergence quicker, but lose accuracy
        )
        return popt, pcov, peaks
    except Exception as e:
        print("Couldn't curve_fit, threw:", e, "Plotting ODMR")
        # print("Couldn't curve_fit, threw:", e, ". trying to fit peaks at frequencies", freqs[peaks],
        #       ". p0=",p0,", lower bounds:", lower,", upper bounds:", upper)
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
    return contrasts, FWHMs, dip_Freqs

def print_contrast_snr(contrasts, snrs, dip_Freqs):
    for (freq, snr_val, contr_val) in zip(dip_Freqs, snrs, contrasts):
        print(f"At frequency {freq:.3f} GHz: Contrast = {contr_val * 100:.3f}%, snr = {snr_val:.3}")
    print(f"SNR avg: {np.mean(snrs):.3}, contrast avg: {np.mean(snrs)*100:.3}%")

def print_dip_params(popt):
    contrasts, FWHMs, dip_Freqs = get_dip_params(popt)

    # Print summary lines
    for (C, FWHM, freq) in zip(contrasts, FWHMs, dip_Freqs):
        # this tells us about T2 time (dephasing rate)
        print(f"At frequency {freq:.3f} GHz: FWHM = {FWHM * 1e3:.3} MHz, Contrast = {C * 100:.3f}%")
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

    k_exclude = 1.5  # was previously 3.0
    for A, f0, gamma in dip_params:
        FWHM = 2.0 * gamma
        off_mask &= (np.abs(freqs - f0) > (k_exclude * FWHM))

    if np.count_nonzero(off_mask) < max(10, 0.1 * len(freqs)):

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
        print(f"At frequency {freq:.3f} GHz: snr = {snr_val:.3}")

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




def odmr_to_delta_freq(counts, freqs):
    delta_freq = 0
    max_peaks = 4
    popt, pcov, counts_norm, fitted_norm, baseline = analyze_data(freqs, counts, max_peaks)
    contrasts, FWHMs, dip_Freqs = get_dip_params(popt)
    # print_SNR(baseline, counts, freqs / 10 ** 9, popt)
    # oPlot.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)
    if (len(dip_Freqs) >= 2):
        # need exactly at least 2 dips to get the difference between the two
        # if >2 dips, assume the additional ones are the middle dips (irrelevant i think)
        delta_freq = dip_Freqs[-1] - dip_Freqs[0]
    # if you didn't get >=2 dips there's no delta, so return 0
    return delta_freq

def counts_to_B_Z(x_points, y_points, counts_2D, freqs):

    # TODO: conmvert this fitting to be on the GPU with JAXFit
    # TODO: find a way to have a background term that's shared between nearby pixels
    B_Z_overall = np.zeros((len(x_points), len(y_points)), dtype=float)
    problem_points = []


    num_printouts = 10
    printout_factor = len(x_points) * len(y_points) // num_printouts

    for x_ind in range(len(x_points)):
        for y_ind in range(len(y_points)):
            if ((x_ind*len(y_points) + y_ind) % printout_factor == 0):
                # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
                print(f"at position (x,y)=({x_ind},{y_ind}); {(x_ind*len(y_points) + y_ind)/(printout_factor*num_printouts)*100}% done")
            delta_freq = odmr_to_delta_freq(counts_2D[x_ind,y_ind], freqs)
            B_Z = delta_freq / (2*cs.gamma_e) # in T
            B_Z_overall[x_ind,y_ind]=B_Z
            if delta_freq == 0:
                # had problem fitting
                problem_points.append((x_ind, y_ind))
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
            popt, pcov, counts_norm, fitted_norm, baseline = analyze_data(freqs, counts_2D[x, y, :], max_peaks)
            contrasts, FWHMs, dip_Freqs = get_dip_params(popt)
            try:
                snrs = get_SNRs(baseline, counts_2D[x, y, :], freqs / 10 ** 9, popt)
                # print_SNR(snrs, dip_Freqs)
            except Exception as e:
                print("getting SNRs returned an error", e)
                print("Fitted dip frequencies at ", dip_Freqs, "GHz, with FWHMs ", FWHMs)
                plot_fitted_data(freqs/10**9, counts_norm, fitted_norm)

            all_snrs[x, y, :] = snrs
            all_contrasts[x, y, :] = contrasts

    return all_snrs, all_contrasts