from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

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
    peaks, props = find_peaks(-vals,threshold=5,prominence=15, distance=5)

    if max_peaks is not None and len(peaks) > max_peaks:
        prominences = -vals[peaks]
        top_idx = np.argsort(prominences)[-max_peaks:]
        peaks = peaks[top_idx]
        peaks = peaks[np.argsort(peaks)]

    c0 = np.median(vals)   # offset
    c1 = 0.0                 # slope value 0

    init_params = []
    results_half = peak_widths(-vals, peaks, rel_height=0.5)
    widths = results_half[0]

    df = np.mean(np.diff(freqs))

    for i, p in enumerate(peaks):
        f0 = freqs[p]
        amp = vals[p] - c0
        w_idx = widths[i]
        fwhm = w_idx * df
        gamma = fwhm / 2.0 if fwhm > 0 else df*3
        init_params.extend([amp, f0, gamma])

    init_params.extend([c0, c1])
    return np.array(init_params), peaks

def fit_odmr_multi_lorentzian(freqs, R_vals, max_peaks=None):
    p0, peaks = guess_initial_params(freqs, R_vals, max_peaks=max_peaks)
    # print(f"guessed initial peaks to be at {freqs[peaks]}GHz")
    n_peaks = (len(p0) - 2) // 3

    lower = []
    upper = []
    for i in range(n_peaks):
        A0, f0, g0 = p0[3*i:3*i+3]
        lower += [A0*3, f0 - 0.02, 0]
        upper += [0,     f0 + 0.02, (freqs.max()-freqs.min())]

    # c0, c1 bounds
    c0, c1 = p0[-2], p0[-1]
    lower += [R_vals.min() - 1, -10]  
    upper += [R_vals.max() + 1,  10]

    try:
        popt, pcov = curve_fit(
            multi_lorentzian,
            freqs,
            R_vals,
            p0=p0,
            bounds=(lower, upper),
            maxfev=10000
        )
        return popt, pcov, peaks
    except:
        print("Couldn't curve_fit, returning find_peaks vals with 0 uncertainty")
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
        C_percent = C_frac * 100.0

        # FWHM (GHz → MHz later)
        FWHM = 2.0 * gamma  # same unit as freqs (GHz)

        contrasts.append(C_frac)
        FWHMs.append(FWHM)
        dip_Freqs.append(f0)
    return contrasts, FWHMs, dip_Freqs


def print_dip_params(popt):
    contrasts, FWHMs, dip_Freqs = get_dip_params(popt)

    # Print summary lines
    for (C, FWHM, freq) in zip(contrasts, FWHMs, dip_Freqs):
        # this tells us about T2 time (dephasing rate)
        print(f"At frequency {freq:.3f} GHz: FWHM = {FWHM * 1e3:.2f} MHz, Contrast = {C * 100:.3f}%")
    for i in range(len(dip_Freqs)-1):
        # this tells us abt magnetic field
        print(f"Frequency delta is {(dip_Freqs[i+1]-dip_Freqs[i])*1000}MHz")
    return contrasts, FWHMs, dip_Freqs

def print_SNR(baseline: Any, counts, freqs, popt):
    noise_signal = counts - baseline

    n_dips = (len(popt) - 2) // 3
    dip_params = popt[:3 * n_dips].reshape(n_dips, 3)
    off_mask = np.ones_like(freqs, dtype=bool)

    k_exclude = 2.0  # was previously 3.0
    for A, f0, gamma in dip_params:
        FWHM = 2.0 * gamma
        off_mask &= (np.abs(freqs - f0) > (k_exclude * FWHM))

    if np.count_nonzero(off_mask) < max(10, 0.1 * len(freqs)):
        raise ValueError("Off-resonance region too small. Decrease k_exclude or widen sweep range.")

    sigma = np.std(noise_signal[off_mask], ddof=1)
    print("sigma of background : ", sigma, " kcps")

    snrs = []
    for i, (A, f0, gamma) in enumerate(dip_params, start=1):
        signal = abs(A)

        snrs.append(signal / sigma)

    snr = np.mean(snrs)

    print("SNR : ", snr)

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
    counts_norm = counts / baseline
    fitted_norm = fitted_counts / baseline

    # print_SNR(baseline, counts, freqs, popt)
    return popt, pcov, counts_norm, fitted_norm, baseline




# ============================
# Plot graph
# ============================

def plot_fitted_data(freqs, I_norm, fit_norm):
    # Expects frequencies in GHz
    fig = plt.figure(figsize=(8,5))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')

    ax.plot(freqs, I_norm, '-o', ms=2, color='k')
    ax.plot(freqs, fit_norm, '-', lw=2, color='C0', alpha = 0.6, label='Lorentzian fit')

    ax.axvline(
        x=2.87,
        color='red',
        linestyle='--',
        linewidth=1.2,
        alpha=0.7,
        label='2.87 GHz'
    )

    ax.set_title("ODMR")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Normalized Intensity")
    ax.legend(loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    plt.show()