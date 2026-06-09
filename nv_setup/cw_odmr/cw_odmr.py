from typing import Any

import qm
from numpy import dtype, float64, ndarray
from qm import QuantumMachinesManager
from qm.qua import *
from config import *
from qualang_tools.plot import interrupt_on_close

import time
from tqdm import tqdm
import os
import numpy as np
import pyvisa
import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(".."))
import connection_setup as cs
sys.path.append(os.path.abspath("..."))
import nv_widefield.odmr_plotting as oPlot

import Lorentzian_fit as Lfit
import QUA_interface as QUAi

"""
This sweeps a range of RF frequencies, while kepeing 532nm light constant, and position constant

This is made to work with the SPCM (single photon counting module) which is read by the QM (quantum machine)

Note: 20kcounts is the limit for the SPCM, if you see this then decrease brightness or exposure time
"""

# -------------------------
# Frequency sweep
# -------------------------

def measure_odmr(sg, job, freqs, dwell, point_duration_s, n_iter: int = 1) -> np.ndarray:

    counts_handle = job.result_handles.get("counts")
    seen=0

    num_printouts = 10
    printout_factor = len(freqs)*n_iter*2 // num_printouts

    kcps_overall = np.zeros((n_iter*2, freqs.size))
    for i in range(n_iter):
        print("Iteration " + str(i))

        kcps=[]
        for j,f in enumerate(freqs):
            if (seen % printout_factor == 0):
                # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
                print(f"at freq {str(f / 10 ** 9)}GHz; {seen/(printout_factor*num_printouts)*100:.0f}% done")
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            job.resume()
            counts_handle.wait_for_values(seen+1)
            all_counts = counts_handle.fetch_all()["value"]
            kcps.append(( all_counts[seen] / point_duration_s ) /1000 ) # maybe need to index at seen+1?
            seen+=1
        kcps_overall[i]=kcps
        kcps=[]
        for j,f in enumerate(freqs[::-1]):
            if (seen % printout_factor == 0):
                # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
                print(f"at freq {str(f / 10 ** 9)}GHz; {seen/(printout_factor*num_printouts)*100:.0f}% done")
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            job.resume()
            counts_handle.wait_for_values(seen+1)
            all_counts = counts_handle.fetch_all()["value"]
            kcps.append(( all_counts[seen] / point_duration_s ) /1000 ) # maybe need to index at seen+1?
            seen+=1
        kcps_overall[n_iter+i]=kcps[::-1]

    return np.sum(kcps_overall,axis=0)/(n_iter*2)

def main():

    # -------------------------
    # Parameters
    # -------------------------
    readout_len_ns = int(50 * u.us) # 50 us is near the max with a 0ND filtering on the 5mW laser (I think)
    n_windows_per_point = 50 # n readouts to increase certainty without hitting the SPCM limit of ~20K (is M?) points
    amp_dbm = -5 #anything bigger than -10 does nothing (Hayden)
    # Always use with 28V on the amplifier, amp_dbm ~30 is the lowest you can set while still seeing the zero-field dips
    # Larger amp means dips are more visible, but also get wider so you lose frequency resolution

    dwell =  0.01 # seconds - time between setting a frequency on fn generator and reading value
    n_iter = 1
    # frequency parameters
    f_center = 2.87e9 # Hz, generally near 2.87GHz
    span = 0.15e9 # Hz, range of frequencies to sample
    N = 51 # num points in the frequency space to sample

    # connect to RF src
    sg = cs.connect_sg386(cs.sg_resource)

    # -------------------------
    # Execute program on QM
    # -------------------------
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, log_level="INFO")
    qm = qmm.open_qm(config)
    prog = QUAi.odmr_qua_program(N * n_iter*2, n_windows_per_point, readout_len_ns)
    job = qm.execute(prog)

    f_start, f_end, freqs = QUAi.calc_sweep_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")
    point_duration_s = (readout_len_ns * n_windows_per_point) / 1e9

    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(0.1) # why sleep for a whole second? (previous was 1)
    try:
        counts = measure_odmr(sg, job, freqs, dwell, point_duration_s, n_iter)
    finally:
        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)


    oPlot.plot_odmr(freqs, counts)

    # Save data in folder with its date
    oPlot.save_point_odmr_measurement(counts, freqs)


    # Calculating space between dips
    max_peaks = 2
    popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
    Lfit.print_dip_params(popt)
    # contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
    # for (C, FWHM, freq) in zip(contrasts, FWHMs, dip_Freqs):
    #     print(f"At frequency {freq:.3f} GHz: FWHM = {FWHM * 1e3:.2f} MHz, Contrast = {C * 100:.3f}%")
    # for i in range(len(dip_Freqs)-1):
    #     print(f"Frequency delta is {(dip_Freqs[i+1]-dip_Freqs[i])*1000}MHz")

    # Since above fn is working, I can change the relevant passage to determine space between dips

    try:
        snrs = Lfit.get_SNRs(baseline, counts, freqs/10**9, popt)
        Lfit.print_SNR(snrs, freqs/10**9)
    except ValueError as e:
        # do nothing cuz printing snr didnt work
        print("getting SNR failed" + str(e))
    oPlot.plot_fitted_data(freqs/10**9, counts_norm, fitted_norm)


if __name__ == "__main__":
    main()