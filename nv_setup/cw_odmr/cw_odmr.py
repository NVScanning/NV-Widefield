import qm
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

"""
This sweeps a range of RF frequencies, while kepeing 532nm light constant, and position constant

Note: 20kcounts is the limit for the SPCM, if you see this then decrease brightness or exposure time
"""

# -------------------------
# sg386 Parameters
# -------------------------

sg_resource = "TCPIP::169.254.2.7::5025::SOCKET"

# -------------------------
# Helper functions
# -------------------------
def calc_freq_range(f_center: float, span: float, n: int):
    f_start = f_center - span / 2
    f_end = f_center + span / 2
    freqs = np.linspace(f_start, f_end, n)
    return freqs, f_start, f_end

def plot_odmr(freqs: np.ndarray, kcps: np.ndarray):

    plt.figure(figsize=(8, 5))
    plt.plot(freqs / 1e9, kcps, "-o", markersize=2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("kcps")
    plt.title("ODMR")
    plt.grid(True)
    plt.show()

# -------------------------
# QUA program
# -------------------------

def odmr_qua_program(N_freq, n_windows_per_point, readout_len_ns):
    with program() as odmr_counts:
        times = declare(int, size=10000)
        counts = declare(int)
        total_counts = declare(int)
        
        i = declare(int)
        k = declare(int)
        counts_st = declare_stream()
        
        with for_(i, 0, i < N_freq, i + 1):
            pause()
            assign(total_counts, 0)
            with for_(k, 0, k < n_windows_per_point, k + 1):
                measure( "readout", "SPCM", time_tagging.analog(times, readout_len_ns, counts))
                assign(total_counts, total_counts + counts)

            save(total_counts, counts_st)

        with stream_processing():
            counts_st.save_all("counts")

    return odmr_counts

# -------------------------
# Frequency sweep
# -------------------------

def measure_odmr(sg, job, freqs, dwell, point_duration_s, n_iter: int = 1) -> np.ndarray:

    counts_handle = job.result_handles.get("counts")
    seen=0


    kcps = []
    for f in freqs:
        if (seen % 10 == 0):
            print("at freq " + str(f/10**9) + "GHz")
        sg.write(f"FREQ {float(f)}")
        time.sleep(dwell)
        job.resume()
        counts_handle.wait_for_values(seen + 1)
        # print('count seen')
        all_counts = counts_handle.fetch_all()["value"]
        kcps.append((all_counts[seen] / point_duration_s) / 1000)
        seen += 1
    return kcps

    # # TODO: use below code to make use of n_iter
    # kcps_overall = np.empty((n_iter, freqs.size))
    # for i in range(n_iter):
    #
    #     kcps=[]
    #     for f in freqs:
    #         sg.write(f"FREQ {float(f)}")
    #         time.sleep(dwell)
    #         job.resume()
    #         counts_handle.wait_for_values(seen+1)
    #         #print('count seen')
    #         all_counts = counts_handle.fetch_all()["value"]
    #         kcps.append(( all_counts[seen] / point_duration_s ) /1000 )
    #         seen+=1
    #     kcps_overall[i]=kcps
    #
    # return np.sum(kcps_overall,axis=0)

def main():

    # -------------------------
    # Parameters
    # -------------------------
    readout_len_ns = int(50 * u.us)
    n_windows_per_point = 1000 # n readouts to increase certainty without hitting the SPCM limit of ~20K points
    amp_dbm = -12 #anything bigger than -10 does nothing TODO: figure out wtf this does

    dwell =  0.01 # seconds I guess

    n_iter = 1 # stub

    # frequency parameters
    f_center = 2.88e9
    span = 0.06e9 # was 0.1e9 previously
    N = 101 # point in the frequency space to sample


    # connect to RF src
    sg = cs.connect_sg386(sg_resource)

    # -------------------------
    # Execute program on QM
    # -------------------------
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, log_level="INFO")
    qm = qmm.open_qm(config)
    prog = odmr_qua_program(N, n_windows_per_point, readout_len_ns)
    job = qm.execute(prog)

    freqs, f_start, f_end = calc_freq_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")
    point_duration_s = (readout_len_ns * n_windows_per_point) / 1e9

    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(1)
    try:
        counts = measure_odmr(sg, job, freqs, dwell, point_duration_s, n_iter)
    finally:
        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)
    plot_odmr(freqs, counts)
    now = datetime.datetime.now()

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    datestamp = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")

    # Combine script directory with your desired data path
    directory = os.path.join(project_root, "NVCFM_Data", datestamp)

    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = os.path.join(directory, f"cw_odmr_{timestamp}.npy")
    np.save(save_path, counts)

    
if __name__ == "__main__":
    main()