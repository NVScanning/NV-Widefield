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

import matplotlib.pyplot as plt

# -------------------------
# sg386 Parameters
# -------------------------

sg_resource = "TCPIP::169.254.2.7::5025::SOCKET"

# -------------------------
# Helper functions
# -------------------------

def connect_sg386(resource: str, timeout_ms: int = 5000):
    rm = pyvisa.ResourceManager()
    sg = rm.open_resource(
        resource,
        write_termination="\n",
        read_termination="\n",
        timeout=timeout_ms,
    )
    print("Connected to sg386")
    return sg

def enable_sg386(sg, amp_dbm: float = -12.0, enable: bool = True):
    sg.write(f"AMPR {amp_dbm}")
    sg.write(f"ENBR {1 if enable else 0}")
    if enable:
        print("sg386 ON")
    else:
        print("sg386 OFF")

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

def measure_odmr(sg, job, freqs, dwell, point_duration_s, n_iter: int = 6) -> np.ndarray:
    
    counts_handle = job.result_handles.get("counts")
    seen=0
    kcps=[]
    for f in freqs:
        sg.write(f"FREQ {float(f)}")
        time.sleep(dwell)
        job.resume()
        counts_handle.wait_for_values(seen+1)
        #print('count seen')
        all_counts = counts_handle.fetch_all()["value"]
        kcps.append(( all_counts[seen] / point_duration_s ) /1000 ) 
        seen+=1

    return kcps

def main():

    # -------------------------
    # Parameters
    # -------------------------
    readout_len_ns = int(5 * u.ms)   # 5 ms readout
    n_windows_per_point = 1000 # 100ms per point  
    amp_dbm = -12 #anything bigger than -10 does nothing 

    dwell =  0.01

    n_iter = 10

    # frequency parameters
    f_center = 2.88e9
    span = 0.1e9
    N = 101


    # connect to RF src
    sg = connect_sg386(sg_resource)

    # -------------------------
    # Execute program on QM
    # -------------------------
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, log_level="INFO")
    qm = qmm.open_qm(config)
    prog = odmr_qua_program(N, n_windows_per_point, readout_len_ns)
    job = qm.execute(prog)

    freqs, f_start, f_end = calc_freq_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to end ", f_end/1e9)
    point_duration_s = (readout_len_ns * n_windows_per_point) / 1e9

    enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(1)
    try:
        counts = measure_odmr(sg, job, freqs, dwell, point_duration_s, n_iter)
    finally:
        enable_sg386(sg, amp_dbm=amp_dbm, enable=False)
    plot_odmr(freqs, counts)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    np.save(f"C:\\Users\\NVCFM\\Desktop\\NVCFM_Data\\cw_odmr_{timestamp}.npy",counts)
    
if __name__ == "__main__":
    main()