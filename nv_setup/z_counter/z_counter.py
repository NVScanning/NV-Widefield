from qm import QuantumMachinesManager
from qm.qua import *
from config import *
from qualang_tools.plot import interrupt_on_close

import os
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(".."))
import APT.thorlabs_apt as apt

import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import connection_setup as cs

# -------------------------
# APT motor Parameters
# -------------------------

# this is the s/n on the stepper control box
# x_motor_id = 90335875
# y_motor_id = 90335876
z_motor_id = 90335877 # s/n of the z motor


# -------------------------
# Helper functions
# -------------------------


def calc_sweep_range(z_center: float, span: float, num_points: int):
    z_start = z_center - span / 2
    z_end = z_center + span / 2
    z_range = np.linspace(z_start, z_end, num_points)
    return z_start, z_end, z_range

def plot_graph(z_range: np.ndarray, kcps: np.ndarray):

    plt.figure(figsize=(8, 5))
    plt.plot(z_range, kcps, "-o", markersize=2)
    plt.xlabel("Position ())")
    plt.ylabel("kcps")
    plt.title("Counts as a fn of Z")
    plt.grid(True)
    plt.show()


STATE_PATH = os.path.join("state", "z_focus.json")

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# def connect_motor(sample_name: str, motor_id: int = 90335877):
#     print("Attempting to connect to motor, ID:", motor_id)
#     # time.sleep(2)
#     motor = apt.Motor(motor_id)
#
#     motor.move_home(True)
#     time.sleep(2)
#     print("Connected to motor, Motor ID:", motor_id)
#
#     state = load_focus_state()
#     if state and state.get("sample") == sample_name:
#         motor.move_to(state["z_mm"])
#         time.sleep(2)
#         print("Previous optimized position loaded")
#
#     return motor
#
# # STATE_PATH = os.path.join("state", "z_focus.json")
# def load_focus_state(path = STATE_PATH):
#     if not os.path.exists(path):
#         return None
#     if os.path.getsize(path) == 0:
#         return None
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


def save_focus_state(sample, z_mm, kcps, path = STATE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "sample": sample,
        "z_mm": float(z_mm),
        "kcps": float(kcps),
        "updated_at": _now_iso(),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return data


# -------------------------
# QUA program
# -------------------------

def z_counts_program(num_z_points, n_windows_per_point, readout_len_ns):
    with program() as z_counts:
        times = declare(int, size=10000)
        counts = declare(int)
        total_counts = declare(int)
        
        i = declare(int)
        k = declare(int)
        counts_st = declare_stream()

        with for_(i, 0, i < num_z_points, i + 1):
            pause()
            assign(total_counts, 0)
            with for_(k, 0, k < n_windows_per_point, k + 1):
                measure( "readout", "SPCM", time_tagging.analog(times, readout_len_ns, counts))
                assign(total_counts, total_counts + counts)

            save(total_counts, counts_st)

        with stream_processing():
            counts_st.save_all("counts")

    return z_counts

# -------------------------
# z_direction sweep
# -------------------------

def find_best_z(motor, job, z_range, dwell, point_duration_s) -> np.ndarray:
    
    counts_handle = job.result_handles.get("counts")
    seen=0
    kcps=[]
    for z in z_range:
        if (seen % 10 == 0):
            print("at z= " + str(z) + "mm")
        motor.move_to(z)
        time.sleep(dwell)
        job.resume()
        counts_handle.wait_for_values(seen+1)
        #print('count seen')
        all_counts = counts_handle.fetch_all()["value"]
        if (seen % 10 == 0):
            print("counts= " + str(all_counts[seen]))
        kcps.append(( all_counts[seen] / point_duration_s ) /1000 ) 
        seen+=1
    max_idx = np.argmax(kcps)
    optimized_z_pos = z_range[max_idx]
    return kcps, optimized_z_pos, max(kcps)
    

def main():

    # -------------------------
    # Parameters
    # -------------------------
    # photo counts parameters
    readout_len_ms = int(0.2 * u.ms)   # readout in ms
    n_windows_per_point = 1         # tbh idk what this does other than just increase the value of readout eln ^

    dwell = 0.2 # units? Used to be 2
    # position z parameters
    z_center = 6.1
    span = 0.1
    N = 51

    # sample name
    sample_name = 'NV bulk'

    # -------------------------
    # Connect to motor
    # -------------------------
    # TODO: check if motor has some connection going on, if it does, then close it
    motor = cs.connect_motor(sample_name, z_motor_id) # use the connection_setup implementation instead

    # -------------------------
    # Execute program on QM
    # -------------------------
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, log_level="INFO")
    qm = qmm.open_qm(config)
    prog = z_counts_program(N, n_windows_per_point, readout_len_ms)
    job = qm.execute(prog)

    z_start, z_end, z_range = calc_sweep_range(z_center, span, N)
    print("sweeping z range from ", z_start, " to end ", z_end)
    point_duration_s = (readout_len_ms * n_windows_per_point) / 1e9

    # -------------------------
    # Measure and get optimum z
    # -------------------------
    counts, optimized_z_pos, max_kcps = find_best_z(motor, job, z_range, dwell, point_duration_s)
    print("Highest count at: ", optimized_z_pos)
    plot_graph(z_range, counts)

    # -------------------------
    # Move and/or save position
    # -------------------------

    print("\nOptimization finished.")
    print(f"Best Z position: {optimized_z_pos:.4f} mm")
    print(f"Max count: {max_kcps:.2f} kcps")

    ans = input("Move to best position? (Y/N): ").strip().lower()

    if ans == "y":
        motor.move_to(optimized_z_pos)
        time.sleep(0.5)
        print("Moved to optimized position and state saved.")
    else:
        print("Stayed at current position.")

    save_focus_state(sample_name, optimized_z_pos, max_kcps)



if __name__ == "__main__":
    main()


















