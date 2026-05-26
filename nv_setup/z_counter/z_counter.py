from qm import QuantumMachinesManager
from config import *
from qualang_tools.plot import interrupt_on_close

import os
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(".."))

import json
from datetime import datetime, timezone

import connection_setup as cs
import QUA_interface as QUAi

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

#
# def calc_sweep_range(z_center: float, span: float, num_points: int):
#     z_start = z_center - span / 2
#     z_end = z_center + span / 2
#     z_range = np.linspace(z_start, z_end, num_points)
#     return z_start, z_end, z_range

def plot_graph(z_range: np.ndarray, kcps: np.ndarray):

    plt.figure(figsize=(8, 5))
    plt.plot(z_range, kcps, "-o", markersize=2)
    plt.xlabel("Z Position (mm)")
    plt.ylabel("kcps")
    plt.title("Counts as a fn of Z")
    plt.grid(True)
    plt.show()


STATE_PATH = os.path.join("state", "z_focus.json")

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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
    readout_len_ns = int(0.2 * u.ms)   # readout in ns
    n_windows_per_point = 1

    dwell = 0.2 # s Used to be 2
    # position z parameters
    z_center = 6.3
    span = 0.2
    N = 51

    # sample name
    sample_name = 'NV bulk'

    # -------------------------
    # Connect to motor
    # -------------------------
    # TODO: check if motor has some connection going on, if it does, then close it
    motor = cs.connect_motor(z_motor_id) # use the connection_setup implementation instead

    # -------------------------
    # Execute program on QM
    # -------------------------
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, log_level="INFO")
    qm = qmm.open_qm(config)
    # prog = z_counts_program(N, n_windows_per_point, readout_len_ns)
    prog = QUAi.odmr_qua_program(N, n_windows_per_point, readout_len_ns)
    job = qm.execute(prog)

    z_start, z_end, z_range = QUAi.calc_sweep_range(z_center, span, N)
    print("sweeping z range from ", z_start, " to end ", z_end)
    point_duration_s = (readout_len_ns * n_windows_per_point) / 1e9

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


















