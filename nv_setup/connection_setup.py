
from typing import Any
from pathlib import Path
from numpy import dtype, float64, ndarray
import pyvisa
import os
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("."))
import APT.thorlabs_apt as apt
import numpy as np
# import json
# import cw_odmr.Lorentzian_fit as Lfit

# Constants
gamma_e = 28.02


def connect_sg386(resource: str, timeout_ms: int = 5000):
    # sg386 is the RF generator
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

# STATE_PATH = os.path.join("state", "z_focus.json")
# def load_focus_state(path = STATE_PATH):
#     if not os.path.exists(path):
#         return None
#     if os.path.getsize(path) == 0:
#         return None
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

def connect_motor(sample_name: str, motor_id: int):
    available_devices = apt.list_available_devices()

    if not available_devices:
        raise Exception("No Thorlabs devices detected. Check USB connection/Power/Opened programs.")

    sns = [device[1] for device in available_devices]

    if motor_id not in sns:
        raise Exception(f"Motor {motor_id} not found. Currently visible: {sns}. ")


    motor = apt.Motor(motor_id)

    motor.move_home(True)
    # time.sleep(2)
    print("Connected to motor, Motor ID:", motor_id)

    # state = load_focus_state()
    # if state and state.get("sample") == sample_name:
    #     motor.move_to(state["z_mm"])
    #     time.sleep(2) # takes some time for motor to move to correct position
    #     print("Previous optimized position loaded")

    return motor



# UTIL STUFSS
def calc_sweep_range(center: float, span: float, num_points: int):
    start = center - span / 2
    end = center + span / 2
    point_array = np.linspace(start, end, num_points)
    return start, end, point_array


def plot_odmr(freqs: np.ndarray, kcps: np.ndarray):

    plt.figure(figsize=(8, 5))
    plt.plot(freqs / 1e9, kcps, "-o", markersize=2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("kcps")
    # plt.ylim(7.4e11,7.45e11)
    plt.title("ODMR")
    plt.grid(True)
    plt.show()

def plot_image(x_points, y_points, B_Z_overall):
    plt.figure(figsize=(10, 6))
    # shading='auto' handles the coordinate mapping automatically
    mesh = plt.pcolormesh(x_points, y_points, B_Z_overall, shading='nearest', cmap='viridis')

    plt.colorbar(mesh, label='B_Z (T)')
    plt.xlabel('x space (mm)')
    plt.ylabel('y space (mm)')
    plt.title('Magnetic Field Heatmap')
    plt.show()


def save_point_odmr_measurement(counts: ndarray[tuple[Any, ...], dtype[Any]],
                                freqs: ndarray[tuple[Any, ...], dtype[float64]]):
    now = datetime.datetime.now()
    datestamp = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    directory = os.path.join(project_root, "NVCFM_Data", datestamp)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, f"cw_odmr_{timestamp}.npz")
    print(f"Saved as: cw_odmr_{timestamp}.npz in directory: {directory}")
    np.savez(save_path, x=freqs, y=counts)


def save_2D_odmr_measurement(x_points, y_points, freqs, B_Z_overall, counts_2D):
    now = datetime.datetime.now()
    datestamp = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")
    script_path = Path(__file__).resolve()
    # project_root = script_path.parent.parent.parent
    project_root = "C:\\Users\\NVCFM\\Desktop"
    directory = os.path.join(project_root, "NVCFM_Data", datestamp)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, f"scanned_cw_odmr_{timestamp}.npz")
    print(f"Saved as: scanned_cw_odmr_{timestamp}.npz in directory: {directory}")
    np.savez(save_path, x=x_points, y=y_points, f=freqs, magnet=B_Z_overall, odmrs=counts_2D)

