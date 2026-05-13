

import pyvisa
import os
import time
import sys
sys.path.append(os.path.abspath("."))
import APT.thorlabs_apt as apt
import json

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

STATE_PATH = os.path.join("state", "z_focus.json")
def load_focus_state(path = STATE_PATH):
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) == 0:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def connect_motor(sample_name: str, motor_id: int):
    available_devices = apt.list_available_devices()

    if not available_devices:
        raise Exception("No Thorlabs devices detected. Check USB connection/Power/Opened programs.")

    sns = [device[1] for device in available_devices]

    if motor_id not in sns:
        raise Exception(f"Motor {motor_id} not found. Currently visible: {sns}. ")


    motor = apt.Motor(motor_id)

    motor.move_home(True)
    time.sleep(2)
    print("Connected to motor, Motor ID:", motor_id)

    state = load_focus_state()
    if state and state.get("sample") == sample_name:
        motor.move_to(state["z_mm"])
        time.sleep(2)
        print("Previous optimized position loaded")

    return motor