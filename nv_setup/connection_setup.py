import pyvisa
import os
import sys
sys.path.append(os.path.abspath("."))
import APT.thorlabs_apt as apt
import numpy as np
import time

# Constants
gamma_e = 28.02 #GHz/T linear term in zeeman splitting for NV centres
sg_resource = "TCPIP::169.254.2.7::5025::SOCKET"
# below are the s/n on the stepper control box
x_mID = 90335875
y_mID = 90335876
z_mID = 90335877 # s/n of the z motor



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

def enable_sg386(sg, amp_dbm: float = -10.0, enable: bool = True):
    # below numbers are for the 2.5-3GHz range we use
    # ZHL 2W amp has nominally between 42.5 and 45dB of gain, max power output of ~30dB
    # SAM100 amp has nominally 47dB of gain, max power output of ~42.3dB

    # Below -30dB I see no result, above -10dB the ZHL amp can't output more power
    # Above -2,-3 dB, the SAM100 definitely burns the silver paint, above -5 it can, depending on tension put on the connector

    if amp_dbm > -8: # -8 is a few db below SAM100 problem, and ZHL can't output enough power
        raise Exception("RF amplitude too high, risk of burning solder, must be < -5dB")

    sg.write(f"AMPR {amp_dbm}")
    sg.write(f"ENBR {1 if enable else 0}")
    if enable:
        print("sg386 ON")
    else:
        print("sg386 OFF")

def connect_motor(motor_id: int):
    motor = find_motor(motor_id)
    prev_position = motor.position

    motor.move_home(True)
    time.sleep(1) # Takes some time for the motor to perform the homing
    print(f"Currently using backlash distance: {motor.backlash_distance:.3}[mm]")
    print("Connected to motor, Motor ID:", motor_id)

    return motor, prev_position

def find_motor(motor_id: int):
    available_devices = apt.list_available_devices()

    if not available_devices:
        raise Exception("No Thorlabs devices detected. Check USB connection/Power/Opened programs.")

    sns = [device[1] for device in available_devices]

    if motor_id not in sns:
        raise Exception(f"Motor {motor_id} not found. Currently visible: {sns}. ")


    motor = apt.Motor(motor_id)

    return motor


# UTIL STUFS
def calc_sweep_range(center: float, span: float, num_points: int):
    start = center - span / 2
    end = center + span / 2
    point_array = np.linspace(start, end, num_points)
    return start, end, point_array

