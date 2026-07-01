
# For each (x,y) point in 2D space, I do an RF scan
# From the RF scan, I can take either:
#   1. the space betweeen dips in Hz (magnetic field)
#   2. the width of dips (FWHM) in Hz (Rabi smth)
#   3. the background counts in cps (idk what this tells us)


# For this to work, I need to first get cw_odmr working at a fixed point, so focus on this

# Instructions:
# create QUA program for reading an ODMR at each x,y point
# create empty 2D array for dip frequency delta
#   (just initial thing to get working, it'll be easy to convert to saving other params from the lorentzian fit)

# iterate through 2D space in x&y with steppers,
#   for each position, take a cw_ODMR measurement and determine dip frequency delta and save it in the above empty array

# plot the (now-filled) 2D array of values as an image
# when imaging bulk diamond this should be pretty much constant

# problem of converting odmr to B field:
# when I first measure, I take each individual ODMR and get the B measured, and average over the n_iter
# when I read a scan, all the odmrs have been averaged, so when I get the B measured it's only from this one
# problem is: idk if there's a difference/which one is better if there is
# Also, this number, 28.02 T/GHz might be for a singular perfectly aligned NV, idk how
# it would change for misaligned and randomized centers

from qm import QuantumMachinesManager
from config import *
import matplotlib.pyplot as plt
import time
import datetime
from pathlib import Path

import connection_setup as cs
import Lorentzian_fit as Lfit
import QUA_interface as QUAi
sys.path.append(os.path.abspath("..."))
import nv_widefield.odmr_plotting as oPlot

"""
This uses a single point sensor (SPCM), and scans over x-y, measuring an ODMR at each point
Each point's ODMR is converted to magnetic field by fitting lorentzians, which is then
displayed as an image
the x-y and frequency arrays, as well as every odmr scan and the final magnetic field is saved

This was used to help develop the code that analyzes an image together as a stepping stone
"""

#TODO: modify code to be able to sweep z as well, and I simply choose the axes I want to sweep through


# -------------------------
# X&Y sweep
# -------------------------
def measure_all_points(sg, job, x_motor, y_motor, x_points, y_points, freqs, dwell, point_duration_s, n_iter=1):

    counts_handle = job.result_handles.get("counts")
    seen=0
    kcps_overall = np.zeros((n_iter, len(x_points), len(y_points), len(freqs)), dtype=float) # create empty array
    B_Z_overall = np.zeros((n_iter, len(x_points), len(y_points)), dtype=float)  # create empty array
    problem_points = []


    num_printouts = 10
    printout_factor = len(freqs) * len(x_points) * len(y_points) // num_printouts


    for i in range(n_iter):
        x_motor.move_to(x_points[0])
        y_motor.move_to(y_points[0])
        time.sleep(1) # give 1s to move to starting position
        for (x_ind,x) in enumerate(x_points):
            for (y_ind,y) in enumerate(y_points):
                # move to (x,y) position and measure an ODMR
                # print(f"trying to move to (x,y)=({x},{y}) using indices:{x_ind},{y_ind}")
                x_motor.move_to(x)
                y_motor.move_to(y)
                time.sleep(dwell)
                kcps=[]
                for f in freqs:
                    if (seen % printout_factor == 0):
                        # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
                        print(f"at position (x,y)=({x},{y}); {seen/(printout_factor*num_printouts*n_iter)*100}% done")
                    sg.write(f"FREQ {float(f)}")
                    time.sleep(0.01)
                    job.resume()
                    counts_handle.wait_for_values(seen+1)
                    all_counts = counts_handle.fetch_all()["value"]
                    kcps.append(( all_counts[seen] / point_duration_s ) /1000 )
                    seen+=1
                kcps_overall[i,x_ind,y_ind,:]=np.array(kcps)
                delta_freq = Lfit.odmr_to_delta_freq(np.array(kcps)  , freqs) # in GHz
                B_Z = delta_freq / (2*cs.gamma_e) # in T
                B_Z_overall[i,x_ind,y_ind]=B_Z
                if delta_freq == 0:
                    # had problem fitting
                    problem_points.append((x_ind, y_ind))
                cs.plot_dFreq_image(x_points, y_points, np.sum(B_Z_overall, axis=0) / i)

                # freq_deltas_temp, problem_points_temp = cs.counts_to_delta_freq(x_points, y_points, counts_2D, freqs)
                # plot_image(x_points, y_points, freq_deltas_temp)

    return np.sum(kcps_overall,axis=0)/n_iter, np.sum(B_Z_overall, axis=0)/n_iter, problem_points
    return kcps_overall, B_Z_overall, problem_points

def main():
    # -------------------------
    # Parameters
    # -------------------------
    # photo counts parameters
    readout_len_ns = int(50 * u.us)  # readout in ns
    n_windows_per_point = 50 # n readouts to increase certainty without hitting the SPCM limit of ~20K (is M?) points
    amp_dbm = -5

    dwell = 0.2  # s/mm
    n_iter = 2 # n_iter not implemented yet, default value is 1

    f_center = 2.87e9 # Hz, generally near 2.87GHz
    f_span = 0.15e9 # Hz, range of frequencies to sample
    f_N = 51 # num points in the frequency space to sample

    # position parameters
    x_center,y_center = 4.8,3.8 #mm, center of measurement
    # centering on an edge, ~x,y=3.8,2.8 is a corner, more positive x and y are the top surface
    # the motor positions are in absolute value set by the homing sequence. Total travel range is 8mm
    # so the x,y points must all be between 0,8
    x_span, y_span = 2,2 #mm,  range in each axis to sample
    x_N, y_N = 20,20 # num points in each axis to sample


    sg = cs.connect_sg386(cs.sg_resource) # connect to RF src
    # connect to motors
    x_motor, x_prev_pos = cs.connect_motor(cs.x_mID)
    y_motor, y_prev_pos = cs.connect_motor(cs.y_mID)

    # Create + Execute program on QM
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, log_level="INFO")
    qm = qmm.open_qm(config)
    prog = QUAi.odmr_qua_program(y_N*x_N*f_N*n_iter, n_windows_per_point, readout_len_ns)
    job = qm.execute(prog)

    x_start, x_end, x_points = QUAi.calc_sweep_range(x_center, x_span, x_N)
    y_start, y_end, y_points = QUAi.calc_sweep_range(y_center, y_span, y_N)
    print(f"sweeping x from {x_start} to {x_end} and y from {y_start} to {y_end}")
    f_start, f_end, freqs = QUAi.calc_sweep_range(f_center, f_span, f_N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")

    point_duration_s = (readout_len_ns * n_windows_per_point) / 1e9

    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(0.1) # why sleep for a whole second? (previous was 1)
    try:
        # Note: it is currently set up to measure all the cw_ODMRs, then convert each of them after the fact
        counts_2D, B_Z_overall, problem_points = measure_all_points(sg, job, x_motor, y_motor, x_points, y_points, freqs, dwell*y_span, point_duration_s, n_iter=n_iter)
    finally:
        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)

    # freq_deltas, problem_points = cs.counts_to_delta_freq(x_points, y_points, counts_2D, freqs)
    print("following indices couldn't fit properly:")
    print(problem_points)
    oPlot.plot_dFreq_image(x_points, y_points, B_Z_overall)
    oPlot.save_2D_odmr_measurement(x_points, y_points, freqs, B_Z_overall, counts_2D)

if __name__ == "__main__":
    main()