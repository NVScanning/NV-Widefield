
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

from qm import QuantumMachinesManager
from config import *
import matplotlib.pyplot as plt

import connection_setup as cs
import QUA_interface as QUAi


# -------------------------
# Constants
# -------------------------

# this is the s/n on the stepper control box
x_motor_id = 90335875
y_motor_id = 90335876
z_motor_id = 90335877 # s/n of the z motor

sg_resource = "TCPIP::169.254.2.7::5025::SOCKET"

# -------------------------
# X&Y sweep
# -------------------------
def measure_all_points(sg, job, x_motor, y_motor, x_points, y_points, freqs, dwell, point_duration_s, n_iter=1):
    # TODO: make the below code (stolen from z_counter and cw_odmr) work in 2D

    # have to sweep through freqs at each (x,y) point


    # counts_handle = job.result_handles.get("counts")
    # seen=0
    # kcps=[]
    # for z in z_range:
    #     if (seen % 10 == 0):
    #         print("at z= " + str(z) + "mm")
    #     motor.move_to(z)
    #     time.sleep(dwell)
    #     job.resume()
    #     counts_handle.wait_for_values(seen+1)
    #     #print('count seen')
    #     all_counts = counts_handle.fetch_all()["value"]
    #     if (seen % 10 == 0):
    #         print("counts= " + str(all_counts[seen]))
    #     kcps.append(( all_counts[seen] / point_duration_s ) /1000 )
    #     seen+=1
    #     return kcps

    # counts_handle = job.result_handles.get("counts")
    # seen=0
    #
    # num_printouts = 5
    # printout_factor = len(freqs) // num_printouts
    #
    # kcps_overall = np.empty((n_iter, freqs.size))
    # for i in range(n_iter):
    #     print("Iteration " + str(i))
    #
    #     kcps=[]
    #     for f in freqs:
    #         if (seen % printout_factor == 0):
    #             # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
    #             print("at freq " + str(f/10**9) + "GHz; "+ str(seen/(printout_factor*num_printouts*n_iter)*100) + "% done")
    #         sg.write(f"FREQ {float(f)}")
    #         time.sleep(dwell)
    #         job.resume()
    #         counts_handle.wait_for_values(seen+1)
    #         all_counts = counts_handle.fetch_all()["value"]
    #         kcps.append(( all_counts[seen] / point_duration_s ) /1000 )
    #         seen+=1
    #     kcps_overall[i]=kcps
    #
    # return np.sum(kcps_overall,axis=0)/n_iter

    return np.zeros((len(x_points), len(y_points), len(freqs)), dtype=float) # stub

# -------------------------
# ODMR analysis
# -------------------------
def counts_to_delta_freq(counts_2D):
    # TODO: use methods from Lorentzian_fit.py to implement this

    # something similar to:

    # max_peaks = 2
    # popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
    # Lfit.print_dip_params(popt)
    # Lfit.print_SNR(baseline, counts, freqs / 10 ** 9, popt)
    # Lfit.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)

    return np.zeros_like(counts_2D)

# -------------------------
# Image plotting
# -------------------------
def plot_image(x_points, y_points, freq_deltas_2D):
    # below stub written by Gemini
    plt.figure(figsize=(10, 6))
    # shading='auto' handles the coordinate mapping automatically
    mesh = plt.pcolormesh(x_points, y_points, freq_deltas_2D, shading='auto', cmap='viridis')

    plt.colorbar(mesh, label='frequency delta (GHz)')
    plt.xlabel('space (mm)')
    plt.ylabel('space (mm)')
    plt.title('ODMR Heatmap')
    plt.show()

def main():
    # -------------------------
    # Parameters
    # -------------------------
    # photo counts parameters
    readout_len_ns = int(0.2 * u.ms)  # readout in ns
    n_windows_per_point = 1  # tbh idk what this does other than just increase the value of readout eln ^
    amp_dbm = -20

    dwell = 0.001  # s

    f_center = 2.88e9 # Hz, generally near 2.87GHz
    f_span = 0.6e9 # Hz, range of frequencies to sample
    f_N = 201 # num points in the frequency space to sample

    # position parameters
    x_center,y_center = 0,0 # center of measurement
    x_span, y_span = 0.2,0.2 # range in each axis to sample
    x_N, y_N = 51,51 # num points in each axis to sample

    # sample name
    sample_name = 'scanned NV bulk'

    # connect to RF src
    sg = cs.connect_sg386(sg_resource)
    # connect to motors
    x_motor = cs.connect_motor(sample_name, x_motor_id)
    y_motor = cs.connect_motor(sample_name, y_motor_id)

    # Create + Execute program on QM
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, log_level="INFO")
    qm = qmm.open_qm(config)
    prog = QUAi.odmr_qua_program(y_N*x_N, n_windows_per_point, readout_len_ns)
    job = qm.execute(prog)

    x_start, x_end, x_points = QUAi.calc_sweep_range(x_center, x_span, x_N)
    y_start, y_end, y_points = QUAi.calc_sweep_range(y_center, y_span, y_N)
    print(f"sweeping x from {x_start} to {x_end} and y from {y_start} to {y_end}")
    f_start, f_end, freqs = QUAi.calc_sweep_range(f_center, f_span, f_N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")

    point_duration_s = (readout_len_ns * n_windows_per_point) / 1e9

    counts_2D = measure_all_points(sg, job, x_motor, y_motor, x_points, y_points, freqs, dwell, point_duration_s, n_iter=1)
    freq_deltas = counts_to_delta_freq(counts_2D)
    plot_image(x_points, y_points, freq_deltas)


if __name__ == "__main__":
    main()