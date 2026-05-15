
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
import time
import datetime
from pathlib import Path

import connection_setup as cs
import Lorentzian_fit as Lfit
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
    # TODO: implement n_iter expansion

    counts_handle = job.result_handles.get("counts")
    seen=0
    kcps_overall = np.zeros((len(x_points), len(y_points), len(freqs)), dtype=float) # create empty array

    num_printouts = 10
    printout_factor = len(freqs) * len(x_points) * len(y_points) // num_printouts
    for (x_ind,x) in enumerate(x_points):
        for (y_ind,y) in enumerate(y_points):
            # move to (x,y) position and measure an ODMR
            # print(f"trying to move to (x,y)=({x},{y}) using indices:{x_ind},{y_ind}")
            x_motor.move_to(x)
            y_motor.move_to(y)
            kcps=[]
            for f in freqs:
                if (seen % printout_factor == 0):
                    # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
                    print(f"at position (x,y)=({x},{y}); {seen/(printout_factor*num_printouts*n_iter)*100}% done")
                sg.write(f"FREQ {float(f)}")
                time.sleep(dwell)
                job.resume()
                counts_handle.wait_for_values(seen+1)
                all_counts = counts_handle.fetch_all()["value"]
                kcps.append(( all_counts[seen] / point_duration_s ) /1000 )
                seen+=1
            kcps_overall[x_ind,y_ind,:]=kcps

    return kcps_overall

# -------------------------
# ODMR analysis
# -------------------------
def counts_to_delta_freq(x_points, y_points, counts_2D, freqs):
    # TODO: use methods from Lorentzian_fit.py to implement this

    delta_freqs = np.zeros((len(x_points), len(y_points)), dtype=float)

    # something similar to:
    for x_ind in range(len(x_points)):
        for y_ind in range(len(y_points)):
            counts = counts_2D[x_ind,y_ind]
            max_peaks = 2
            popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
            contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
            # Lfit.print_SNR(baseline, counts, freqs / 10 ** 9, popt)
            # Lfit.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)
            if (len(dip_Freqs) == 2):
                # need exactly 2 dips to get the difference between the two
                delta_freqs[x_ind,y_ind] = dip_Freqs[1] - dip_Freqs[0]
            else:
                delta_freqs[x_ind, y_ind] = 0 # if you didn't get 2 dips there's no delta ig
    return delta_freqs

# -------------------------
# Image plotting
# -------------------------
def plot_image(x_points, y_points, freq_deltas_2D):
    # below stub written by Gemini
    plt.figure(figsize=(10, 6))
    # shading='auto' handles the coordinate mapping automatically
    mesh = plt.pcolormesh(x_points, y_points, freq_deltas_2D, shading='auto', cmap='viridis')

    plt.colorbar(mesh, label='frequency delta (GHz)')
    plt.xlabel('x space (mm)')
    plt.ylabel('y space (mm)')
    plt.title('frequency delta Heatmap')
    plt.show()

# -------------------------
# Saving
# -------------------------
def save_odmr_measurement(x_points, y_points, freqs, freq_deltas):
    now = datetime.datetime.now()
    datestamp = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    directory = os.path.join(project_root, "NVCFM_Data", datestamp)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = os.path.join(directory, f"scanned_cw_odmr_{timestamp}.npz")
    print(f"Saved as: scanned_cw_odmr_{timestamp}.npz")
    np.savez(save_path, x=x_points, y=y_points, f=freqs, deltas=freq_deltas)

#

def main():
    # -------------------------
    # Parameters
    # -------------------------
    # photo counts parameters
    readout_len_ns = int(50 * u.us)  # readout in ns
    n_windows_per_point = 5  # tbh idk what this does other than just increase the value of readout eln ^
    amp_dbm = -20

    dwell = 0.001  # s
    n_iter = 1

    f_center = 2.88e9 # Hz, generally near 2.87GHz
    f_span = 0.1e9 # Hz, range of frequencies to sample
    f_N = 21 # num points in the frequency space to sample

    # position parameters
    x_center,y_center = 4,4 # center of measurement
    x_span, y_span = 0.2,0.2 # range in each axis to sample
    x_N, y_N = 20,20 # num points in each axis to sample

    # sample name
    sample_name = 'scanned NV bulk'

    sg = cs.connect_sg386(sg_resource) # connect to RF src
    # connect to motors
    x_motor = cs.connect_motor(sample_name, x_motor_id)
    y_motor = cs.connect_motor(sample_name, y_motor_id)

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
        counts_2D = measure_all_points(sg, job, x_motor, y_motor, x_points, y_points, freqs, dwell, point_duration_s, n_iter=n_iter)
    finally:
        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)

    freq_deltas = counts_to_delta_freq(x_points, y_points, counts_2D, freqs)
    plot_image(x_points, y_points, freq_deltas)
    save_odmr_measurement(x_points, y_points, freqs, freq_deltas)

    # TODO: save freq_deltas + axes

if __name__ == "__main__":
    main()