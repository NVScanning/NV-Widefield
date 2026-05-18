from qm import QuantumMachinesManager
from qm.qua import *
from config import *
from qualang_tools.plot import interrupt_on_close

import matplotlib.pyplot as plt

import datetime
from pathlib import Path


"""
This is a debug file, which constantly counts using the SPCM (single photon counting module), and plots the last ~60 seconds of counts at ~200ms intervals

Used for optical alignment of laser path as well as of the NV itself
"""

# -------------------------
# Parameters
# -------------------------
single_integration_time_ns = int(50 * u.us)   # 50 us time-tagging window
n_windows_per_point = 2000                    # 2000 * 50 us = 100 ms per plotted point
num_mins = 2
num_points = 300*num_mins # 300 points is ~ 1 minute with 0.1 pause
save_fig = False

# -------------------------
# QUA program
# -------------------------
with program() as spcm_counter:
    times = declare(int, size=10000)
    counts = declare(int)
    total_counts = declare(int)
    n = declare(int)
    counts_st = declare_stream()

    with infinite_loop_():
        assign(total_counts, 0)

        with for_(n, 0, n < n_windows_per_point, n + 1):
            measure( "readout", "SPCM", time_tagging.analog(times, single_integration_time_ns, counts))
            assign(total_counts, total_counts + counts)

        save(total_counts, counts_st)

    with stream_processing():
        counts_st.with_timestamps().save("counts")

# -------------------------
# Execute program on QM
# -------------------------
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, log_level="INFO")
qm = qmm.open_qm(config)
job = qm.execute(spcm_counter)

# -------------------------
# Live plotting
# -------------------------
# TODO: add an averaging step, get points with uncertainty, maybe a rolling average

res_handles = job.result_handles
counts_handle = res_handles.get("counts")
counts_handle.wait_for_values(1)

point_duration_s = (single_integration_time_ns * n_windows_per_point) / 1e9

t_list, kcps_list = [], []
fig = plt.figure()
interrupt_on_close(fig, job)


while res_handles.is_processing():


    if save_fig:
        now = datetime.datetime.now()
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent
        datestamp = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%H-%M-%S")
        directory = os.path.join(project_root, "nv_setup/counter/figures", datestamp)
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = os.path.join(directory, f"counts_{timestamp}")
        # Below is for debugging, it saves the figure after numpoints elapses
        if len(t_list) > num_points :
            print("got " + str(len(t_list)) + " points, saving fig")
            plt.cla()
            plt.plot(t_list[-num_points:], kcps_list[-num_points:]) if len(t_list) > num_points else plt.plot(t_list, kcps_list)
            plt.xlabel("time [s]")
            plt.ylabel("counts [kcps]")
            plt.title("SPCM Counter")
            plt.savefig(save_path + ".png")
            np.save(save_path + ".npy", kcps_list)
            quit()



    new_counts = counts_handle.fetch_all() 
    kcps_list.append((new_counts["value"] / point_duration_s) /1000 )
    t_list.append(new_counts["timestamp"] / u.s)  # Convert timestamps to seconds

    plt.cla()
    plt.xlabel("time [s]")
    plt.ylabel("counts [kcps]")
    # plt.ylim(17500,17600) # for debugging, remove
    plt.title("SPCM Counter")
    plt.plot(t_list[-num_points:], kcps_list[-num_points:]) if len(t_list) > num_points else plt.plot(t_list, kcps_list)
    plt.pause(0.1)

print(f"Average: {np.average(kcps_list)}")
print(f"Error bar: {np.std(kcps_list)}")