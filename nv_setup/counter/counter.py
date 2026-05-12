from qm import QuantumMachinesManager
from qm.qua import *
from config import *
from qualang_tools.plot import interrupt_on_close

import matplotlib.pyplot as plt

"""
This is a debug file, which constantly counts using the SPCM (single photon counting module), and plots the last ~60 seconds of counts at ~200ms intervals

Used for optical alignment of laser path as well as of the NV itself
"""

# -------------------------
# Parameters
# -------------------------
single_integration_time_ns = int(50 * u.us)   # 50 us time-tagging window
n_windows_per_point = 2000                    # 2000 * 50 us = 100 ms per plotted point
num_points = 500

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

    new_counts = counts_handle.fetch_all() 
    kcps_list.append((new_counts["value"] / point_duration_s) /1000 )
    t_list.append(new_counts["timestamp"] / u.s)  # Convert timestamps to seconds

    plt.cla()
    plt.plot(t_list[-num_points:], kcps_list[-num_points:]) if len(t_list) > num_points else plt.plot(t_list, kcps_list)
    plt.xlabel("time [s]")
    plt.ylabel("counts [kcps]")
    # plt.ylim(17500,17600) # for debugging, remove
    plt.title("SPCM Counter")
    plt.pause(0.1)

print(f"Average: {np.average(kcps_list)}")
print(f"Error bar: {np.std(kcps_list)}")