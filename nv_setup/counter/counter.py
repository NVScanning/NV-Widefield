from qm import QuantumMachinesManager
from qm.qua import *
from config import *
from qualang_tools.plot import interrupt_on_close

import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
single_integration_time_ns = int(50 * u.us)   # 50 us time-tagging window
n_windows_per_point = 2000                    # 100 ms per plotted point

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
    plt.plot(t_list[-300:], kcps_list[-300:]) if len(t_list) > 300 else plt.plot(t_list, kcps_list)
    plt.xlabel("time [s]")
    plt.ylabel("counts [kcps]")
    plt.title("SPCM Counter")
    plt.pause(0.1)

print(f"Average: {np.average(kcps_list)}")
print(f"Error bar: {np.std(kcps_list)}")