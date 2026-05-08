from qm import QuantumMachinesManager
from qm.qua import *
from config import *
from qualang_tools.plot import interrupt_on_close
# For simulation
from qm import SimulationConfig
from qm import LoopbackInterface

import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
single_integration_time_ns = int(50 * u.us)   # 50 us time-tagging window
n_windows_per_point = 20                    # 50 us * 2000 = 100 ms per plotted point

# "single photon source" trigger settings
source_fire_prob = 0.05                      # 5% prob of firing pulse 
source_wait_cycles = 100                      # 100 cycles = 400 ns (1 cycle = 4 ns)
pulse_period_ns = source_wait_cycles * 4      # 400 ns

attempts_per_window = single_integration_time_ns // pulse_period_ns

# -------------------------
# QUA program
# -------------------------
with program() as spcm_counter:
    times = declare(int, size=10000)
    counts = declare(int)
    total_counts = declare(int)
    k = declare(int)
    n = declare(int)
    m = declare(int)
    counts_st = declare_stream()

    with for_(k, 0, k < 10, k + 1):
        # initialize on every 1 ms point 
        assign(total_counts, 0)

        # 50 us window loop
        with for_(n, 0, n < n_windows_per_point, n + 1):

            align("SPCM", "photon_source")

            # SPCM: 50 us time-tagging measure 
            # measure(
            #     "readout", "SPCM", None,
            #     time_tagging.analog(times, single_integration_time_ns, counts)
            # )

            # photon_source: same 50 us random trigger pulses
            with for_(m, 0, m < attempts_per_window, m + 1):
                play(
                    "pulse", "photon_source",
                    condition=Random().rand_fixed() > (1.0 - source_fire_prob),
                )
                wait(source_wait_cycles, "photon_source")
                assign(counts, counts + 1)

            # accumulate counts
            assign(total_counts, total_counts + counts)

        # send the result to stream
        save(total_counts, counts_st)

    with stream_processing():
        counts_st.save_all("counts")

# -------------------------
# Run
# -------------------------
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, log_level="INFO")

simulation_config = SimulationConfig(duration=10_000)
job = qmm.simulate(config, spcm_counter, simulation_config)

job.get_simulated_samples().con1.plot()
plt.show()

# -------------------------
# Live plotting
# -------------------------
# res_handles = job.result_handles
# counts_handle = res_handles.get("counts")
# counts_handle.wait_for_values(1)
# vals = counts_handle.fetch_all()
# print(vals)


res_handles = job.result_handles
counts_handle = res_handles.get("counts")
counts_handle.wait_for_values(1)
vals = res_handles.get("counts").fetch_all()
print(vals)

# point_duration_s = (single_integration_time_ns * n_windows_per_point) / 1e9  # 0.1 s

# t_list, kcps_list = [], []
# fig = plt.figure()
# interrupt_on_close(fig, job)

# while res_handles.is_processing():
#     new = counts_handle.fetch_new()
#     if len(new) == 0:
#         plt.pause(0.05)
#         continue

#     for v, ts in zip(new["value"], new["timestamp"]):
#         kcps_list.append((v / point_duration_s) / 1000.0)
#         t_list.append(ts / u.s)

#     plt.cla()
#     if len(t_list) > 300:
#         plt.plot(t_list[-300:], kcps_list[-300:])
#     else:
#         plt.plot(t_list, kcps_list)

#     plt.xlabel("time [s]")
#     plt.ylabel("counts [kcps]")
#     plt.title(f"SPCM Counter (source running during each {single_integration_time_ns/1000:.1f} us window)")
#     plt.pause(0.1)