from qm import QuantumMachinesManager
from qm.qua import *
from config_sim import *
from qualang_tools.plot import interrupt_on_close

# For simulation
from qm import SimulationConfig
from qm import LoopbackInterface

import matplotlib.pyplot as plt

single_integration_time_ns = int(1 * u.us)   # 1 us time-tagging window
n_windows_per_point = 20                     # 1 us * 20 = 20 us per plotted point

# "single photon source" trigger settings
source_fire_prob = 0.05                      # 5% prob of firing pulse 
source_wait_cycles = 25                      # 25 cycles = 100 ns (1 cycle = 4 ns)
pulse_period_ns = source_wait_cycles * 4     # 100 ns

attempts_per_window = single_integration_time_ns // pulse_period_ns   # 10 times

# -------------------------
# QUA program
# -------------------------
with program() as spcm_counter:
    counts = declare(int)
    total_counts = declare(int)
    times = declare(int, size=10000)
    n = declare(int)
    m = declare(int)
    counts_st = declare_stream()

    with infinite_loop_():
        assign(total_counts, 0)

        with for_(n, 0, n < n_windows_per_point, n + 1):

            assign(counts, 0)
            # align("SPCM", "photon_source")
            measure(
                "readout", "SPCM", None,
                time_tagging.analog(times, single_integration_time_ns, counts)
            )

            with for_(m, 0, m < attempts_per_window, m + 1):
                play("pulse", "photon_source", condition= Random().rand_fixed() > (1.0 - source_fire_prob))
                wait(source_wait_cycles, "photon_source")
            # accumulate counts
            assign(total_counts, total_counts + counts)

        # send the result to stream
        save(total_counts, counts_st)

    with stream_processing():
        counts_st.save_all("counts")


qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, log_level="INFO")

simulate_config = SimulationConfig(
    duration=20000,
    simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)]),
)

job = qmm.simulate(config, spcm_counter, simulate_config)
job.get_simulated_samples().con1.plot()
plt.show()

res_handles = job.result_handles
counts_handle = res_handles.get("counts")
counts_handle.wait_for_values(1)
vals = counts_handle.fetch_all()

print(vals)

