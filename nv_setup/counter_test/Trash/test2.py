from qm import QuantumMachinesManager
from qm.qua import *
from config_sim import *
from qualang_tools.plot import interrupt_on_close

# For simulation
from qm import SimulationConfig
from qm import LoopbackInterface

import matplotlib.pyplot as plt

with program() as p:
    times = declare(int, size=1000)
    counts = declare(int)
    st = declare_stream()

    align("SPCM", "photon_source")
    play("pulse", "photon_source")

    measure("readout", "SPCM", None,
            time_tagging.analog(times, readout_len, counts))

    save(counts, st)

    with stream_processing():
        st.save_all("counts")

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, log_level="INFO")

simulation_config = SimulationConfig(duration=10_000)

job = qmm.simulate(config, p, simulation_config)

job.get_simulated_samples().con1.plot()
plt.show()

res_handles = job.result_handles
counts_handle = res_handles.get("counts")
counts_handle.wait_for_values(1)
vals = counts_handle.fetch_all()

print(vals)
