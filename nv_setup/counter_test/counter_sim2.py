from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration2 import *
from qm import LoopbackInterface

###################
# The QUA program #
###################

total_integration_time = int(100 * u.ms)  # 100ms
single_integration_time_ns = int(50 * u.us)  # 500us
single_integration_time_cycles = single_integration_time_ns // 4
n_count = int(total_integration_time / single_integration_time_ns)

simulate = True

if simulate:
    n_count = 20
    single_integration_time_ns = 5000

with program() as counter:
    times = declare(int, size=10000)
    counts = declare(int)
    total_counts = declare(int)
    n = declare(int)
    m = declare(int)
    counts_st = declare_stream()
    with infinite_loop_():
        with for_(m, 0, m < 10, m + 1):
            play("gauss", "photon_source", condition= Random().rand_fixed() > 0.95)  # plays single_photon operation on qubit
            wait(100, "photon_source")  # qubit waits 4 clock cycles (16 ns)

    with infinite_loop_():
        with for_(n, 0, n < n_count, n + 1):
            measure("readout", "SPCM", None, time_tagging.analog(times, single_integration_time_ns, counts))
            assign(total_counts, total_counts + counts)
        save(total_counts, counts_st)
        assign(total_counts, 0)

    with stream_processing():
        if simulate:
            counts_st.save_all("counts")
        else:
            counts_st.with_timestamps().save("counts")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host='192.168.88.250', port='80',log_level='DEBUG')



if simulate:
    simulate_config = SimulationConfig(
        duration=int(20000),
        simulation_interface=LoopbackInterface(
            ([("con1", 1, "con1", 1)])
        ),
    )  # simulation properties
    job_sim = qmm.simulate(config, counter, simulate_config)
    # Simulate blocks python until the simulation is done
    job_sim.get_simulated_samples().con1.plot()
    plt.show()
    res_handle = job_sim.result_handles  # creates handles to access results
    res_handle.wait_for_all_values()  # wait for all values to arrive before granting access to results
    counts = res_handle.get(
        "counts"
    ).fetch_all().tolist()  # fetches number of single-photon events
    plt.figure()
    plt.plot(counts)

else:
    qm = qmm.open_qm(config)

    job = qm.execute(counter)
    # Get results from QUA program
    res_handles = job.result_handles
    counts_handle = res_handles.get("counts")
    counts_handle.wait_for_values(1)
    time = []
    counts = []
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while res_handles.is_processing():
        new_counts = counts_handle.fetch_all() 
        counts.append((new_counts["value"] / (total_integration_time / 1000000000)) /1000 )
        time.append(new_counts["timestamp"] / u.s)  # Convert timestams to seconds
        plt.cla()
        if len(time) > 300:
            plt.plot(time[-300:], counts[-300:])
        else:
            plt.plot(time, counts)

        plt.xlabel("time [s]")
        plt.ylabel("counts [kcps]")
        plt.title("Counter")
        plt.pause(0.1)