from qm.qua import *
import numpy as np

# -------------------------
# QUA program
# -------------------------

def odmr_qua_program(num_points, n_windows_per_point, readout_len_ns):
    # Queue up the exact number of requested measurements in the quantum machine,
    # so the measure_odmr can resume exeactly as many times as it needs

    with program() as odmr_counts:
        times = declare(int, size=10000) # should size=N_freq*n_iter?
        counts = declare(int)
        total_counts = declare(int)

        iteration = declare(int)  # Outer loop index
        i = declare(int)
        k = declare(int)
        counts_st = declare_stream()

        # Outer loop ensures the job doesn't finish until all iterations are done
        with for_(i, 0, i < num_points, i + 1):
            pause()
            assign(total_counts, 0)
            with for_(k, 0, k < n_windows_per_point, k + 1):
                measure("readout", "SPCM", time_tagging.analog(times, readout_len_ns, counts))
                assign(total_counts, total_counts + counts)

                save(total_counts, counts_st)

        with stream_processing():
            counts_st.save_all("counts")

    return odmr_counts


# -------------------------
# Helpers
# -------------------------

def calc_sweep_range(center: float, span: float, num_points: int):
    start = center - span / 2
    end = center + span / 2
    point_array = np.linspace(start, end, num_points)
    return start, end, point_array
