import time
import connection_setup as cs
import helper_classes.Log as Log
import numpy as np
import sys


def run_odmr_measurement(amp_dbm, fn, odmr_params):
    # Safely runs measurement with cam and sg, turning everything off correctly at the end

    sg = cs.connect_sg386(cs.sg_resource)
    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(0.1)

    try:
        ret_vals = fn(sg, *odmr_params)
    finally:
        cs.enable_sg386(sg, enable=False)
    return ret_vals


def sweep_freqs(sg, dwell, freqs, cam_frame_time, n_windows, n_iter, iteration, t0=None) -> npt.NDArray[np.float64]:
    point_duration_s = cam_frame_time * n_windows
    time.sleep(dwell*3) # camera heats slightly
    for j, f in enumerate(freqs):
        Log.log("at freq " + str(np.round(f/10**9,2)) + "GHz, updating progress")
        if t0 is not None:
            cs.print_odmr_progress(iteration * len(freqs) + j, len(freqs) * n_iter, iteration, f, time.time()-t0)
        else:
            cs.print_odmr_progress(iteration * len(freqs) + j, len(freqs) * n_iter, iteration, f, None)

        # turn the RF off for some time before the exposure to cool it down
        # Log.log("turning sg386 off")
        # cs.enable_sg386(sg, enable=False)
        # time.sleep(0.1)
        # cs.enable_sg386(sg, enable=True)

        Log.log("setting new freq on sg386")
        # supposedly this value will wrap around so I shouldn't get an index-out-of-bounds error when using it to get img
        sg.write(f"FREQ {float(f)}")
        time.sleep(dwell)
        Log.log("starting cam exposure")
        for i in range(n_windows):
            time.sleep(cam_frame_time)

        # this normalizes counts between each frame, to account for any RF heating:

        # Log.log("turning sg386 off for background measurement")
        # cs.enable_sg386(sg, enable=False)
        # time.sleep(dwell)
        # cam.wait_for_new_image()
        # Log.log("getting background measurement")
        # img_off = get_image_sub_bkg(cam)
        # counts_off = bin_image(img_off)

    return



def measure_odmr(sg, freqs, dwell, cam_frame_time, n_windows, n_iter: int = 1) -> np.ndarray:
    point_duration_s = cam_frame_time * n_windows
    print(f"measuring binned ODMR with {n_iter} iteration(s), estimate time to completion"
          f" ~{(n_iter)*2 * ((len(freqs) + 1) * (dwell + point_duration_s) + 0.02):.0f}s")
    # seen=0

    # num_printouts = 5
    # printout_factor = len(freqs)*n_iter*2 // num_printouts

    # first ODMR sweep is fucked, throw it out
    sweep_freqs(sg, dwell, freqs, cam_frame_time, n_windows, 1, 0)
    # sweep_freqs_binned_ringBuf(cam, sg, dwell, freqs[::-1], n_windows, n_iter * 2,0 * 2 + 1)

    t0 = time.time()
    brightnesses = np.zeros((n_iter*2, freqs.size)) # should be n_iter*2 when reversing as well
    for i in range(n_iter):
        # print("Iteration " + str(i))
        # time.sleep(dwell)
        # Log.log("Before reading image")
        # pci.read_image(cam,n_windows) # ignore = first image - it has higher intensity for some reason
        # Log.log("after reading image, before sweeping freqs")
        Log.log("Sweeping freqs:")
        brightnesses[i] = sweep_freqs(sg, dwell, freqs, cam_frame_time, n_windows, n_iter * 2, i * 2, t0)
        brightnesses[n_iter + i] = sweep_freqs(sg, dwell, freqs[::-1], cam_frame_time, n_windows, n_iter * 2, i * 2 + 1, t0)[::-1]

        # brightnesses[i] = sweep_freqs(sg, dwell, freqs[::-1], cam_frame_time, n_windows, n_iter * 2, i * 2, t0)[::-1]
        # brightnesses[n_iter + i] = sweep_freqs(sg, dwell, freqs, cam_frame_time, n_windows, n_iter * 2, i * 2 + 1, t0)

        Log.log("after sweeping freqs")



        # TODO: have it save partial measurements after each iteration, like widefield does
    sys.stdout.write(f"\r\033[KODMR finished, took {time.time()-t0:.0f}s\n") # Clear progress bar
    sys.stdout.flush()
    return np.sum(brightnesses,axis=0)/(n_iter*2)


def main():
    # Log.start()

    # params
    cam_frame_time = 0.005 #camera exposure time in s

    n_windows_per_point = 3 # n readouts to increase certainty without overexposing
    amp_dbm = -10 #anything bigger than -10 does nothing (Hayden)
    # Always use with 28V on the amplifier, amp_dbm ~30 is the lowest you can set while still seeing the zero-field dips
    # Larger amp means dips are more visible, but also get wider so you lose frequency resolution

    dwell =  0.04 # seconds - time between setting a frequency on fn generator and reading value
    n_iter = 10
    # frequency parameters
    f_center = 2.87e9 # Hz, generally near 2.87GHz
    span = 3e9 # Hz, range of frequencies to sample
    N = 601 # num points in the frequency space to sample


    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    # freqs = np.linspace(f_center, f_center + span, N) # only measure half the range for less wasted measurement time
    # f_start = f_center
    # f_end = f_center + span
    # print(f"Frequency range from {f_start/1e9:.3f} to {f_end/1e9:.3f}GHz")

    run_odmr_measurement(amp_dbm, measure_odmr, (freqs, dwell, cam_frame_time, n_windows_per_point, n_iter))




if __name__ == "__main__":
    main()