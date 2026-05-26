
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(".."))
import nv_setup.connection_setup as cs
import pco_cam_interface as pci

# Todo:
#   Vary the z in tiny steps, and move to the optimal position (prolly maximize brightness) (can mostly copy-paste the z-counter code from before, binning the whole camera)
#   vary the binning size from 1,2,4,8...2048 and compare contrasts/SNRs
#   vary the exposure time and compare contrasts/SNRs
#   vary both in a 2D array, and plot the contrasts/SNRs in a heatmap

def plot_graph(z_range: np.ndarray, kcps: np.ndarray):

    plt.figure(figsize=(8, 5))
    plt.plot(z_range, kcps, "-o", markersize=2)
    plt.xlabel("Position (mm)")
    plt.ylabel("kcps")
    plt.title("Counts as a fn of Z")
    plt.grid(True)
    plt.show()

def find_best_z(cam, motor, z_range, dwell, point_duration_s, n_windows):
    # TODO: convert this to work with camera instead
    seen = 0
    num_printouts = 5
    brightnesses = np.zeros(z_range.size)
    printout_factor = len(z_range) // num_printouts
    motor.move_to(z_range[0])
    time.sleep(5) # move to starting position, before connecting cam
    for z in z_range:
        if (seen % 10 == 0):
            print(f"at z={z:.4f}mm {seen/(printout_factor*num_printouts)*100:.1f}% done")
        motor.move_to(z)
        time.sleep(dwell)
        # print('count seen')
        image = pci.read_image(cam,n_windows)
        all_counts = pci.bin_image(image)
        brightnesses[seen] = (all_counts / point_duration_s/1000)
        seen += 1
    max_idx = np.argmax(brightnesses)
    optimized_z_pos = z_range[max_idx]
    return brightnesses, optimized_z_pos, max(brightnesses)

def optimize_z():
    print("Optimizing Z")
    z_motor = cs.connect_motor(cs.z_mID)
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 512  # in pixels, diameter of circle of laser point, must be a multiple of either 4 or 16
    focus_point_centre_x, focus_point_centre_y = 1000, 1100  # in pixels, center point of the laser point
    n_windows_per_point = 1
    dwell = 0.2 # s
    # position z parameters
    z_center = 5.898
    span = 0.05
    N = 51

    z_start, z_end, z_range = cs.calc_sweep_range(z_center, span, N)
    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    z_motor.move_to(z_center)
    time.sleep(5) # move to rough middle, to have good auto exposure time
    cam, exposure_time, sg = pci.connect_cam_RF(roi, binning_amount)
    point_duration_s = exposure_time * n_windows_per_point
    print(f"Optimizing Z, estimate time to completion ~ {N * (dwell * point_duration_s) + 10}s")
    # here exact exposure time doesn't matter, as long as its constant throughout the range
    brightnesses, optimized_z_pos, max_brightness = find_best_z(cam, z_motor, z_range, dwell, point_duration_s, n_windows_per_point)
    cam.stop()
    print(f"optimal z was found at z={optimized_z_pos:.4f}mm, with a brightness of {max_brightness:.2f}")
    plot_graph(z_range, brightnesses)

    ans = input("Move to best position? (Y/N): ").strip().lower()

    if ans == "y":
        z_motor.move_to(optimized_z_pos)
        time.sleep(0.5)
        print("Moved to optimized position and state saved.")
    else:
        print("Stayed at current position.")
    return

def main():
    # do things, perhaps only one at a time or all at once idk:
    # 1: vary z, record the whole binned image at each step (with very small steps, <10um)
    # 2: keep exposure time constant, and vary the post-processed binning (probably in an exponential stepsize)
    #   for each pixel, do the ODMR and record SNR&contrast, then average them between all pixels, and record these numbers
    # 3: keep binning to 1x1, and vary total exposure time (change n_windows_per_point, force exposure time to a constant ~


    # 1: optimizing z:
    optimize_z()

    # 2:



if __name__ == "__main__":
    main()