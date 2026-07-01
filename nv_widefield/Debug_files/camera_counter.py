import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import scipy.ndimage as ndimage
sys.path.append(os.path.abspath(".."))
import helper_classes.pco_cam_interface as pci


"""
Repeatedly take images from the camera and repeatedly (~1Hz) plot a few params from then:
1. bin them together to one "brightness"
2. find peak pixel value
3. take variance of the laplacian, helpful for on-sensor focus 
    ^ not very useful once I realized on-sensor focus is irrelevant for ODMR contrast
    in favour of on-diamond focus, which can only (afaik) be determined by measuring ODMR contrast
    therefore this plotting has been removed
"""

num_points = 9000
roi = None

t0 = time.time()
timestamps = []
total_brightness = []
peak_brightness = []
laplacian_variance = []

def get_new_data(cam,n_windows,point_duration_s):
    image = pci.read_image(cam, n_windows)
    # pci.plot_image(image)
    laplacian = ndimage.laplace(image.astype(float))
    lScore = laplacian.var()
    all_counts = pci.bin_image(image)

    total_brightness.append(all_counts / point_duration_s / 1000)
    peak_brightness.append(np.max(image)) # not divided by exposure time or 1000
    laplacian_variance.append(lScore)
    timestamps.append(time.time() - t0)

def plot_graphs():
    fig, ax1 = plt.subplots(figsize=(8, 5), layout='constrained')  # (width, height) in inches

    ax2 = ax1.twinx()
    # ax3 = ax1.twinx()

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("total brightness [sum vals/s]")
    ax2.set_ylabel("peak brightness [pixel val]")
    # ax3.set_ylabel("laplacian variance [unitless?]")

    # right, left, top, bottom
    # ax3.spines['right'].set_position(('outward', 60))

    # color1, color2, color3 = plt.cm.viridis([0, .5, .9])
    color1, color2= plt.cm.viridis([0, .5])

    if len(timestamps)>num_points:
        # p3 = ax3.plot(timestamps[-num_points:], laplacian_variance[-num_points:],color=color3)
        p2 = ax2.plot(timestamps[-num_points:], peak_brightness[-num_points:],color=color2)
        p1 = ax1.plot(timestamps[-num_points:], total_brightness[-num_points:],color=color1)
    else :
        # p3 = ax3.plot(timestamps, laplacian_variance,color=color3)
        p2 = ax2.plot(timestamps, peak_brightness,color=color2)
        p1 = ax1.plot(timestamps, total_brightness,color=color1)

    ax1.yaxis.label.set_color(p1[0].get_color())
    ax2.yaxis.label.set_color(p2[0].get_color())
    # ax3.yaxis.label.set_color(p3[0].get_color())

    ax1.tick_params(axis='y', colors=p1[0].get_color())
    ax2.tick_params(axis='y', colors=p2[0].get_color())
    # ax3.tick_params(axis='y', colors=p3[0].get_color())

    plt.show()
    # plt.ion()
    # plt.show(block=False)

    # plt.pause(0.2)
    plt.close(fig)

def main():
    n_windows_per_point = 1 # n readouts to increase certainty without overexposing
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 2048  # in pixels, approximate width of image taken, must be >=32 after binning
    focus_point_centre_x, focus_point_centre_y = 1024,1024  # in pixels, center of the laser point
    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    roi = (1,1,2048,2048)
    if roi is not None:
        print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    else:
        print("Using previous roi")

    cam = pci.connect_cam(roi, binning_amount, 0.001)


    point_duration_s = cam.exposure_time * n_windows_per_point

    # time.sleep(0.1)

    try:
        while(True):
            # for i in range(100):
            #     get_new_data(cam,n_windows_per_point,point_duration_s)
            get_new_data(cam,n_windows_per_point,point_duration_s)
            plot_graphs()
    finally:
        # cam.stop()
        cam.close()




if __name__ == "__main__":
    main()