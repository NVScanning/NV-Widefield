import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import scipy.ndimage as ndimage
sys.path.append(os.path.abspath(".."))
import helper_classes.pco_cam_interface as pci
import scipy.fft
import nv_widefield.helper_classes.odmr_plotting as oPlot


"""
Repeatedly take images from the camera and repeatedly (~1Hz) plot a few params from then:
1. bin them together to one "brightness"
2. find peak pixel value
3. take variance of the laplacian, helpful for on-sensor focus 
    ^ not very useful once I realized on-sensor focus is irrelevant for ODMR contrast
    in favour of on-diamond focus, which can only (afaik) be determined by measuring ODMR contrast
    therefore this plotting has been removed
"""

num_points = 6000
roi = None

t0 = time.time()
timestamps = []
brightnesses = []

pixels = [(35,35),(35,10),(10,35)]

def get_new_data(cam,n_windows,point_duration_s):

    cam.wait_for_new_image()
    curr_img_num = cam.recorded_image_count
    image = pci.get_image_sub_bkg(cam)
    images = np.zeros((image.shape[0], image.shape[1], n_windows))
    for i in range(n_windows):
        if cam.recorded_image_count == curr_img_num:
            cam.wait_for_new_image()
        curr_img_num = cam.recorded_image_count
        images[:,:,i] = pci.get_image_sub_bkg(cam)

    global brightnesses
    new_frame_3d = np.expand_dims(images.sum(axis=2) / point_duration_s, axis=2)

    # Append along the third (time/step) dimension
    brightnesses = np.concatenate((brightnesses, new_frame_3d / point_duration_s), axis=2)
    # brightnesses = images
    timestamps.append(time.time() - t0)

def plot_graphs():
    # images = np.array(brightnesses)
    fig = plt.figure(figsize=(8, 5), layout='constrained')  # (width, height) in inches


    # ax2 = ax1.twinx()
    # ax3 = ax1.twinx()

    plt.xlabel("time [s]")
    plt.ylabel("pixel brightness [counts/s]")

    color1, color2, color3 = plt.cm.viridis([0, .5, .9])

    global brightnesses

    plt.plot(timestamps, brightnesses[*pixels[2]],color=color3, markersize=5, label="left middle")
    plt.plot(timestamps, brightnesses[*pixels[1]],color=color2, markersize=5, label="lower middle")
    plt.plot(timestamps, brightnesses[*pixels[0]],color=color1, markersize=5, label="centre")

    # ax3.yaxis.label.set_color(p3[0].get_color())
    # ax2.yaxis.label.set_color(p2[0].get_color())
    # ax1.yaxis.label.set_color(p1[0].get_color())

    # ax1.tick_params(axis='y', colors=p1[0].get_color())

    plt.legend()

    plt.show()

    plt.close(fig)

def main():
    n_windows_per_point = 1 # n readouts to increase certainty without overexposing
    binning_amount = 4 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 288  # in pixels, approximate width of image taken, must be >=32 after binning
    focus_point_centre_x, focus_point_centre_y = 830,1070 # in pixels, center of the laser point
    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    # roi = (1,1,2048//binning_amount,2048//binning_amount)
    if roi is not None:
        print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    else:
        print("Using previous roi")

    cam = pci.connect_cam(roi, binning_amount, 0.02)


    point_duration_s = cam.exposure_time * n_windows_per_point

    # time.sleep(0.1)
    global brightnesses
    brightnesses = np.empty((len(x_space), len(y_space),0))

    try:
        cam.record(mode="ring buffer", number_of_images=n_windows_per_point * 10)
        cam.wait_for_new_image()
        imgnum=0
        for _ in range(num_points):
            get_new_data(cam,n_windows_per_point,point_duration_s)
            if imgnum%30==0:
                plot_graphs()
            imgnum += 1
    # except Exception as e:
    #     print("threw error: ", e)
    finally:

        print("Finished collecting data")
        cam.stop()
        cam.close()

        # images = np.array(brightnesses)


        # Analyze + Plot FFTs
        df = abs(timestamps[1] - timestamps[0])
        # df = point_duration_s
        for pixel in pixels:
            fft = scipy.fft.fft(brightnesses[*pixel])
            fft_times = scipy.fft.fftfreq(len(timestamps), d=df)
            fft_shifted = scipy.fft.fftshift(fft)
            times_shifted = scipy.fft.fftshift(fft_times)

            magnitude = np.abs(fft_shifted)
            pos_mask = times_shifted > 0
            plot_times = times_shifted[pos_mask]
            plot_mag = magnitude[pos_mask]

            plt.figure(figsize=(6, 5))
            plt.plot(plot_times, plot_mag, marker='.', color='tab:red')
            plt.xlabel("Modulation Frequency [cycles / s]", fontsize=12)
            plt.ylabel("FFT Magnitude", fontsize=12)
            plt.title(f"Fourier Transform of ODMR at (x,y)={pixel[0]},{pixel[1]}", fontsize=13, fontweight='bold')
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()




if __name__ == "__main__":
    main()