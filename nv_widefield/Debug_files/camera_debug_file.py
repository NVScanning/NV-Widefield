import numpy as np
import time
import connection_setup as cs
import os
import sys
sys.path.append(os.path.abspath(".."))
import helper_classes.pco_cam_interface as pci
import matplotlib.pyplot as plt


"""
Debug file, ignore method names

Measure difference between counts at different RF frequencies
"""


def measure_odmr(cam, sg, freqs, dwell, point_duration_s, n_windows, n_iter: int = 1) -> np.ndarray:
    print(f"measuring ODMR, estimate time to completion ~{n_iter * len(freqs) * (dwell + point_duration_s):.2f}s")
    seen=0
    image = pci.read_image(cam,1)

    brightnesses = np.zeros((n_iter*2, image.shape[0], image.shape[1], freqs.size)) # should be n_iter*2 when reversing as well
    for i in range(n_iter):

        for j,f in enumerate(freqs):
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = pci.read_image(cam,n_windows)
            brightnesses[i,:,:,j]=image / point_duration_s/1000
            # pci.plot_image(image)
            seen+=1

    return np.sum(brightnesses,axis=0)/(n_iter*2)

def plot_counts_image(x_points, y_points, counts_diff):
    plt.figure(figsize=(10, 6))
    # shading='auto' handles the coordinate mapping automatically
    mesh = plt.pcolormesh(x_points, y_points, counts_diff, shading='nearest', cmap='inferno')

    plt.colorbar(mesh, label='counts')
    plt.xlabel('x space (mm)')
    plt.ylabel('y space (mm)')
    plt.title('relative count difference Heatmap')
    plt.show()



def main():
    # params
    binning_amount = 1 # built-int pco camera binning, can only be 1,2,4
    focus_point_size = 64  # in physical (unbinned) pixels, diameter of circle of laser point
    focus_point_centre_x, focus_point_centre_y = 1016, 1024  # in pixels, center point of the laser point
    n_windows_per_point = 1000 # n readouts to increase certainty without overexposing
    amp_dbm = -20 #anything bigger than -10 does nothing (Hayden)
    # Always use with 28V on the amplifier, amp_dbm ~30 is the lowest you can set while still seeing the zero-field dips
    # Larger amp means dips are more visible, but also get wider so you lose frequency resolution
    dwell =  0.0 # seconds - time between setting a frequency on fn generator and reading value
    n_iter = 1

    roi, x_space, y_space = pci.get_spacial_params(binning_amount,(focus_point_size, focus_point_centre_x, focus_point_centre_y))
    print(f"Using the following roi: {roi} and binning a {binning_amount}x{binning_amount} region")
    cam, sg = pci.connect_cam_RF(roi, binning_amount)
    f_start, f_end, freqs = 2.87*10**9, 2.88*10**9, np.array([2.87,2.88])*10**9
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")
    point_duration_s = cam.exposure_time * n_windows_per_point

    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(0.1) # why sleep for a whole second? (previous was 1)
    try:
        counts_2D = measure_odmr(cam, sg, freqs, dwell, point_duration_s, n_windows_per_point, n_iter)
    finally:
        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)

    counts_diff = (counts_2D[:,:,1] - counts_2D[:,:,0])/counts_2D[:,:,0]
    plot_counts_image(x_space, y_space, counts_diff)




if __name__ == "__main__":
    main()