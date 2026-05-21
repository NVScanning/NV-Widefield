import pco
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import connection_setup as cs


import os
import sys
sys.path.append(os.path.abspath(".."))
import nv_setup.cw_odmr.Lorentzian_fit as Lfit

"""
Two parts: 
first use a single pixel of the camera sensor as a point ODMR sensor
then use the whole camera sensor, and do all the ODMRS for all the pixels
"""

# TODO:
#   figure out whats wrong with the exposure/lorentzian fitting, perhaps initial guess params aren't good enough
#   Potential measurements:
#   vary the binning size from 1,2,4,8...2048 and compare contrasts/SNRs
#   vary the exposure time and compare contrasts/SNRs
#   vary both in a 2D array, and plot the contrasts/SNRs in a heatmap

# Constants
camera_resolution = 2048 # pixels, range from 1 tpo 2048 inclusive
sg_resource = "TCPIP::169.254.2.7::5025::SOCKET"

# params
focus_point_size = 500  # in pixels, diameter of circle of laser point
focus_point_centre_x, focus_point_centre_y = 1000, 1100  # in pixels, center point of the laser point
# TODO: maybe make use of 2D-gaussian to determine centre of focus point

def auto_expose(cam, target_intensity=0.9, tolerance=0.05, max_iter=10):
    max_val = 65535 # 2^16-1
    target = max_val * target_intensity # highest individual pixel brightness we want to allow

    for i in range(max_iter):
        cam.stop()
        cam.record(mode="sequence")
        image, meta = cam.image()
        peak = np.max(image)

        # Calculate ratio and adjust exposure
        if peak == 0:  # Prevent division by zero
            new_exposure = cam.exposure_time * 10 # aggressive, increases by exposure by factor of 10 if too dark
        else:
            ratio = target / peak
            if abs(1 - ratio) < tolerance:
                break
            # new_exposure = cam.configuration['exposure time'] * ratio
            new_exposure = cam.exposure_time * ratio

        # config = cam.configuration
        # config['exposure time'] = new_exposure
        # cam.configuration = config
        cam.exposure_time = new_exposure.item()

        print(f"Adjusting exposure to {new_exposure:.5f}s (Peak was: {peak}, now will be ~{peak * ratio})")


    return cam.exposure_time

def set_cam_settings(cam, exposure_time, roi=(1, 1, camera_resolution, camera_resolution), binning=(1, 1)):
    cam.configuration = {
        'exposure time': exposure_time,  # in seconds
        'roi': roi, # Region of Interest (x0, y0, x1, y1)
        # 'pixel rate': 100_000_000, # lowk no idea what this means
        'trigger': 'auto sequence',
        'binning': binning
    }

def read_image(cam,n_windows):
    cam.stop()
    # Re-arm specifically for this new capture
    cam.record(mode="sequence", number_of_images=n_windows)
    image = cam.image_average()
    return image

def setup_cam():
    cam = pco.Camera()
    set_cam_settings(cam, 10e-3)
    return cam

def plot_image(image, x_points=None, y_points=None):
    if x_points is None:
        x_points = np.arange(image.shape[1])
    if y_points is None:
        y_points = np.arange(image.shape[0])
    mesh = plt.pcolormesh(x_points, y_points, image, shading='nearest', cmap='gray') # use bone, gray, inferno, nihfire, viridis
    plt.colorbar(mesh, label='637nm brightness (arb units)')
    plt.gca().invert_yaxis()
    plt.xlabel('x space (pixels)')
    plt.ylabel('y space (pixels)')
    plt.title('camera image')
    plt.show()

def bin_image(image):
    return np.sum(image)
def select_one_pixel(image,x,y):
    return image[x,y]

def measure_odmr(cam, sg, freqs, dwell, point_duration_s, n_windows, x_points, y_points, n_iter: int = 1) -> np.ndarray:

    seen=0

    num_printouts = 10
    printout_factor = len(freqs)*n_iter*2 // num_printouts

    brightnesses = np.zeros((n_iter*2, len(x_points), len(y_points), freqs.size)) # should be n_iter*2 when reversing as well
    for i in range(n_iter):
        print("Iteration " + str(i))

        # brightness=[]
        for j,f in enumerate(freqs):
            if (seen % printout_factor == 0):
                # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
                print(f"at freq {str(f / 10 ** 9)}GHz; {seen/(printout_factor*num_printouts)*100:.1f}% done")
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = read_image(cam,n_windows)
            brightnesses[i,:,:,j]=image / point_duration_s/1000
            # plot_image(image)
            seen+=1
        for j,f in enumerate(freqs[::-1]):
            if (seen % printout_factor == 0):
                # Below approximation for %done isn't exact, but it gives round numbers which are easier to read
                print(f"at freq {str(f / 10 ** 9)}GHz; {seen/(printout_factor*num_printouts)*100:.1f}% done")
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)
            image = read_image(cam,n_windows)
            # plot_image(image)
            brightnesses[n_iter+i,:,:,j]=image / point_duration_s/1000
            seen+=1

    return np.sum(brightnesses,axis=0)/(n_iter*2)


def main():


    # x_points = np.arange(focus_point_centre_x-focus_point_size//2,focus_point_centre_x+focus_point_size//2,1)
    # y_points = np.arange(focus_point_centre_y-focus_point_size//2,focus_point_centre_y+focus_point_size//2,1)
    # region of interest, crop into this portion of the camera's view
    fps = focus_point_size//8*8 # forces fps to be a multiple of 8
    fpcx = focus_point_centre_x//8*8
    fpcy = focus_point_centre_y//8*8
    # x_measure, y_measure = fps//2+1, fps//2+1
    x_points = np.arange(fpcx-fps//2+1,fpcx+fps//2+1)
    y_points = np.arange(fpcy-fps//2+1,fpcy+fps//2+1)
    roi = (fpcx-fps//2+1, fpcy-fps//2+1,
           fpcx+fps//2, fpcy+fps//2)
    # roi=(1,1,2048,2048)
    print(f"Using the following roi: {roi}")

    n_windows_per_point = 10 # n readouts to increase certainty without overexposing
    # 40 windows with the 5mw laser is just about sufficient
    amp_dbm = -10 #anything bigger than -10 does nothing (Hayden)
    # Always use with 28V on the amplifier, amp_dbm ~30 is the lowest you can set while still seeing the zero-field dips
    # Larger amp means dips are more visible, but also get wider so you lose frequency resolution

    dwell =  0.001 # seconds - time between setting a frequency on fn generator and reading value
    n_iter = 1
    # frequency parameters
    f_center = 2.87e9 # Hz, generally near 2.87GHz
    span = 0.2e9 # Hz, range of frequencies to sample
    N = 101 # num points in the frequency space to sample


    # connect to RF src
    sg = cs.connect_sg386(sg_resource)
    # connect to cam
    cam = setup_cam()
    set_cam_settings(cam, 10e-3, roi=roi)
    exposure_time = auto_expose(cam, target_intensity=0.8) # returns exposure time in s
    # exposure_time = 0.0025 # roughly match SPCM exposure time
    print("Exposure time: ", exposure_time)
    set_cam_settings(cam, exposure_time, roi=roi)
    # cam.record(mode="sequence",number_of_images=n_windows_per_point)

    f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
    print("Frequency range from ", f_start/1e9, " to ", f_end/1e9, " GHz")
    point_duration_s = exposure_time * n_windows_per_point

    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(0.1) # why sleep for a whole second? (previous was 1)
    try:
        counts_2D = measure_odmr(cam, sg, freqs, dwell, point_duration_s, n_windows_per_point, x_points, y_points, n_iter)
    finally:
        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)
        cam.stop()

    print("Sweep done, now converting odmrs to B deltas")
    B_Z_overall, problem_points = Lfit.counts_to_B_Z(x_points, y_points, counts_2D, freqs)
    # cs.plot_odmr(freqs, counts)
    # Save data in folder with its date
    # cs.save_point_odmr_measurement(counts, freqs)
    print("Conversion done, saving and plotting")
    cs.save_2D_odmr_measurement(x_points, y_points, freqs, B_Z_overall, counts_2D)
    cs.plot_image(x_points, y_points, B_Z_overall)



    # max_peaks = 2
    # popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
    # c0, c1 = popt[-2], popt[-1]
    # print(f"baseline: {c0} + {c1}f[GHz]")
    # Lfit.print_dip_params(popt)


    # try:
    #     Lfit.print_SNR(baseline, counts, freqs/10**9, popt)
    # except ValueError as e:
    #     # do nothing cuz printing snr didn't work
    #     print("getting SNR failed: " + str(e))
    # Lfit.plot_fitted_data(freqs/10**9, counts_norm, fitted_norm)



if __name__ == "__main__":
    main()