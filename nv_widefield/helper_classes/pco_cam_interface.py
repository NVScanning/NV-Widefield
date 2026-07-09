import time
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pco
from pco import Camera, camera_exception
import connection_setup as cs
import sys
import helper_classes.Log as Log
import os
"""
Main interface for using the PCO camera, many functions are defined for repeated usage 
such as setting up the camera or running a frequency sweep
"""

# Constants
camera_resolution = 2048 # pixels, range from 1 tpo 2048 inclusive
extra_row_size = 8
max_pixel_val = 65535 # 2^16-1
min_roi_dims = 32
objective_magnification = 50
dark_frame_path = "C:\\Users\\NVCFM\\Desktop\\NV-widefield Experiment\\nv_widefield\\helper_classes\\master_dark.npy"
master_dark_frame: npt.NDArray[np.float64] | None
master_roi: tuple[int, int, int, int] | None
master_bin: int | None
master_dark_frame = None
master_roi = None
master_bin = None

# background_rate = 0 # 5600 000 / 128^2  # total brightness divided by num physical pixels divided by exposure time
# # ^ This num ends up being like 341.8 counts/s on each pixel
# # There is a clear pattern in the dark currents, so maybe record a picture one time and save it
# # as a 2D array, then index into it with the relevant ROI and subtract from each image directly here

def get_image_sub_bkg(cam):
    image, dict = cam.image(image_index=-1)  # changed to always get the newest image
    if master_dark_frame is not None:
        image = image - cam.exposure_time * master_dark_frame[
            (master_roi[1] - 1) // master_bin:master_roi[3] // master_bin, (master_roi[0] - 1) // master_bin:
                                                                           master_roi[2] // master_bin]
    if (image.shape[0] < 2048 - extra_row_size):
        image = image[0:image.shape[0] - extra_row_size, :]  # cut off 8 pixels from top of y
    if np.amax(image) > 0.95*max_pixel_val:
        # overexposed
        print("Image is likely overexposed, highest pixel val",np.amax(image))
    return image

def auto_expose(cam:Camera, target_intensity=0.9, tolerance=0.05, max_iter=5):

    target = max_pixel_val * target_intensity # highest individual pixel brightness we want to allow
    original_exposure = cam.exposure_time

    for i in range(max_iter):
        cam.stop()
        cam.record(mode="sequence")
        image, meta = cam.image()
        peak = np.max(image[0:image.shape[0]-extra_row_size,:])
        og_exposure = cam.exposure_time

        # Calculate ratio and adjust exposure
        if peak == 0:  # Prevent division by zero
            new_exposure = og_exposure * 10 # aggressive, increases by exposure by factor of 10 if too dark
        else:
            ratio = target / peak
            if abs(1 - ratio) < tolerance:
                break
            # new_exposure = cam.configuration['exposure time'] * ratio
            new_exposure = og_exposure * ratio
        cam.exposure_time = float(min(new_exposure,0.499)) # camera allows 500ms max, but set 499 due to floating point error

        # print(f"Adjusting exposure from {og_exposure:.3f} to {cam.exposure_time:.3f}s (Peak was: {peak:.3g}, now will be ~{peak * cam.exposure_time / og_exposure:.3g})")
    print(f"Autoexpose changing exposure from {original_exposure:.3f} to {cam.exposure_time:.3f}s")

    return

def set_cam_settings(cam:Camera, exposure_time, roi=None, binning=(1, 1)):
    global master_dark_frame
    global master_roi
    global master_bin
    if os.path.exists(dark_frame_path):
        master_dark_frame = np.load(dark_frame_path)
        binning_amount = binning[0]
        master_bin = binning_amount
        if roi is None:
            master_roi = (1,1,camera_resolution//binning_amount,camera_resolution//binning_amount)
        else:
            master_roi = ((roi[0]-1)*binning_amount + 1,(roi[1]-1)*binning_amount + 1,roi[2]*binning_amount,roi[3]*binning_amount)
        print("Loaded master dark background successfully.")
    else:
        master_dark_frame = None
        master_roi = None
        master_bin = None
        print("Warning: No master dark frame found. Subtraction disabled.")

    cam.delay_time = 0
    if roi is not None:
        cam.configuration = {
            'exposure time': exposure_time,  # in seconds
            'trigger': 'auto sequence',
            'binning': binning,
            'roi': roi # Region of Interest (x0, y0, x1, y1)
        }
    else:
        cam.configuration = {
            'exposure time': exposure_time,  # in seconds
            'trigger': 'auto sequence'
        }

def read_image(cam:Camera,n_windows):
    Log.log("setting camera record")
    cam.record(mode="sequence", number_of_images=n_windows)
    # ^ line takes a long time, im guessing its a blocking wait
    # Even though exposure time is 100ms, it takes 210-230ms, whereas cam.image_average() returns immediately
    # This time is constant regardless of image size, so it's likely not a data transfer issue
    # With a 10ms exposure time, it takes 120-130ms, so it seems there's a ~110ms wait for some reason
    # With a 500ms exposure time, it takes ~610-620ms, supporting the above constant wait
    # TODO: figure out why tf this constant delay exists
    # maybe solution is to use a different recording mode, which isn't blocking, doesn't require resetting after each picture
    # it'll be something like a sequence of images for the whole freq sweep, but that I time properly in time with frequency changes
    # Log.log("getting average image value")

    image = cam.image_average()
    if master_dark_frame is not None:
        image = image - cam.exposure_time * master_dark_frame[
            (master_roi[1] - 1) // master_bin:master_roi[3] // master_bin, (master_roi[0] - 1) // master_bin:
                                                                           master_roi[2] // master_bin]
    if (image.shape[0] < 2048 - extra_row_size):
        image = image[0:image.shape[0] - extra_row_size, :]  # cut off 8 pixels from top of y
    if np.amax(image) > 0.95*max_pixel_val:
        # overexposed
        print("Image is likely overexposed, highest pixel val",np.amax(image))
    return image


def plot_image(image, x_points=None, y_points=None, title = ""):
    if x_points is None:
        x_points = np.arange(image.shape[1])
    if y_points is None:
        y_points = np.arange(image.shape[0])
    mesh = plt.pcolormesh(x_points, y_points, image, shading='nearest', cmap='gray', vmin=0) # use bone, gray, inferno, nihfire, viridis
    plt.colorbar(mesh, label='637nm brightness (arb units)')
    plt.gca().invert_yaxis()
    plt.xlabel('x space (pixels)')
    plt.ylabel('y space (pixels)')
    if title != "":
        plt.title(title)
    else:
        plt.title('camera image')
    plt.show()

def bin_image(image):
    return np.sum(image)
def select_one_pixel(image,x,y):
    return image[x,y]


def connect_cam_RF(roi: tuple[int, int, int, int] | None,binning_amount, forced_exposure = None) -> tuple[Camera, float]:
    # connect to RF src
    sg = cs.connect_sg386(cs.sg_resource)
    # connect to cam
    cam = connect_cam(roi, binning_amount, forced_exposure = forced_exposure)
    # if forced_exposure is not None:
    #     cam.exposure_time = forced_exposure
    return cam, sg


def connect_cam(roi: tuple[int, int, int, int] | None,binning_amount=1, forced_exposure = None) -> Camera:

    # connect to cam
    cam = pco.Camera()
    set_cam_settings(cam, 10e-3/binning_amount**2, roi=roi, binning=(binning_amount, binning_amount))
    if forced_exposure is not None:
        # if forced_exposure > cam.exposure_time:
        #     print(f"Manually forcing exposure time of {forced_exposure:.3f}s. Warning, higher than autoexposed value, may cause overexposure")
        print(f"Manually forcing exposure time of {forced_exposure:.3f}s")
        cam.exposure_time = forced_exposure
    else:
        # Only auto-expose if a set value isn't given
        auto_expose(cam, target_intensity=0.3)  # sets cameras exposure time
    img = read_image(cam,1)
    plot_image(img)
    print("Example image plotted")
    return cam



def get_spacial_params(binning_amount, pos_data) -> tuple[tuple[int, int, int, int], float, float]:
    focus_point_size, focus_point_centre_x, focus_point_centre_y = pos_data
    if focus_point_size/binning_amount < 32:
        raise camera_exception.CameraException("Focus point size too small must be >=32 after binning")
    if focus_point_centre_x <8 | focus_point_centre_y <8 | focus_point_centre_x > 2040 | focus_point_centre_y > 2040:
        raise camera_exception.CameraException("Focus centre outside camera frame, must be 8<=x,y <=2040")
    fps = focus_point_size // 16 * 16 // binning_amount
    fpcx = focus_point_centre_x // 16 * 16 // binning_amount
    fpcy = focus_point_centre_y // 16 * 16 // binning_amount
    x_points = np.arange(fpcx - fps // 2 + 1, fpcx + fps // 2 + 1)
    y_points = np.arange(fpcy - fps // 2 + 1, fpcy + fps // 2 + 1)
    # Approximately 130nm per physical pixel (6.5um pixel width, and 50x objective)
    x_space = x_points * 6.5/objective_magnification / 10 ** 3 * binning_amount # position in mm
    y_space = y_points * 6.5/objective_magnification / 10 ** 3 * binning_amount
    # region of interest, crop into this portion of the camera's view

    # add a row of 8 pixels to y if space allows, to remove the bright row
    if (fpcy + fps // 2 <= 2048-extra_row_size):
        roi = (fpcx - fps // 2 + 1, fpcy - fps // 2 + 1,
               fpcx + fps // 2, fpcy + fps // 2 + extra_row_size)
    else:
        roi = (fpcx - fps // 2 + 1, fpcy - fps // 2 + 1,
               fpcx + fps // 2, fpcy + fps // 2)
    return roi, x_space, y_space

def run_odmr_measurement(cam_rf_params, amp_dbm, fn, odmr_params):
    # Safely runs measurement with cam and sg, turning everything off correctly at the end

    cam, sg = connect_cam_RF(*cam_rf_params)
    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(0.1)

    try:
        ret_vals = fn(cam, sg, *odmr_params)
    finally:
        cs.enable_sg386(sg, enable=False)
        cam.close()
    return ret_vals



def bin_counts(counts_2D, binning_num, x_space, y_space):
    # counts_2D has shape (x_width, y_with, freqs_len)
    # x,y width are (generally) equal and a multiple of binning_amount
    if len(x_space) == 0 or len(y_space) == 0:
        raise camera_exception.CameraException(
            f"Cannot bin empty spatial coordinates. x_space len: {len(x_space)}, y_space len: {len(y_space)}"
        )

    if binning_num < 1:
        raise camera_exception.CameraException(f"Binning number must be >= 1, was {binning_num}")
    elif binning_num == 1:
        return counts_2D, x_space, y_space
    elif binning_num > len(x_space):
        raise camera_exception.CameraException(f"Binning number must be smaller than dimension width, was {binning_num} with image dimensions {len(x_space)}x{len(y_space)}")
    elif len(x_space) % binning_num != 0:
        raise camera_exception.CameraException(f"Binning number must be an even multiple of dimension width, was {binning_num} with image dimensions {len(x_space)}x{len(y_space)}")

    # counts_reshaped is of the shape: counts_x/bin, bin, counts_y/bin, bin, freq
    counts_reshaped = np.reshape(counts_2D, (counts_2D.shape[0] // binning_num, binning_num, counts_2D.shape[1] // binning_num, binning_num, counts_2D.shape[2]))
    # print("Reshaping image worked")
    # counts_binned is of the shape: counts_x/bin, counts_y/bin, freq, meaning we have to sum over the two bin axes
    counts_binned = counts_reshaped.sum(axis=1).sum(axis=2) # sum over the
    # print("Summing image worked, returning")

    return counts_binned, x_space[0::binning_num], y_space[0::binning_num]



def sweep_freqs_binned(cam, sg, dwell, freqs, n_windows, n_iter, iteration) -> npt.NDArray[np.float64]:
    point_duration_s = cam.exposure_time * n_windows
    brightness = np.zeros((freqs.size))
    _ = read_image(cam, n_windows) # ignore first image
    for j, f in enumerate(freqs):
        # Log.log("at freq " + str(np.round(f/10**9,2)) + "GHz, updating progress")
        cs.print_odmr_progress(iteration * len(freqs) + j, len(freqs) * n_iter, iteration, f)
        # Log.log("setting new freq on signal gen")
        sg.write(f"FREQ {float(f)}")
        # Log.log("sleeping " + str(dwell) + "s")
        time.sleep(dwell)
        Log.log("reading image")
        image = read_image(cam, n_windows)
        Log.log("binning image")
        all_counts = bin_image(image)
        # pci.plot_image(image)
        # Log.log("saving binned value")
        brightness[j]=(all_counts / point_duration_s)
    return brightness


def sweep_freqs(cam, sg, dwell, freqs, n_windows, n_iter, iteration) -> npt.NDArray[np.float64]:
    point_duration_s = cam.exposure_time * n_windows
    image = read_image(cam,1) # Throw out first image, it's often too bright
    brightnesses = np.zeros((image.shape[0], image.shape[1], freqs.size)) # should be n_iter*2 when reversing as well
    for j, f in enumerate(freqs):
        # Log.log("at freq " + str(np.round(f/10**9,2)) + "GHz, updating progress")
        cs.print_odmr_progress(iteration * len(freqs) + j, len(freqs) * n_iter, iteration, f)
        # Log.log("setting new freq on signal gen")
        sg.write(f"FREQ {float(f)}")
        # Log.log("sleeping " + str(dwell) + "s")
        time.sleep(dwell)
        Log.log("reading image")
        image = read_image(cam, n_windows)
        Log.log("saving image in array")
        # pci.plot_image(image)
        brightnesses[:,:,j] =(image / point_duration_s)
    return brightnesses


# def sweep_freqs_binned_fifo(cam, sg, dwell, freqs, n_windows, n_iter, iteration) -> npt.NDArray[np.float64]:
#     point_duration_s = cam.exposure_time * n_windows
#     brightness = np.zeros((freqs.size))
#     cam.record(mode="fifo", number_of_images=n_windows*10)
#     for j, f in enumerate(freqs):
#         Log.log("at freq " + str(np.round(f/10**9,2)) + "GHz, updating progress")
#         # cs.print_odmr_progress(iteration * len(freqs) + j, len(freqs) * n_iter, iteration, f)
#
#         # Log.log("setting new freq on signal gen")
#         sg.write(f"FREQ {float(f)}")
#         # Log.log("sleeping " + str(dwell) + "s")
#         time.sleep(dwell)
#         all_counts=0
#         for i in range(n_windows):
#             Log.log("sending software trigger")
#             cam.send_software_trigger() # this doesn't exist apparently, even tho this website seems convinced it does: https://pylablib.readthedocs.io/en/latest/.apidoc/pylablib.devices.PCO.html#pylablib.devices.PCO.SC2.PCOSC2Camera.send_software_trigger
#             Log.log("reading image")
#             image, dict = cam.image(image_index=0) # if fifo, then start index always 0
#             if (image.shape[0] < 2048-extra_row_size):
#                 image = image[0:image.shape[0]-extra_row_size,:] # cut off 8 pixels from top of y
#             if np.amax(image) > 0.95*max_pixel_val:
#                 # overexposed
#                 print("Image is likely overexposed, highest pixel val",np.amax(image))
#             Log.log("binning image")
#             all_counts += bin_image(image)
#         # pci.plot_image(image)
#         # Log.log("saving binned value")
#         brightness[j]=(all_counts / point_duration_s)
#     cam.stop()
#     return brightness


# Another method would be to use ring buffer constantly recording, and simply collect images
# after having set the frequency and waiting for a new image, so no trigger is required
# TODO: check how long it takes for the cam.image() method to return, this should only be data transfer time (no blocking)

def sweep_freqs_binned_ringBuf(cam, sg, dwell, freqs, n_windows, n_iter, iteration) -> npt.NDArray[np.float64]:
    point_duration_s = cam.exposure_time * n_windows
    brightness = np.zeros((freqs.size))
    cam.record(mode="ring buffer", number_of_images=n_windows*10)
    cam.wait_for_new_image()
    image, dict = cam.image(image_index=-1) # Ignore first image
    cam.wait_for_new_image()
    image, dict = cam.image(image_index=-1) # Ignore second image
    for j, f in enumerate(freqs):
        Log.log("at freq " + str(np.round(f/10**9,2)) + "GHz, updating progress")
        cs.print_odmr_progress(iteration * len(freqs) + j, len(freqs) * n_iter, iteration, f)

        # Log.log("setting new freq on signal gen")
        # supposedly this value will wrap around so I shouldn't get an index-out-of-bounds error when using it to get img
        sg.write(f"FREQ {float(f)}")
        cam.wait_for_new_image() # wait for new image, to be sure recorded ones are it's only for this frequency
        # If i don't wait, i'm blurring the lines between freqs, adding a bit of x uncertainty, maybe its worth it for the speed
        curr_img_num = cam.recorded_image_count
        # Log.log("sleeping " + str(dwell) + "s")
        time.sleep(dwell)
        all_counts=0
        for i in range(n_windows):
            if cam.recorded_image_count == curr_img_num:
                Log.log("waiting for new img")
                cam.wait_for_new_image()
                Log.log("finished waiting for new img")
            curr_img_num = cam.recorded_image_count
            # there's a smarter way to do this, where if theres multiple new images waiting I request them all at once
            # problem with the above is that i'd have to be very careful not to accidentally read old images as new ones
            # The above is rare to happen though, cuz the couple lines of code execute quite quickly compared to exposure time (unless its ~1ms)
            Log.log("reading image")
            # image, dict = cam.image(image_index=-1) # changed to always get the newest image
            # if master_dark_frame is not None:
            #     image = image - cam.exposure_time * master_dark_frame[(master_roi[1]-1)//master_bin:master_roi[3]//master_bin,(master_roi[0]-1)//master_bin:master_roi[2]//master_bin]
            #
            # # curr_img_num+=1
            # if (image.shape[0] < 2048-extra_row_size):
            #     image = image[0:image.shape[0]-extra_row_size,:] # cut off 8 pixels from top of y
            # if np.amax(image) > 0.95*max_pixel_val:
            #     # overexposed
            #     print("Image is likely overexposed, highest pixel val",np.amax(image))
            image = get_image_sub_bkg(cam)
            Log.log("binning image")
            all_counts += bin_image(image)
        # pci.plot_image(image)
        # Log.log("saving binned value")
        brightness[j]=(all_counts / point_duration_s)
    cam.stop()
    return brightness


def sweep_freqs_binned_recorded(cam, sg, dwell, freqs, n_windows, n_iter, iteration) -> npt.NDArray[np.float64]:
    point_duration_s = cam.exposure_time * n_windows
    cam.wait_for_new_image()
    image, dict = cam.image(image_index=-1)
    brightness = np.zeros((freqs.size))
    for j, f in enumerate(freqs):
        Log.log("at freq " + str(np.round(f/10**9,2)) + "GHz, updating progress")
        cs.print_odmr_progress(iteration * len(freqs) + j, len(freqs) * n_iter, iteration, f)

        # Log.log("setting new freq on signal gen")
        # supposedly this value will wrap around so I shouldn't get an index-out-of-bounds error when using it to get img
        sg.write(f"FREQ {float(f)}")
        cam.wait_for_new_image() # wait for new image, to be sure recorded ones are it's only for this frequency
        # If i don't wait, i'm blurring the lines between freqs, adding a bit of x uncertainty, maybe its worth it for the speed
        curr_img_num = cam.recorded_image_count
        # Log.log("sleeping " + str(dwell) + "s")
        time.sleep(dwell)
        all_counts=0
        for i in range(n_windows):
            if cam.recorded_image_count == curr_img_num:
                Log.log("waiting for new img")
                cam.wait_for_new_image()
                Log.log("finished waiting for new img")
            curr_img_num = cam.recorded_image_count
            # there's a smarter way to do this, where if theres multiple new images waiting I request them all at once
            # problem with the above is that i'd have to be very careful not to accidentally read old images as new ones
            # The above is rare to happen though, cuz the couple lines of code execute quite quickly compared to exposure time (unless its ~1ms)
            Log.log("reading image")
            image = get_image_sub_bkg(cam)
            # image, dict = cam.image(image_index=-1) # changed to always get the newest image
            # if master_dark_frame is not None:
            #     image = image - cam.exposure_time * master_dark_frame[
            #         (master_roi[1] - 1) // master_bin:master_roi[3] // master_bin, (master_roi[0] - 1) // master_bin:
            #                                                                        master_roi[2] // master_bin]
            # # curr_img_num+=1
            # if (image.shape[0] < 2048-extra_row_size):
            #     image = image[0:image.shape[0]-extra_row_size,:] # cut off 8 pixels from top of y
            # if np.amax(image) > 0.95*max_pixel_val:
            #     # overexposed
            #     print("Image is likely overexposed, highest pixel val",np.amax(image))
            Log.log("binning image")
            all_counts += bin_image(image)
        # pci.plot_image(image)
        # Log.log("saving binned value")
        brightness[j]=(all_counts / point_duration_s)
    return brightness

# TODO: upgrade the freq sweep so that I don't have to relaunch the recording each iteration,
#  will save 10s of ms/iter so not super urgent


def measure_binned_odmr(cam, sg, freqs, dwell, n_windows, n_iter: int = 1) -> np.ndarray:
    point_duration_s = cam.exposure_time * n_windows
    cam.record(mode="ring buffer", number_of_images=n_windows * 10)
    cam.wait_for_new_image()
    print(f"measuring binned ODMR with {n_iter} iterations, estimate time to completion"
          f" ~{n_iter*2 * ((len(freqs) + 1) * (dwell + point_duration_s) + 0.02):.0f}s")
    t0 = time.time()
    brightnesses = np.zeros((n_iter*2, freqs.size)) # should be n_iter*2 when reversing as well
    for i in range(n_iter):
        Log.log("Sweeping freqs:")
        brightnesses[i] = sweep_freqs_binned_recorded(cam, sg, dwell, freqs, n_windows, n_iter * 2, i * 2)
        brightnesses[n_iter + i] = sweep_freqs_binned_recorded(cam, sg, dwell, freqs[::-1], n_windows, n_iter * 2, i * 2 + 1)[::-1]
        Log.log("after sweeping freqs")

        # TODO: have it save partial measurements after each iteration, like widefield does
    cam.stop()
    sys.stdout.write(f"\r\033[KODMR finished, took {time.time()-t0:.0f}s\n") # Clear progress bar
    sys.stdout.flush()
    return np.sum(brightnesses,axis=0)/(n_iter*2)


def save_master_dark(cam, save_path, n_averages, num_windows):
    """
    Captures several dark frames at full sensor resolution, averages them
    to eliminate random temporal readout noise, and saves the master frame.
    """
    dark_buffer = []

    print(f"Capturing {n_averages} dark frames. Ensure lens cap is secured...")
    for _ in range(n_averages):
        # Replace with your actual camera frame acquisition call
        cam.record(mode="sequence", number_of_images=num_windows)
        image = cam.image_average()
        dark_buffer.append(image)

    # Average along the third axis to create a clean, low-noise master dark
    master_dark = np.mean(dark_buffer, axis=0).astype(np.float64) * 10 # multiply by 10 so this is background per second (using 100ms exposure)

    np.save(save_path, master_dark)
    print(f"Master dark frame saved successfully to {save_path}")

def main():
    binning_amount = 1
    roi = (1,1,camera_resolution//binning_amount,camera_resolution//binning_amount)
    global master_dark_frame
    master_dark_frame = None
    cam = connect_cam(roi, binning_amount, forced_exposure = 0.1)
    save_master_dark(cam, dark_frame_path, 50, 10)

if __name__ == "__main__":
    main()