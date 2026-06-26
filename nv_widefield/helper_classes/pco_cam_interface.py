import time
import numpy as np
import matplotlib.pyplot as plt
import pco
from pco import Camera, camera_exception
import connection_setup as cs

# Constants
camera_resolution = 2048 # pixels, range from 1 tpo 2048 inclusive
extra_row_size = 8
max_pixel_val = 65535 # 2^16-1
min_roi_dims = 32
objective_magnification = 50

background_rate = 0 # 5600 000 / 128^2  # total brightness divided by num physical pixels divided by exposure time
# ^ This num ends up being like 341.8 counts/s on each pixel

def auto_expose(cam, target_intensity=0.9, tolerance=0.05, max_iter=5):

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

def set_cam_settings(cam, exposure_time, roi=None, binning=(1, 1)):
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
            'trigger': 'auto sequence',
        }

def read_image(cam,n_windows):
    cam.record(mode="sequence", number_of_images=n_windows)
    image = cam.image_average()
    if (image.shape[0] < 2048-extra_row_size):
        image = image[0:image.shape[0]-extra_row_size,:] # cut off 8 pixels from top of y
    if np.amax(image) > 0.95*max_pixel_val:
        # overexposed
        print("Image is likely overexposed, highest pixel val",np.amax(image))
    return np.maximum(image - cam.exposure_time * background_rate, [0])


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



def sweep_freqs_binned(cam, sg, dwell, freqs, n_windows, n_iter, iteration) -> tuple[list[int]]:
    point_duration_s = cam.exposure_time * n_windows
    brightness = []
    for j, f in enumerate(freqs):
        cs.print_odmr_progress(iteration * len(freqs) + j, len(freqs) * n_iter, iteration, f)
        # if (seen % printout_factor == 0):
        #     print(f"at iteration {i} and freq {(f / 10 ** 9):.2f}GHz; {seen/(printout_factor*num_printouts)*100:.0f}% done")
        sg.write(f"FREQ {float(f)}")
        time.sleep(dwell)
        image = read_image(cam, n_windows)
        all_counts = bin_image(image)
        # pci.plot_image(image)
        brightness.append(all_counts / point_duration_s)
    return brightness


def sweep_freqs(cam, sg, dwell, freqs, n_windows, n_iter, iteration) -> tuple[list[int]]:
    point_duration_s = cam.exposure_time * n_windows
    image = read_image(cam,1) # Throw out first image, it's often too bright
    brightnesses = np.zeros((image.shape[0], image.shape[1], freqs.size)) # should be n_iter*2 when reversing as well
    for j, f in enumerate(freqs):
        cs.print_odmr_progress(iteration * len(freqs) + j, len(freqs) * n_iter, iteration, f)
        # if (seen % printout_factor == 0):
        #     print(f"at iteration {i} and freq {(f / 10 ** 9):.2f}GHz; {seen/(printout_factor*num_printouts)*100:.0f}% done")
        sg.write(f"FREQ {float(f)}")
        time.sleep(dwell)
        image = read_image(cam, n_windows)
        # pci.plot_image(image)
        brightnesses[:,:,j] =(image / point_duration_s)
    return brightnesses

