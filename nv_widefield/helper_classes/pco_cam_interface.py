
import numpy as np
import matplotlib.pyplot as plt
import pco
from pco import Camera
import connection_setup as cs

# Constants
camera_resolution = 2048 # pixels, range from 1 tpo 2048 inclusive
extra_row_size = 8
max_pixel_val = 65535 # 2^16-1

def auto_expose(cam, target_intensity=0.9, tolerance=0.05, max_iter=5):

    target = max_pixel_val * target_intensity # highest individual pixel brightness we want to allow

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

        # config = cam.configuration
        # config['exposure time'] = new_exposure
        # cam.configuration = config
        # print(f"want new exposure time: {new_exposure:.5f}")
        cam.exposure_time = float(min(new_exposure,0.499))

        print(f"Adjusting exposure from {og_exposure:.3f} to {cam.exposure_time:.3f}s (Peak was: {peak:.3g}, now will be ~{peak * cam.exposure_time / og_exposure:.3g})")


    return cam.exposure_time

def set_cam_settings(cam, exposure_time, roi=(1, 1, camera_resolution, camera_resolution), binning=(1, 1)):
    cam.configuration = {
        'exposure time': exposure_time,  # in seconds
        # 'pixel rate': 100_000_000, # lowk no idea what this means
        'trigger': 'auto sequence',
        'binning': binning,
        'roi': roi # Region of Interest (x0, y0, x1, y1)
    }

def read_image(cam,n_windows):
    # Re-arm specifically for this new capture
    # cam.stop()
    cam.record(mode="sequence", number_of_images=n_windows)
    image = cam.image_average()
    if (image.shape[0] < 2048-extra_row_size):
        image = image[0:image.shape[0]-extra_row_size,:] # cut off 8 pixels from top of y
    # cam.stop() # should be called automaticcaly when n_windows reached
    if np.amax(image) > 0.95*max_pixel_val:
        # overexposed
        print("Image is likely overexposed, highest pixel val",np.amax(image))
    return image

def setup_cam():
    cam = pco.Camera()
    # cam.sdk.set_metadata_mode(0)  # 0 = off, 1 = on/embedded
    set_cam_settings(cam, 10e-3)
    return cam

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


def connect_cam_RF(roi: tuple[int, int, int, int],binning_amount) -> tuple[Camera, float, float]:
    # connect to RF src
    sg = cs.connect_sg386(cs.sg_resource)
    # connect to cam
    cam, exposure_time = connect_cam(roi, binning_amount)
    return cam, exposure_time, sg


def connect_cam(roi: tuple[int, int, int, int],binning_amount) -> tuple[Camera, float]:
    # connect to cam
    cam = setup_cam()
    set_cam_settings(cam, 10e-3/binning_amount**2, roi=roi, binning=(binning_amount, binning_amount))
    exposure_time = auto_expose(cam, target_intensity=0.3)  # returns exposure time in s
    # exposure_time = 0.0025 # roughly match SPCM exposure time
    # print("Exposure time: ", exposure_time)
    # set_cam_settings(cam, exposure_time, roi=roi, binning=(binning_amount, binning_amount))
    img = read_image(cam,1)
    plot_image(img) # if roi fills, then plot full image
    print("Example image plotted")
    return cam, exposure_time



def get_spacial_params(binning_amount, pos_data) -> tuple[tuple[int, int, int, int], float, float]:
    focus_point_size, focus_point_centre_x, focus_point_centre_y = pos_data
    fps = focus_point_size // 8 * 8 // binning_amount  # forces fps to be a multiple of 8, before dividing by the binning
    fpcx = focus_point_centre_x // 8 * 8 // binning_amount
    fpcy = focus_point_centre_y // 8 * 8 // binning_amount
    x_points = np.arange(fpcx - fps // 2 + 1, fpcx + fps // 2 + 1)
    y_points = np.arange(fpcy - fps // 2 + 1, fpcy + fps // 2 + 1)
    x_space = x_points * 50 / 10 ** 6 * binning_amount
    y_space = y_points * 50 / 10 ** 6 * binning_amount
    # region of interest, crop into this portion of the camera's view

    # note: I'm adding a row of 8 pixels to the y
    if (fpcy + fps // 2 <= 2048-extra_row_size):
        roi = (fpcx - fps // 2 + 1, fpcy - fps // 2 + 1,
               fpcx + fps // 2, fpcy + fps // 2 + extra_row_size)
    else:
        roi = (fpcx - fps // 2 + 1, fpcy - fps // 2 + 1,
               fpcx + fps // 2, fpcy + fps // 2)
    return roi, x_space, y_space