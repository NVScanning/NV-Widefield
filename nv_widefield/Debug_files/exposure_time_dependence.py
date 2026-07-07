import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))
import connection_setup as cs
import helper_classes.pco_cam_interface as pci


def main():
    binning_amount = 4
    focus_point_size = 300
    focus_point_centre_x, focus_point_centre_y = 880, 1070
    num_windows = 50

    exposure_range = np.array([1,2,3,4,5,10,15,20,25,50,75,100,150,200,300,400,500])/1000


    roi, _, _ = pci.get_spacial_params(
        binning_amount,
        (focus_point_size, focus_point_centre_x, focus_point_centre_y)
    )
    roi = (1,1,pci.camera_resolution//binning_amount,pci.camera_resolution//binning_amount)

    cam = pci.connect_cam(roi, binning_amount, 0.001)

    total_brightness_results = []

    t0 = time.time()
    print(f"Beginning exposure linearity sweep, estimate time to completion {num_windows*(sum(exposure_range)) + len(exposure_range)*0.1:.1f} s")

    try:
        for idx, exp_time in enumerate(exposure_range):
            # Update camera exposure register dynamically
            pci.set_cam_settings(cam, float(exp_time))
            # cam.exposure_time = float(exp_time)
            time.sleep(0.05)  # Let the camera register settle

            # Record an image frame and grab total array brightness
            # Uses standard sdk or underlying frame buffer array depending on your interface wrappers
            # cam.record(number_of_images=num_windows, mode='sequence')
            image = pci.read_image(cam, num_windows)
            # Sum up the total pixel values in your target ROI region
            total_intensity = np.sum(image) /exp_time
            total_brightness_results.append(total_intensity)

    finally:
        # Close connection handle cleanly
        cam.close()

    # Clear progress bar line
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()
    print(f"Sweep complete in {time.time() - t0:.1f}s.")

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(exposure_range * 1000, total_brightness_results, 'o', color='tab:green', markersize=5)

    # Fit a linear line to verify sensor saturation limits or linearity behaviors
    # slope, intercept = np.polyfit(exposure_range * 1000, total_brightness_results, 1)
    # plt.plot(
    #     exposure_range * 1000,
    #     slope * (exposure_range * 1000) + intercept,
    #     '--',
    #     color='gray',
    #     label=f'Linear Fit (R² verification)'
    # )

    plt.xlabel("Exposure Time [ms]", fontsize=12)
    plt.ylabel("Total Brightness (Integrated Counts)/s", fontsize=12)
    plt.title("Camera Sensor Linearity Check: Integrated Intensity vs Exposure", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    # plt.legend(loc="upper left")
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks([1,5,10,25,50,100,200,300,400,500])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()