import os
import sys
import time
import datetime
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from numpy import dtype, ndarray

sys.path.append(os.path.abspath(".."))
import connection_setup as cs
import helper_classes.pco_cam_interface as pci
import helper_classes.odmr_plotting as oPlot
import nv_widefield.helper_classes.file_interface as iFile
import widefield_odmr as wODMR
import nv_setup.cw_odmr.Lorentzian_fit as Lfit


"""
bin together pixels in a square region, and compare multiple regions from a widefield measurement

Useful to check that B is varying at all and improve SNR without forcing a low resolution measurement 
from significant pixel binning on the whole dataset
"""

NEW_MEASUREMENT = False  # Set to True for hardware run, False to read old widefield measurement
roi_width = 8

date = "2026-10-19" # YYYY-MM-DD
time = "10-28-46"   # HH-MM-SS

data_dir = "C:\\Users\\NVCFM\\Desktop\\NVCFM_Data"
date_directory = os.path.join(data_dir, date)
old_Measurement_Path = os.path.join(date_directory, "widefield_cw_odmr_"+time+".npz")

def extract_region_odmrs(counts_3d, x_space, y_space, regions):
    """
    Extracts averaged 1D ODMR spectra for a list of target regions.

    Parameters
    ----------
    counts_3d : np.ndarray
        Array of shape (y_dim, x_dim, n_freqs) returned by camera acquisition.
    x_space : np.ndarray
        1D array of absolute X pixel coordinates for the ROI.
    y_space : np.ndarray
        1D array of absolute Y pixel coordinates for the ROI.
    regions : list of dict
        List of target squares: [{'center': (x, y), 'width': w}, ...] in absolute ROI pixel coordinates.

    Returns
    -------
    region_odmrs : list of np.ndarray
        1D ODMR intensity spectra corresponding to each specified region.
    region_metadata : list of dict
        Bounding box indices and pixel space coordinates for each region.
    """
    region_odmrs = []

    for idx, reg in enumerate(regions):
        cx, cy = reg["center"]
        w = reg["width"]

        y_min, y_max = cy-w//2,cy+w//2
        x_min, x_max = cx-w//2,cx+w//2

        counts_roi = counts_3d[y_min:y_max, x_min:x_max, :]

        region_counts = np.mean(counts_roi, axis=(0, 1))
        region_odmrs.append(region_counts)

    return region_odmrs


def region_to_metadata(x_space, y_space, regions):
    """
    Extracts averaged 1D ODMR spectra for a list of target regions.

    Parameters
    ----------
    counts_3d : np.ndarray
        Array of shape (y_dim, x_dim, n_freqs) returned by camera acquisition.
    x_space : np.ndarray
        1D array of absolute X pixel coordinates for the ROI.
    y_space : np.ndarray
        1D array of absolute Y pixel coordinates for the ROI.
    regions : list of dict
        List of target squares: [{'center': (x, y), 'width': w}, ...] in absolute ROI pixel coordinates.

    Returns
    -------
    region_odmrs : list of np.ndarray
        1D ODMR intensity spectra corresponding to each specified region.
    region_metadata : list of dict
        Bounding box indices and pixel space coordinates for each region.
    """
    region_metadata = []

    for idx, reg in enumerate(regions):
        cx, cy = reg["center"]
        w = reg["width"]
        half_w = w / 2.0

        # x_indices = np.where((x_space >= cx - half_w) & (x_space < cx + half_w))[0]
        # y_indices = np.where((y_space >= cy - half_w) & (y_space < cy + half_w))[0]
        #
        # if x_indices.size == 0 or y_indices.size == 0:
        #     raise ValueError(f"Region {idx} at center ({cx}, {cy}) with width {w} falls outside ROI spatial bounds.")
        y_min, y_max = cy-w//2,cy+w//2
        x_min, x_max = cx-w//2,cx+w//2

        region_metadata.append({
            "region_idx": idx,
            "center": (cx, cy),
            "width": w,
            "y_bounds": (y_min, y_max),
            "x_bounds": (x_min, x_max),
            "x_coords": x_space[x_min:x_max],
            "y_coords": y_space[y_min:y_max]
        })

    return region_metadata



def plot_regions(mean_image, x_space, y_space, region_metadata):
    """
    Plots the spatial intensity image with colored bounding boxes and
    the corresponding overlaid ODMR spectra using matching colors.
    """
    x_space, y_space = x_space*1000, y_space*1000 # convert to um
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, (ax_img) = plt.subplots(1, 1, figsize=(7.5, 7))

    # Determine spatial pixel spacing to convert width (in px) to spatial units
    dx = np.abs(x_space[1] - x_space[0])
    dy = np.abs(y_space[1] - y_space[0])

    # --- Plot Spatial Image with Bounding Boxes ---
    # Use origin="lower" to keep standard increasing Cartesian axes for easy patch rendering
    generate_img_rois(ax_img, colors, dx, dy, fig, mean_image, region_metadata, x_space, y_space)


    plt.tight_layout()
    plt.show()

def plot_regions_and_odmrs(counts_3d, freqs, x_space, y_space, region_odmrs, region_metadata, normalize=True):
    """
    Plots the spatial intensity image with colored bounding boxes and
    the corresponding overlaid ODMR spectra using matching colors.
    """
    x_space, y_space = x_space*1000, y_space*1000 # convert to um
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mean_image = np.mean(counts_3d, axis=-1) # average along the freq axis

    fig, (ax_img, ax_odmr) = plt.subplots(1, 2, figsize=(15, 7))

    # Determine spatial pixel spacing to convert width (in px) to spatial units
    dx = np.abs(x_space[1] - x_space[0])
    dy = np.abs(y_space[1] - y_space[0])

    # --- Plot Spatial Image with Bounding Boxes ---
    # Use origin="lower" to keep standard increasing Cartesian axes for easy patch rendering
    generate_img_rois(ax_img, colors, dx, dy, fig, mean_image, region_metadata, x_space, y_space)

    # --- Plot ODMR Spectra with Matching Colors ---
    freqs_ghz = freqs / 1e9
    for idx, (odmr, meta) in enumerate(zip(region_odmrs, region_metadata)):
        color = colors[idx % len(colors)]
        trace = (odmr-np.min(odmr)) / (np.max(odmr)-np.min(odmr)) if normalize else (odmr - np.min(odmr))

        # label_str = f"Region {idx} | Center: {meta['center']}, Width: {meta['width']} px"
        label_str = f"Region {idx} | Center: {meta['center']}"
        ax_odmr.plot(freqs_ghz, trace, label=label_str, color=color, linewidth=1.5)

    ax_odmr.set_xlabel("Frequency (GHz)")
    ax_odmr.set_ylabel("Normalized Intensity" if normalize else "PL Intensity difference (ADC Counts)")
    ax_odmr.set_title("Overlaid Region-Averaged ODMR Spectra")
    ax_odmr.grid(True, linestyle="--", alpha=0.6)
    ax_odmr.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.show()


def generate_img_rois(ax_img, colors, dx: ndarray[tuple[Any, ...], dtype[Any]] | Any,
                      dy: ndarray[tuple[Any, ...], dtype[Any]] | Any, fig: Figure, mean_image, region_metadata,
                      x_space: Any, y_space: Any):
    im = ax_img.imshow(
        mean_image,
        extent=[x_space[0], x_space[-1], y_space[0], y_space[-1]],
        cmap="gray",
        origin="lower",
    )
    fig.colorbar(im, ax=ax_img, label="Mean PL Intensity (ADC counts)")

    for idx, meta in enumerate(region_metadata):
        color = colors[idx % len(colors)]
        cx_idx, cy_idx = meta["center"]
        w_px = meta["width"]

        # Map pixel index center to spatial coordinates
        cx_spatial = x_space[cx_idx]
        cy_spatial = y_space[cy_idx]

        w_spatial_x = w_px * dx
        w_spatial_y = w_px * dy

        # Bottom-left corner in spatial units
        bottom_left_x = cx_spatial - (w_spatial_x / 2.0)
        bottom_left_y = cy_spatial - (w_spatial_y / 2.0)

        rect = patches.Rectangle(
            (bottom_left_x, bottom_left_y),
            w_spatial_x,
            w_spatial_y,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            label=f"Region {idx}",
        )
        ax_img.add_patch(rect)
        ax_img.text(
            cx_spatial,
            bottom_left_y + w_spatial_y + (0.02 * w_spatial_y),
            f"R{idx}",
            color=color,
            fontsize=10,
            weight="bold",
            ha="center",
            va="bottom",
        )

    ax_img.set_xlabel("X Space (um)")
    ax_img.set_ylabel("Y Space (um)")
    ax_img.set_title("Photoluminescence Image & Region ROIs")


def main():
    # target_regions = [
    #     {"center": (120, 150), "width": 16},
    #     {"center": (155, 140), "width": 16},
    #     {"center": (20, 240), "width": 16},
    # ]
    target_regions = [
        {"center": (80, 30), "width": roi_width},
        {"center": (80, 150), "width": roi_width},
        # {"center": (20, 60), "width": roi_width},
        {"center": (80, 80), "width": roi_width},
        {"center": (10, 100), "width": roi_width},
        {"center": (150, 100), "width": roi_width},
        {"center": (80, 100), "width": roi_width},
        # {"center": (110, 120), "width": roi_width},
    ]


    if NEW_MEASUREMENT:
        camera_binning = 1
        focus_point_size = 160
        focus_point_centre_x, focus_point_centre_y = 950, 1070

        n_windows_per_point = 10
        amp_dbm = -10
        dwell = 0.04
        n_iter = 10

        f_center = 2.87e9
        span = 0.3e9
        N = 301


        roi, x_space, y_space = pci.get_spacial_params(
            camera_binning, (focus_point_size, focus_point_centre_x, focus_point_centre_y)
        )
        print(f"Using ROI: {roi} with binning {camera_binning}x{camera_binning}")

        f_start, f_end, freqs = cs.calc_sweep_range(f_center, span, N)
        region_meta = region_to_metadata(x_space, y_space, target_regions)



        cam, sg = pci.connect_cam_RF(roi, camera_binning, 0.005)
        image = pci.read_image(cam, 1)

        plot_regions(image, x_space, y_space, region_meta)

        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
        time.sleep(0.1)

        try:
            counts_2D, prev_path = wODMR.measure_odmr(cam, sg, freqs, dwell, n_windows_per_point, n_iter, x_space, y_space)
            # _ = iFile.overwrite_2D_odmr_measurement(x_space, y_space, freqs, counts_2D, prev_path, print_saving=True)
        finally:
            cs.enable_sg386(sg, enable=False)
            cam.close()

    else:

        data = np.load(old_Measurement_Path)

        x_space = data["x"]
        y_space = data["y"]
        freqs = data["f"]
        counts_2D = data["odmrs"]
        region_meta = region_to_metadata(x_space, y_space, target_regions)
        # mean_image = np.mean(counts_2D, axis=-1) # average along the freq axis
        # plot_regions(mean_image, x_space, y_space, region_meta)

    print("Frequency sweeping complete. Processing defined regions...")
    region_odmrs = extract_region_odmrs(counts_2D, x_space, y_space, target_regions)

    # plot_stacked_odmrs(freqs, region_odmrs, region_meta, normalize=True, vertical_offset=0.0)
    plot_regions_and_odmrs(counts_2D, freqs, x_space, y_space, region_odmrs, region_meta, normalize=True)

if __name__ == "__main__":
    main()