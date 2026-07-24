import numpy as np
import matplotlib.pyplot as plt
import os

import nv_setup.cw_odmr.Lorentzian_fit as Lfit
# Maintain directory path expansions
# import sys
# sys.path.append(os.path.abspath("..."))
# import nv_widefield.helper_classes.odmr_plotting as oPlot



# Parameters
# date = "2026-07-17"
# # time = "14-10-59" # high field single dip with weird trends in middle pixels
# # time = "15-16-49" #
# # time = "15-23-28" # super noisy low brightness measurement of zero-field
# # time = "15-40-31" #
# time = "16-50-33"

date = "2026-07-22"
time = "15-22-30"
target_idx = 15  # The specific spatial x-index to extract across all y's
slice_axis = "y"        # Choose "x" or "y" to slice along that axis
                        # slicing along x, means y is constant (horizontal line)
max_peaks = 6
plot_fits = False

diff_start_idx = 0
diff_end_idx = 31

offset = 0.003

def plot_odmr_differences(freqs, axis_points, counts_2D, target_x_idx, start_idx, end_idx):
    """
    Normalizes ODMR traces in a specified y-index range, computes consecutive
    differences (y_{i+1} - y_i), and plots the residuals.
    """
    # print("Looking at differences between following Y:", selected_axis)
    selected_axis = axis_points[diff_start_idx:diff_end_idx]


    # Extract and normalize traces in the specified range
    normalized_traces = []
    for idx in range(start_idx, end_idx):
        if slice_axis == "x":
            raw_counts = counts_2D[target_x_idx, idx, :]
        else:
            raw_counts = counts_2D[idx, target_x_idx, :]
        normalized_traces.append(raw_counts / np.max(raw_counts))
        # normalized_traces.append(raw_counts)

    plt.figure(figsize=(10, 6), layout="constrained")
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(selected_axis) - 1))

    # Calculate and plot consecutive trace differences
    for i in range(len(normalized_traces) - 1):
        diff = normalized_traces[i + 1] - normalized_traces[i]
        if slice_axis == "x":
            label_text = f"Δ (x={selected_axis[i + 1]:.4f} - x={selected_axis[i]:.4f})"
        else:
            label_text = f"Δ (y={selected_axis[i + 1]:.4f} - y={selected_axis[i]:.4f})"

        plt.plot(
            freqs / 1e9,
            diff,
            label=label_text,
            color=colors[i],
            linewidth=1.5,
            alpha=0.85
        )

    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Frequency [GHz]", fontsize=12)
    plt.ylabel("Difference in Normalized Counts (arb units)", fontsize=12)
    if slice_axis == "x":
        plt.title(f"Consecutive ODMR Differences (x-indices {start_idx} to {end_idx})", fontsize=13)
    else:
        plt.title(f"Consecutive ODMR Differences (y-indices {start_idx} to {end_idx})", fontsize=13)

    # plt.legend(loc="best", frameon=True, ncol=2, fontsize=8)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.show()

def plot_dip_vs_axis(y_positions, centers, fixed_pos_val):
    """
    Plots the extracted Lorentzian center frequency against spatial y coordinates.
    """
    plt.figure(figsize=(8, 5), layout="constrained")
    plt.plot(y_positions, centers, 'o-', color='crimson', markersize=6, linewidth=1.5, label='Fitted $f_0$')

    if slice_axis == "x":
        plt.xlabel("x space (mm)", fontsize=11)
        plt.title(f"Dip Center Frequency $f_0$ vs. x (Fixed y = {fixed_pos_val:.4f} mm)", fontsize=12)
    else:
        plt.xlabel("y space (mm)", fontsize=11)
        plt.title(f"Dip Center Frequency $f_0$ vs. y (Fixed x = {fixed_pos_val:.4f} mm)", fontsize=12)

    plt.ylabel("Lorentzian Dip Center [GHz]", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best")
    plt.show()

def plot_odmrs(freqs, sweep_results, fixed_axis_val, start_idx, end_idx):
    N_steps = len(sweep_results)//2
    print("There should be ", N_steps, "ODMRS plotted")
    plt.figure(figsize=(10, 6), layout="constrained")

    fitted_axis_positions = []
    dip_centers_ghz = []

    # Use plasma colormap matching previous scripts
    colors = plt.cm.plasma(np.linspace(0, 0.85, N_steps))



    for idx, (axis_pos, counts) in enumerate(sweep_results.items()):

        if idx < start_idx or idx > end_idx:
            # print(f"skipping over index={idx}")
            continue
        # max_counts = np.max(counts)
        # normalized_counts = counts / max_counts if max_counts > 0 else counts

        # Alternatively, uncomment to stack with a vertical offset if traces overlap heavily:
        # y_data = normalized_counts + 0.05 * idx
        # y_data = normalized_counts

        plt.plot(
            freqs / 1e9,
            counts / max(counts) + offset*idx,
            label=f"y = {axis_pos:.4f} mm",
            color=colors[idx],
            linewidth=1.5,
            alpha=0.85
        )

        if idx == 1:
            popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
            contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
            plt.axvline(
                x=dip_Freqs[0], # change index to choose which dip to plot a vline for
                color='red',
                linestyle='--',
                linewidth=1.2,
                alpha=0.7,
                label='2.87 GHz'
            )
            plt.axvline(
                x=dip_Freqs[-1], # change index to choose which dip to plot a vline for
                color='red',
                linestyle='--',
                linewidth=1.2,
                alpha=0.7,
                label='2.87 GHz'
            )

        if plot_fits:
            try:
                # Fit the ODMR trace and retrieve fit parameters along with the model curve
                # If your Lfit signature differs, adjust this call accordingly
                popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
                contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
                snrs = Lfit.get_SNRs(baseline, counts, freqs, popt)

                # Plot the fitted curve on top of the raw data with matching offset
                plt.plot(
                    freqs / 1e9,
                    fitted_norm + offset*idx,
                    color=colors[idx],
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.9
                )

                # Calculate center frequency and dip minimum point coordinate
                center_freq_ghz = dip_Freqs[0]
                dip_minimum_val = np.min(fitted_norm) + offset*idx

                # Overlay a red dot at the fitted minimum of the dip
                plt.plot(
                    center_freq_ghz,
                    dip_minimum_val,
                    marker="o",
                    color="red",
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    zorder=12
                )

                # # Print parameters to standard output for this specific spatial point
                # print(f"--- Fit Results for y = {axis_pos:.4f} mm ---")
                # Lfit.print_contrast_snr_FWHM(contrasts, snrs, FWHMs, dip_Freqs)
                fitted_axis_positions.append(axis_pos)
                dip_centers_ghz.append(dip_Freqs[0])

            except Exception as fit_error:
                # Catch fitting failures gracefully to avoid halting the full spatial sweep
                print(f"[Warning] Fit failed for y = {axis_pos:.4f} mm: {fit_error}")

    plt.xlabel("Frequency [GHz]", fontsize=12)
    plt.ylabel("Normalized Brightness (arb units/s)", fontsize=12)

    if slice_axis == "x":
        plt.title(f"Widefield ODMR Trace Slice (Fixed y = {fixed_axis_val:.4f} mm)", fontsize=13)
    else:
        plt.title(f"Widefield ODMR Trace Slice (Fixed x = {fixed_axis_val:.4f} mm)", fontsize=13)
    # plt.xlim(2.68,2.79)
    # plt.ylim(0.997, 1 + len(z_sweep_results.items())*0.0002)
    # plt.grid(True, linestyle="--", alpha=0.5)

    # If there are many Y points, truncate the legend or display columns

    # plt.legend(loc="best", frameon=True, shadow=False, ncol=2, fontsize=9)

    plt.show()
    if len(dip_centers_ghz) > 0:
        plot_dip_vs_axis(fitted_axis_positions, dip_centers_ghz, fixed_axis_val)


def plot_magnet_with_slice(x_space, y_space, B_field_2D, target_axis_idx):
    """
    Plots the 2D magnetic field distribution and overlays a vertical line
    at the spatial x-coordinate corresponding to target_x_idx.
    """
    # Safeguard x-bounds and retrieve physical value
    space_to_check = y_space if slice_axis == "x" else x_space
    if target_axis_idx >= len(space_to_check) or target_axis_idx < 0:
        raise IndexError(f"Target index {target_axis_idx} is out of bounds for the selected space slice.")

    slice_axis_coord = space_to_check[target_axis_idx]

    # if target_x_idx >= len(x_space) or target_x_idx < 0:
    #     raise IndexError(f"Target index {target_x_idx} is out of bounds for x_space of size {len(x_space)}")

    # slice_axis_coord = space_to_check[target_axis_idx]

    # Establish robust colormap scaling (1st to 99th percentile) to handle hot pixels
    # vmin = np.percentile(B_field_2D, 1)
    # vmax = np.percentile(B_field_2D, 99)

    plt.figure(figsize=(8, 6), layout='constrained')


    mesh = plt.pcolormesh(x_space, y_space, B_field_2D, shading='nearest', cmap='bwr')


    # Plot high-contrast indicators for the sliced region
    # axvline spans the full height of the axes
    if slice_axis == "x":
        plt.axhline(
            y=slice_axis_coord,
            color='black',
            linestyle=':',
            linewidth=5.0,
            label=f"Slice line (y = {slice_axis_coord:.4f} mm)",
            zorder=10
        )
    else:
        plt.axvline(
            x=slice_axis_coord,
            color='black',
            linestyle=':',
            linewidth=5.0,
            label=f"Slice line (x = {slice_axis_coord:.4f} mm)",
            zorder=10
        )

    # plt.axvline(
    #     x=slice_axis_coord,
    #     color='black',
    #     linestyle=':',
    #     linewidth=5.0,
    #     label=f"Slice line (x = {slice_axis_coord:.4f} mm)",
    #     zorder=10
    # )

    # # Optional: Highlight the terminals of the vertical scan
    # plt.scatter(
    #     [slice_axis_coord, slice_axis_coord],
    #     [y_space[0], y_space[-1]],
    #     color='cyan',
    #     edgecolors='black',
    #     s=80,
    #     zorder=11
    # )

    # plt.gca().invert_yaxis()  # Maintain spatial match to physical sensor layout

    plt.colorbar(mesh, label=r'$\partial$ B (T)')
    plt.xlabel('x space (mm)')
    plt.ylabel('y space (mm)')
    if slice_axis == "x":
        plt.title(f'2D Magnet Image with Slice Overlay (y_ind={target_axis_idx})')
    else:
        plt.title(f'2D Magnet Image with Slice Overlay (x_ind={target_axis_idx})')

    plt.legend(loc='upper right', framealpha=0.9)

    plt.show()


def main():

    desktop_dir = "C:\\Users\\NVCFM\\Desktop"
    filepath = os.path.join(desktop_dir, "NVCFM_Data", date, f"widefield_cw_odmr_{time}.npz")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Specified dataset could not be located at {filepath}")

    # Load widefield acquisition structure
    data = np.load(filepath)
    x_points = data["x"]
    y_points = data["y"]
    freqs = data["f"]
    B_Z_overall = data["magnet"]
    counts_2D = data["odmrs"]  # Expected shape: (X_pixels, Y_pixels, Freq_points)

    primary_points = x_points if slice_axis == "x" else y_points
    secondary_points = y_points if slice_axis == "x" else x_points

    # Check bounds of target x-slice
    # if target_x_idx >= len(x_points) or target_x_idx < 0:
    #     raise IndexError(f"Specified x index {target_x_idx} is out of bounds for x space of size {len(x_points)}")
    if target_idx >= len(primary_points) or target_idx < 0:
        raise IndexError(f"Specified index {target_idx} is out of bounds for the selection.")

    # Extract the coordinate value
    actual_x_val = x_points[target_idx]
    print(f"Slicing spatial data along x_index={target_idx} (x={actual_x_val:.4f} mm)")

    # Slice out all Y-scans for the single specified X-coordinate
    # Structure: { y_coordinate_val: 1D array of counts across frequencies }
    slice_results = {}
    # for y_idx, y_val in enumerate(y_points):
    #     # Slice format extracts [X_fixed, Y_current, all_frequencies]
    #     slice_results[y_val] = counts_2D[target_idx, y_idx, :]
    for secondary_idx, secondary_val in enumerate(secondary_points):
        if slice_axis == "x":
            slice_results[secondary_val] = counts_2D[target_idx, secondary_idx, :]
        else:
            slice_results[secondary_val] = counts_2D[secondary_idx, target_idx, :]

    # Run the plot routine
    # oPlot.plot_magnet_image(x_points, y_points, B_Z_overall)
    plot_magnet_with_slice(x_points, y_points, B_Z_overall, target_idx)
    plot_odmrs(freqs, slice_results, actual_x_val,
               diff_start_idx,
               diff_end_idx)

    # plot_odmr_differences(
    #     freqs,
    #     y_points,
    #     counts_2D,
    #     target_idx,
    #     diff_start_idx,
    #     diff_end_idx
    # )

if __name__ == "__main__":
    main()