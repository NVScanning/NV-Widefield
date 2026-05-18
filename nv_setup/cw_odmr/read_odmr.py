import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import Lorentzian_fit as Lfit
import scanned_cw_odmr as scwodmr

"""
this file is to read npy/z files created by cw_odmr

Simply paste in the date and time of the file as it's stored in /NVCFM_Data/date and it will display it


note: measurements previous to 2026-05-13 at 15:02 only have cps data, not frequency stored
^ this means you don't have any x data, and filetype is different

note: scanned odmrs from may 15, 2026 don't have the odmrs saved only the frequency deltas 
"""

# Params to change
date = "2026-05-18"
time = "14-24-06"
scanned_measurement = True


# TODO: add reading for scanned odmrs, maybe the option to select a pixel or smth idk


script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent

if (not scanned_measurement):


    # Because its in YY-MM-DD we can do:
    is_freq_saved = False
    if (date > "2026-05-13"):
        is_freq_saved = True
    elif (date == "2026-05-13") & (time > "15-02-05"):
        is_freq_saved = True

    directory = os.path.join(project_root, "NVCFM_Data", date, "cw_odmr_" + time)


    plt.figure(figsize=(8, 5))

    if (is_freq_saved):
        data = np.load(directory + ".npz") # npy for old, npz for new measurements
        plt.plot(data["x"]/10**9,data["y"], "-o", markersize=2) # uncomment for measurements after 2026-05-13 at 15 hours
        plt.xlabel("Frequency (GHz)") # uncomment for measurements after 2026-05-13 at 15 hours
    else:
        data = np.load(directory + ".npy") # npy for old, npz for new measurements
        plt.plot(data, "-o", markersize=2) # uncomment for measurements previous to 2026-05-13 at 15 hours
        plt.xlabel("index (frequency isn't saved)") # uncomment for measurements previous to 2026-05-13 at 15 hours

    plt.ylabel("kcps")
    plt.title("ODMR")
    plt.grid(True)
    plt.show()

    if (is_freq_saved):
        freqs, counts = data["x"], data["y"]
        max_peaks = 2

        popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
        Lfit.print_dip_params(popt)
        Lfit.print_SNR(baseline, counts, freqs / 10 ** 9, popt)
        Lfit.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)
else:
    # scanned measurement
    directory = os.path.join(project_root, "NVCFM_Data", date, "scanned_cw_odmr_" + time)
    plt.figure(figsize=(8, 5))

    data = np.load(directory + ".npz") # npy for old, npz for new measurements
    #
    # # shading='auto' handles the coordinate mapping automatically
    # mesh = plt.pcolormesh(data["x"], data["y"], data["deltas"], shading='auto', cmap='viridis')
    #
    # plt.colorbar(mesh, label='frequency delta (GHz)')
    # plt.xlabel('x space (mm)')
    # plt.ylabel('y space (mm)')
    # plt.title('frequency delta Heatmap')
    # plt.show()
    #
    # # TODO: look at an odmr scan for a specific point

    x_points = data["x"]
    y_points = data["y"]
    freqs = data["f"]

    freq_deltas = data["deltas"]
    counts_2D = data["odmrs"]

    # Enable interactive mode so plt.show() doesn't freeze the script
    # plt.ion()

    # --- Your existing plotting code ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ext = [x_points.min(), x_points.max(), y_points.min(), y_points.max()]
    # Assuming data shape is (X_len, Y_len, Freq_len), project down to a 2D map
    # e.g., taking the mean count across all frequencies for the spatial map

    mesh = ax.imshow(
        freq_deltas, extent=ext, aspect="auto", origin="lower", cmap="viridis"
    )
    plt.colorbar(mesh, label="Mean Counts (kcps)")
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")
    ax.set_title("frequency delta Heatmap")

    # Force draw the plot window without pausing execution
    plt.draw()
    plt.pause(0.001)

    # --- User Interaction Loop ---
    print("\n" + "=" * 40)
    print("Spatial scan complete. Select a pixel to view its 1D ODMR spectrum.")
    print(f"Available X indices: 0 to {len(x_points) - 1}")
    print(f"Available Y indices: 0 to {len(y_points) - 1}")
    print("Type 'exit' to quit.")
    print("Type 'reanalyze' to re-analyze image from saved odmrs")
    print("=" * 40)

    while True:
        user_input = input(
            "\nEnter pixel indices as 'X, Y' (e.g.: '5, 12'): "
        ).strip()

        if user_input.lower() == "exit":
            print("Exiting interaction loop.")
            break
        if user_input.lower() == "reanalyze":
            print("Reanalyzing, will write new file")

            freq_deltas, problem_points = scwodmr.counts_to_delta_freq(x_points, y_points, counts_2D, freqs)
            print("following indices couldn't fit properly:")
            print(problem_points)
            scwodmr.plot_image(x_points, y_points, freq_deltas)
            scwodmr.save_odmr_measurement(x_points, y_points, freqs, freq_deltas, counts_2D)

        try:
            # Parse the input string into integers
            x_str, y_str = user_input.split(",")
            x_ind = int(x_str.strip())
            y_ind = int(y_str.strip())

            # Boundary validation
            if not (0 <= x_ind < len(x_points)) or not (
                    0 <= y_ind < len(y_points)
            ):
                print(
                    f"Indices out of bounds! Keep X within [0, {len(x_points) - 1}] and Y within [0, {len(y_points) - 1}]."
                )
                continue

            # --- Extract and plot the 1D ODMR line scan for that pixel ---
            # Slice the 3D data array at the selected spatial pixel
            odmr_spectrum = counts_2D[x_ind, y_ind, :]

            # Open a new window for the 1D spectrum so the map stays visible
            # plt.figure()
            # plt.plot(freqs / 1e9, odmr_spectrum, "o-", color="crimson")
            # plt.xlabel("Frequency (GHz)")
            # plt.ylabel("Counts (kcps)")
            # plt.title(
            #     f"1D ODMR Spectrum at Pixel ({x_ind}, {y_ind}) | X={x_points[x_ind]:.3f}, Y={y_points[y_ind]:.3f}"
            # )
            # plt.grid(True)
            # plt.draw()
            # plt.pause(0.001)


            counts = counts_2D[x_ind,y_ind]
            max_peaks = 2
            popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, max_peaks)
            print("analyzed data")
            Lfit.print_dip_params(popt)
            # contrasts, FWHMs, dip_Freqs = Lfit.get_dip_params(popt)
            # Lfit.print_SNR(baseline, counts, freqs / 10 ** 9, popt)
            Lfit.plot_fitted_data(freqs / 10 ** 9, counts_norm, fitted_norm)

        except ValueError:
            print(
                "Invalid format. Please enter two integers separated by a comma (e.g., '2, 4')."
            )

    # Turn off interactive mode and keep windows open at the very end
    plt.ioff()
    plt.show()

