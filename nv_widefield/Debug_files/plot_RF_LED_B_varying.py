import os
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
import nv_setup.cw_odmr.Lorentzian_fit as Lfit

def parse_metadata_file(filepath):
    """Parses timestamps and numerical values from the text configuration log for multiple sweeps."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Split flexibly on either "Following sweep " or "Following is a sweep in "
    sections = re.split(r'Following\s+(?:is a\s+)?sweep\s+(?:in\s+)?', content)

    sweep_results = {}

    for section in sections:
        # 1. Parse OLED / LED current sweeps
        if section.startswith('LED power') or section.startswith('OLED power'):
            pattern = r'(\d+)\s*(?:uA|mA)[^\n]*:\s*.*?saved as\s+(\d{2}-\d{2}-\d{2})'
            matches = re.findall(pattern, section, re.DOTALL)
            if matches:
                data_map = {int(val): time_str for val, time_str in matches}
                sweep_results["oled"] = {
                    "data": data_map,
                    "title": "ODMR Dependence on OLED Current",
                    "label_prefix": "OLED current",
                    "unit": "uA",
                    "cmap": "plasma"
                }

        # 2. Parse RF power sweeps (Updated to support decimals and 'Saving as:' format)
        elif section.startswith('RF power'):
            # This pattern matches both:
            #   "-30 dBm:\n saved as 15-53-39"
            #   "power=-51.67 dBm\n Saving as: cw_odmr_16-36-27.npz"
            pattern = r'(?:power=)?(-?\d+(?:\.\d+)?)\s*dBm.*?(?:saved as|Saving as:\s+cw_odmr_)\s*(\d{2}-\d{2}-\d{2})'
            matches = re.findall(pattern, section, re.DOTALL)
            if matches:
                # Store keys as floats to preserve decimal precision (-51.67, etc.)
                data_map = {float(val): time_str for val, time_str in matches}
                sweep_results["rf"] = {
                    "data": data_map,
                    "title": "ODMR Dependence on RF Power",
                    "label_prefix": "RF Power (pre-amp)",
                    "unit": "dBm",
                    "cmap": "plasma"
                }

        # 3. Parse number of magnets sweeps
        elif section.startswith('num magnets') or section.startswith('magnet count'):
            pattern = r'(\d+)\s*magnets?:\s*.*?saved as\s+(\d{2}-\d{2}-\d{2})'
            matches = re.findall(pattern, section, re.DOTALL)
            if matches:
                data_map = {int(val): time_str for val, time_str in matches}
                sweep_results["magnets"] = {
                    "data": data_map,
                    "title": "ODMR Dependence on Number of Magnets",
                    "label_prefix": "Magnets",
                    "unit": "",
                    "cmap": "plasma"
                }

    return sweep_results

def plot_overlaid_data(data_map, base_dir, date_str, title, label_prefix, unit, cmap_name):
    """Loads matching files from disk and overlays them onto a single figure."""
    if not data_map:
        print(f"No valid data mapped for {title}")
        return

    plt.figure(figsize=(6, 6))
    colors = plt.get_cmap(cmap_name)(np.linspace(0, 0.85, len(data_map)))
    sorted_items = sorted(data_map.items())

    for idx, (val, time_str) in enumerate(sorted_items):
        filename = f"cw_odmr_{time_str}.npz"
        full_path = os.path.join(base_dir, "NVCFM_Data", date_str, filename)

        if not os.path.exists(full_path):
            print(f"File missing: {full_path} -- skipping profile value {val}{unit}")
            continue

        try:
            data = np.load(full_path)
            freqs = data["x"]
            counts = data["y"]

            if "magnet" in title.lower():
                popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, 6)
            else:
                popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, 2)

            plt.plot(
                freqs / 1e9,
                counts_norm + 0.005 * idx,
                label=f"{label_prefix} = {val} {unit}".strip(),
                color=colors[idx],
                marker=".",
                markersize=2,
                linestyle=""
            )
            plt.plot(
                freqs / 1e9,
                fitted_norm + 0.005 * idx,
                label=f"{label_prefix} = {val} {unit}".strip(),
                color=colors[idx],
                linewidth=1.5,
                marker="",
                linestyle="--"
            )
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    plt.xlabel("Frequency [GHz]", fontsize=12)
    plt.yticks([])
    plt.title(title, fontsize=14, fontweight='bold', loc="left")

    if "magnet" in title.lower():
        plt.xlim(2.69,3.05)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        shadow=True,
        fancybox=True,
        ncol=1
    )
    plt.tight_layout()


def main():
    # File to look at
    date_str = "2026-06-19"
    desktop_dir = r"C:\Users\NVCFM\Desktop"
    file_name = "B_thin_bulk_redo_LED.txt" # Jun 19
    # file_name = "B_thin_bulk_LED.txt" # Jun 18
    # file_name = "Current_OLED_illumination.txt" # jun 17
    # file_name = "Z_B_OLED_illumination.txt" # jun 16
    # file_name = "RF_Current_OLED_illumination.txt" # jun 15
    txt_path = os.path.join(desktop_dir, "NVCFM_Data", date_str, file_name)

    # Output save directories
    now = datetime.datetime.now()
    datestamp = now.strftime("%Y-%m-%d")
    save_directory = os.path.join(desktop_dir, "NVCFM_Data", datestamp)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Step 1: Extract any active profiles from the textual logs
    try:
        active_sweeps = parse_metadata_file(txt_path)
    except FileNotFoundError:
        print(f"Log file not found at: {txt_path}")
        return

    if not active_sweeps:
        print("No valid sweep configurations parsed from file.")
        return

    # Step 2: Dynamically loop through and generate plots for found parameters
    for sweep_key, config in active_sweeps.items():
        print(f"\nProcessing parsed sweep: {sweep_key} ({len(config['data'])} data points found)")

        plot_overlaid_data(
            data_map=config["data"],
            base_dir=desktop_dir,
            date_str=date_str,
            title=config["title"],
            label_prefix=config["label_prefix"],
            unit=config["unit"],
            cmap_name=config["cmap"]
        )

        # Save output figure to folder using config metadata
        sanitized_title = (config["title"] + "_" + date_str).replace(" ", "_").lower()
        pdf_filename = f"{sanitized_title}.tif"
        plt.savefig(os.path.join(save_directory, pdf_filename), dpi=300)

    plt.show()


if __name__ == "__main__":
    main()