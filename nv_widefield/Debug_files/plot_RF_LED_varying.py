import os
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
import nv_setup.cw_odmr.Lorentzian_fit as Lfit


def parse_metadata_file(filepath):
    """Parses timestamps and numerical values from the text configuration log."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Split into sections based on your header keywords
    sections = re.split(r'Following sweep ', content)

    oled_data = {}
    rf_data = {}

    for section in sections:
        if section.startswith('LED power'):
            # Matches: 'X'uA or 'X'mA followed by text blocks containing 'saved as hh-mm-ss'
            # Uses a greedy match to find the final timestamp if duplicates exist in a block
            pattern = r'(\d+)\s*(?:uA|mA)[^\n]*:\s*.*?saved as\s+(\d{2}-\d{2}-\d{2})'
            matches = re.findall(pattern, section, re.DOTALL)
            for val, time_str in matches:
                oled_data[int(val)] = time_str

        elif section.startswith('RF power'):
            # Matches signed integers: '-X dBm' or '-X.X dBm' followed by text blocks with 'saved as hh-mm-ss'
            pattern = r'(-?\d+)\s*dBm:\s*.*?saved as\s+(\d{2}-\d{2}-\d{2})'
            matches = re.findall(pattern, section, re.DOTALL)
            for val, time_str in matches:
                rf_data[int(val)] = time_str

        # Else, parse some other section to add

    return oled_data, rf_data



def plot_overlaid_data(data_map, base_dir, date_str, title, label_prefix, unit, cmap_name):
    """Loads matching files from disk and overlays them onto a single figure."""
    if not data_map:
        print(f"No valid data mapped for {title}")
        return

    plt.figure(figsize=(6, 6))
    colors = plt.get_cmap(cmap_name)(np.linspace(0, 0.85, len(data_map)))

    # Sort by numerical key (ascending order)
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

            popt, pcov, counts_norm, fitted_norm, baseline = Lfit.analyze_data(freqs, counts, 2)

            plt.plot(
                freqs / 1e9,
                # counts/max(counts), # normalize to the max value rather than
                counts_norm + 0.0008*idx,
                label=f"{label_prefix} = {val} {unit}",
                color=colors[idx],
                linewidth=2,
                marker=".",
                linestyle=""
                # make this dots
            )
            plt.plot(
                freqs / 1e9,
                fitted_norm + 0.0008*idx,
                label=f"{label_prefix} = {val} {unit}",
                color=colors[idx],
                linewidth=2,
                marker="",
                linestyle="--"
                # make this dashedline
            )
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    plt.xlabel("Frequency [GHz]", fontsize=12)
    # plt.ylabel("Binned Brightness (arb units/s)", fontsize=12)
    plt.yticks([])
    plt.title(title, fontsize=14, fontweight='bold', loc="left")
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend(loc="lower right", frameon=True,
    #             shadow=True, bbox_to_anchor=(+0.2, +0.2),
    #             fancybox=True, ncol=1)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",  # Anchor point on the legend box itself
        bbox_to_anchor=(1.02, 1),  # Places anchor point slightly past the right edge (x=1.02) at the top (y=1)
        frameon=True,
        shadow=True,
        fancybox=True,
        ncol=1
    )

    plt.tight_layout()


def main():
    date_str = "2026-06-15"
    desktop_dir = r"C:\Users\NVCFM\Desktop"
    txt_path = os.path.join(desktop_dir, "NVCFM_Data", date_str, "Checking_LED_illumination.txt")

    now = datetime.datetime.now()
    datestamp = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")
    project_root = "C:\\Users\\NVCFM\\Desktop"
    directory = os.path.join(project_root, "NVCFM_Data", datestamp)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Step 1: Extract profiles from textual logs
    try:
        oled_map, rf_map = parse_metadata_file(txt_path)
    except FileNotFoundError:
        print(f"Log file not found at: {txt_path}")
        return

    # Step 2: Render LED Power Dependence Group
    print(f"Loaded {len(oled_map)} entries for OLED sweeps.")
    plot_overlaid_data(
        data_map=oled_map,
        base_dir=desktop_dir,
        date_str=date_str,
        title="ODMR Dependence on OLED Current",
        label_prefix="OLED current",
        unit="uA",
        cmap_name="plasma"
    )
    plt.savefig(directory + "/ODMR dependence on OLED current.pdf")

    # Step 3: Render RF Power Dependence Group
    print(f"Loaded {len(rf_map)} entries for RF sweeps.")
    plot_overlaid_data(
        data_map=rf_map,
        base_dir=desktop_dir,
        date_str=date_str,
        title="ODMR Dependence on RF Power",
        label_prefix="RF Power (pre-amp)",
        unit="dBm",
        cmap_name="plasma"
    )

    plt.savefig(directory + "/ODMR dependence on RF power.pdf")
    # Display both plots asynchronously in SciView
    plt.show()


if __name__ == "__main__":
    main()