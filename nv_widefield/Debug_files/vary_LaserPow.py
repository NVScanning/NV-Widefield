import time
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import helper_classes.pco_cam_interface as pci
import helper_classes.Novanta_Laser as nLas
import helper_classes.odmr_plotting as oPlot



NEW_MEASUREMENT = False  # Set to True for hardware run, False to parse log_text

old_Measurement_Path = "C:\\Users\\NVCFM\\ownCloud\\QIQM\\NVCFM Data\\2026-08-25\\saturation_sweep_02-58-54.txt"


def read_txt(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Log text file not found at: {txt_path}")

    powers = []
    means = []
    std_t = []

    power_pattern = re.compile(r"Setting laser power to ([\d\.]+)\s*mW")
    stats_pattern = re.compile(r"Mean Brightness:\s*([\d\.e\+-]+)\s*\|\s*Temporal SD:\s*([\d\.e\+-]+)")

    current_power = None

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()

            power_match = power_pattern.search(line_str)
            if power_match:
                current_power = float(power_match.group(1))
                continue

            stats_match = stats_pattern.search(line_str)
            if stats_match and current_power is not None:
                mean_val = float(stats_match.group(1))
                sd_val = float(stats_match.group(2))

                powers.append(current_power)
                means.append(mean_val)
                std_t.append(sd_val)

                current_power = None

    return np.array(powers), np.array(means), np.array(std_t)
class NovantaLaserWrapper:
    """Wrapper around LaserInterface to expose set_power and get_power methods."""

    def __init__(self, port='COM3', baudrate=9600, timeout=1.0):
        self.laser = nLas.LaserInterface(port=port, baudrate=baudrate, timeout=timeout)
        self.laser.send_command("CONTROL=POWER")
        self.laser.send_command("ON")

    def set_power(self, power_mw: float):
        self.laser.send_command(f"POWER={int(power_mw)}")

    def get_power(self) -> float:
        resp = self.laser.send_command("POWER?")
        match = nLas.re.search(r"[-+]?\d*\.\d+|\d+", resp) if hasattr(nLas, 're') else None
        if match:
            return float(match.group(0))
        return 0.0

    def close(self):
        self.laser.send_command("OFF")
        self.laser.close()


def saturation_model(P, I_max, P_sat, I_bg):
    return I_bg + (I_max * P) / (P + P_sat)


def run_saturation_sweep(
        laser,
        cam,
        power_setpoints,
        settle_time_s=1800,
        num_frames=100,
        exposure_time_s=0.01,
        roi=None
):
    mean_intensities = []
    std_temporal = []
    std_spatial = []
    actual_powers = []

    print(f"Starting overnight saturation sweep across {len(power_setpoints)} setpoints.")
    print(
        f"Estimated total run time: ~{(len(power_setpoints) * (settle_time_s + num_frames * exposure_time_s) + 0.1) / 3600:.2f} hours.\n")

    for i, p_set in enumerate(power_setpoints):
        print(f"[{i + 1}/{len(power_setpoints)}] Setting laser power to {p_set} mW...")

        laser.set_power(p_set)
        p_actual = laser.get_power() if hasattr(laser, "get_power") else p_set
        actual_powers.append(p_actual)

        print(f"   Stabilizing for {settle_time_s / 60:.1f} minutes...")
        time.sleep(settle_time_s)

        print(f"   Acquiring {num_frames} frames...")

        # Collect actual camera stack via PCI interface
        stack = pci.record_stack(cam, num_frames)

        # Temporal statistics across frame dimension (axis 0)
        frame_means = stack.mean(axis=(1, 2))
        mean_val = np.mean(frame_means)
        std_temp = np.std(frame_means, ddof=1)

        # Spatial statistics across average image
        mean_frame_spatial = stack.mean(axis=0)
        std_spat = np.std(mean_frame_spatial)

        mean_intensities.append(mean_val)
        std_temporal.append(std_temp)
        std_spatial.append(std_spat)

        print(f"   Mean Brightness: {mean_val:.2e} | Temporal SD: {std_temp:.2e}")

    powers = np.array(actual_powers)
    means = np.array(mean_intensities)
    std_t = np.array(std_temporal)
    std_s = np.array(std_spatial)

    save_path = oPlot.get_newfile_dir("saturation_sweep_")
    np.savez(
        save_path,
        power=powers,
        mean_brightness=means,
        std_temporal=std_t,
        std_spatial=std_s
    )

    return powers, means, std_t, std_s


def fit_saturation_curve(powers, means, std_t):
    p0 = [max(means) - min(means), np.median(powers), min(means)]
    sigma = np.where(std_t == 0, 1e-6, std_t)

    popt, pcov = curve_fit(
        saturation_model,
        powers,
        means,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def plot_saturation_results(powers, means, std_t, fit_params=None):
    fig, ax1 = plt.subplots(figsize=(8, 5), layout="constrained")

    color = "crimson"
    ax1.set_xlabel("Laser Power [mW]", fontsize=12)
    ax1.set_ylabel("Mean NV Brightness [counts/s]", color=color, fontsize=12)
    ax1.errorbar(
        powers,
        means,
        yerr=std_t,
        fmt="o",
        color=color,
        ecolor="red",
        capsize=4,
        label="Data (±1 temporal SD)"
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)

    if fit_params is not None:
        I_max, P_sat, I_bg = fit_params
        P_smooth = np.linspace(min(powers), max(powers), 200)
        I_fit = saturation_model(P_smooth, I_max, P_sat, I_bg)
        ax1.plot(
            P_smooth,
            I_fit,
            "--",
            color="black",
            linewidth=1.8,
            label=f"Fit: $P_{{sat}}={P_sat:.2f}$ mW"
        )

    ax2 = ax1.twinx()
    color = "blue"
    ax2.set_ylabel("Relative Temporal uncertainty [unitless]", color=color, fontsize=12)
    ax2.plot(powers, std_t/means, "s", color=color, linewidth=1.2, label="Relative uncertainty")
    ax2.tick_params(axis="y", labelcolor=color)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
    # ax1.legend(loc="upper left")
    # ax2.legend(loc="upper left")
    plt.title("NV Saturation & Power Stability Sweep", fontsize=13)
    plt.show()


def main():
    power_setpoints = [5, 10, 20, 50, 75, 100, 125, 150, 200, 250]
    if not NEW_MEASUREMENT:
        powers, means, std_t = read_txt(old_Measurement_Path)
    else:
        try:

            settle_time_s = 3600
            num_frames = 10000
            exposure_time_s = 0.02

            binning_amount = 1

            # Instantiate hardware interface objects
            laser = NovantaLaserWrapper(port='COM3')

            # Preserved spatial setup
            roi, _, _ = pci.get_spacial_params(binning_amount, (128, 1024, 1024))
            cam = pci.connect_cam(roi, binning_amount, forced_exposure=0.01)
            powers, means, std_t, std_s = run_saturation_sweep(
                laser=laser,
                cam=cam,
                power_setpoints=power_setpoints,
                settle_time_s=settle_time_s,
                num_frames=num_frames,
                exposure_time_s=exposure_time_s,
                roi=roi
            )

        finally:
            laser.close()



    popt, perr = fit_saturation_curve(powers, means, std_t)
    I_max, P_sat, I_bg = popt

    print("\n--- Fit Results ---")
    print(f"I_max: {I_max:.3e} ± {perr[0]:.3e} counts")
    print(f"P_sat: {P_sat:.3f} ± {perr[1]:.3f} mW")
    print(f"I_bg : {I_bg:.3e} ± {perr[2]:.3e} counts")

    plot_saturation_results(powers, means, std_t, fit_params=popt)


if __name__ == "__main__":
    main()