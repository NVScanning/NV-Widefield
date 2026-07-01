import matplotlib.pyplot as plt
import numpy as np


"""
Similar to ODMR plotting, but for some odd debug files I use
"""


def plot_Z_dep_graph(z_range: np.ndarray, kcps: np.ndarray):
    plt.figure(figsize=(8, 5))
    plt.plot(z_range, kcps, "-o", markersize=2)
    plt.xlabel("Z Position (mm)")
    plt.ylabel("kcps")
    plt.title("Counts as a fn of Z")
    plt.grid(True)
    plt.show()

def plot_binned_snr_contr(binned_contrast_avg,ubinned_contrast_avg, binned_snr_avg,ubinned_snr_avg, n_bins: int):
    bins = np.power(2,range(n_bins))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('log num bins')
    ax1.set_ylabel('Average SNR', color=color)
    ax1.errorbar(bins, binned_snr_avg, yerr=ubinned_snr_avg, markersize=3, capsize=5, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Average contrast(%)', color=color)  # we already handled the x-label with ax1
    ax2.errorbar(bins, np.array(binned_contrast_avg)*100, yerr=np.array(ubinned_contrast_avg)*100, markersize=3, capsize=5, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.xscale('log')
    plt.xticks(bins)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(True)
    plt.show()
def plot_exposure_snr_contr(contr_avg, ucontr_avg, snr_avg, usnr_avg, n_windows: int):
    windows = np.power(2,range(n_windows))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('log num windows')
    ax1.set_ylabel('Average SNR', color=color)
    ax1.errorbar(windows, snr_avg, yerr=usnr_avg, markersize=3, capsize=5, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Average contrast(%)', color=color)  # we already handled the x-label with ax1
    ax2.errorbar(windows, np.array(contr_avg) * 100, yerr=np.array(ucontr_avg) * 100, markersize=3, capsize=5, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.xscale('log')
    plt.xticks(windows)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid(True)
    plt.show()

def plot_exposure_snr_contr_bin(contr_avg, snr_avg, n_windows: int, n_bins: int):
    windows = range(n_windows)
    bins = range(n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    mesh1 = ax1.pcolormesh(windows, bins, contr_avg.T, shading='nearest', cmap='inferno')
    fig.colorbar(mesh1, ax=ax1, label='Contrast')
    ax1.set_xlabel('log Num Windows')
    ax1.set_ylabel('log Num Bins')
    ax1.set_title('ODMR Contrast Heatmap')
    ax1.set_xticks(windows)
    ax1.set_xticklabels(windows)
    ax1.set_yticks(bins)
    ax1.set_yticklabels(bins)

    mesh2 = ax2.pcolormesh(windows, bins, snr_avg.T, shading='nearest', cmap='inferno')
    fig.colorbar(mesh2, ax=ax2, label='SNR')
    ax2.set_xlabel('log Num Windows')
    ax2.set_ylabel('log Num Bins')
    ax2.set_title('Signal-to-Noise Ratio (SNR) Heatmap')
    ax2.set_xticks(windows)
    ax2.set_xticklabels(windows)
    ax2.set_yticks(bins)
    ax2.set_yticklabels(bins)

    # Automatically optimize spacing between subplots to avoid overlapping labels
    plt.tight_layout()
    plt.show()
