import time
import os
from datetime import datetime
from enum import Enum, auto
import numpy as np
import pyvisa
import matplotlib.pyplot as plt
from tqdm import tqdm
from zhinst.toolkit import Session

import sys
sys.path.append(os.path.abspath(".."))
from nv_setup import connection_setup as cs

"""
ODMR (optically detected magnetic resonance) measurement file

can use AM, FM modulation or LSR(idk what this is)

this is made to work with the lock-in amp (mfli)
"""


# ----------------------------
# Connection helpers
# ----------------------------
def connect_mfli(host: str, device_id: str, demod_index: int = 0):
    # mfli is the lock-in amp
    session = Session(host)
    mfli = session.connect_device(device_id)
    print("Connected to MFLI")

    demod = mfli.demods[demod_index]
    demod.enable(1)
    return mfli, demod


# ----------------------------
# Sweep configuration
# ----------------------------

def calc_freq_range(f_center: float, span: float, n: int):
    f_start = f_center - span / 2
    f_end = f_center + span / 2
    freqs = np.linspace(f_start, f_end, n)
    return freqs, f_start, f_end


def compute_dwell_from_demod(demod, multiplier: float = 3.0):
    """Compute dwell time as multiplier * demod time constant."""
    tau = demod.timeconstant() # ms
    print("LPF time constant at:", tau*1e3, "ms")
    return float(multiplier * tau), float(tau)

def AM_modulation(sg, mod_freq, mod_depth):
    sg.write("TYPE 0")          # 0 = AM
    sg.write("MFNC 3")          # 3 = Sine
    sg.write(f"RATE {mod_freq}")  # Hz 
    sg.write(f"ADEP {mod_depth}")  # 0–100 %
    sg.write("MODL 1")           # modulation enable

def FM_modulation(sg, mod_rate, mod_dev):
    sg.write("TYPE 1")          # 1 = FM
    sg.write("MFNC 3")          # 3 = Sine
    sg.write(f"RATE {mod_rate}")  # Hz 
    sg.write(f"FDEV {mod_dev}")  # f_0 +- delta
    sg.write("MODL 1")           # modulation enable

# ----------------------------
# ODMR Mode
# ----------------------------

class ODMRMode(Enum):
    LSR = auto()
    AM  = auto()
    FM  = auto()

# ----------------------------
# Measurement
# ----------------------------


def measure_odmr(
    sg,
    demod,
    freqs: np.ndarray,
    dwell: float,
    n_iter: int = 6,
    phase_matched: bool = False
) -> np.ndarray:
    lia_signal = []
    for iter in tqdm (range(n_iter), desc="Averaging ODMR sweeps"):
        demod_vals = []
        for f in freqs:
            sg.write(f"FREQ {float(f)}")
            time.sleep(dwell)

            s = demod.sample()
            x = float(s["x"][0])
            y = float(s["y"][0])
            
            if phase_matched:
                demod_vals.append(x)
            else:
                demod_vals.append((x * x + y * y) ** 0.5)

        if iter == 0 :
            continue
        lia_signal.append(demod_vals)

    lia_signal = np.array(lia_signal, dtype=float)
    return np.mean(lia_signal, axis=0)

# ----------------------------
# Savd data & plot graph
# ----------------------------

def save_data(mode, n_magnet, n_iter, amp_dbm, freqs, lia_signal):
   
    mode_name = mode.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    date_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join("ODMR", date_str, mode_name)
    os.makedirs(save_dir, exist_ok=True)

    if mode_name == 'LSR':
        mod_freq = "Mf535"
    elif mode_name == 'AM':
        mod_freq = "Mf2k"
    elif mode_name == 'FM':
        mod_freq = "Mf5k_5M"
    else:
        raise ValueError(f"Unsupported mode: {mode_name}")

    fname = (
        f"odmr_{mode_name}_{timestamp}"
        f"_P{amp_dbm}dBm"
        f"_{mod_freq}"
        f"_B{n_magnet}"
        f"_iter{n_iter-1}.npz"
    )
    full_path = os.path.join(save_dir, fname)

    print("Saving to:", full_path)

    np.savez(
        full_path,
        freqs=freqs,
        signal_raw=lia_signal
    )


def plot_odmr(freqs: np.ndarray, R: np.ndarray):
    # freqs in Hz, R in Volts, do unit conversion when plotting
    # R is magnitude of phase-matched signal as from LabOne software
    plt.figure(figsize=(8, 5))
    plt.plot(freqs / 1e9, R * 1000, "-o", markersize=2)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Lock-in signal (mV)")
    plt.title("ODMR")
    plt.grid(True)
    plt.show()
    
    
# ----------------------------
# Main
# ----------------------------
def main():
    # ---- instruments parameters ----
    mfli_host = "192.168.91.174"
    mfli_dev = "dev5867"
    # sg_resource = "TCPIP::169.254.2.7::5025::SOCKET"

    # --------------------------------------------------------------------------------------------

    # ---- User parameters ----

    # --------------------------------------------------------------------------------------------

    # sweep frequency
    f_center = 2.88e9
    span = 0.1e9
    N = 101
    # bias field
    n_magnet=0
    # RF power
    amp_dbm = -10.0
    # num of averaging
    n_iter = 1 # used to be 20
    
        # Mode 
    # --------------------------------------------------------------------------------------------
    # LSR, AM, FM
    # --------------------------------------------------------------------------------------------

    # mode = ODMRMode.FM, ODMRMode.AM , ODMRMode.LSR
    # ^ choose one of the above, note: idk abt LSR, but it seems to be an option?

    #mode = ODMRMode.AM
    #AM modulation param
    mod_freq   = 20e3 ##2e3  # TODO: why is the comment saying 2e3 not 2e4?
    mod_depth  = 500.0  # 100 % # TODO: how is 500.0 corresponding to 100%?

    mode = ODMRMode.FM
    #FM modulation param
    mod_rate   = 2e3 #2e3  
    mod_dev  = 5.6e6

    # mode = ODMRMode.LSR
    #LSR modulation

    freqs, f_start, f_end = calc_freq_range(f_center, span, N)
    
    print(f"Frequency sweep: {f_start/1e9:.3f} -> {f_end/1e9:.3f} GHz (N={N})")

    # ---- Connect instruments ----
    _, demod = connect_mfli(mfli_host, mfli_dev, demod_index=0)
    sg = cs.connect_sg386(cs.sg_resource)

    # ---- Timing ----
    dwell, tau = compute_dwell_from_demod(demod, multiplier=5.0) # why choose mult=5?
    print("Dwell time:", dwell*1e3, "ms")

    # ---- Measure ----
    cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=True)
    phase_matched = True
    try:
        if mode == ODMRMode.AM:
            AM_modulation(sg, mod_freq, mod_depth)
        elif mode == ODMRMode.FM:
            FM_modulation(sg, mod_rate, mod_dev)
        elif mode == ODMRMode.LSR:
            phase_matched = False              
        else:
            raise ValueError(f"Unsupported ODMR mode: {mode}")
        lia_signal = measure_odmr(sg, demod, freqs, dwell=dwell, n_iter=n_iter, phase_matched=phase_matched)
    finally:
        sg.write("MODL 0") 
        cs.enable_sg386(sg, amp_dbm=amp_dbm, enable=False)
        

    # ---- Save data ----

    save_data(mode, n_magnet, n_iter, amp_dbm, freqs, lia_signal)
    print("measurement finished")

    plot_odmr(freqs, lia_signal)
    
if __name__ == "__main__":
    main()