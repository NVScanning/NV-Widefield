import numpy as np
from qualang_tools.units import unit

u = unit()

# QOP connection 
qop_ip = "192.168.88.250"
qop_port = 80

# -------------------------
# Helper waveform
# -------------------------
def square_pulse(amplitude, length):
    return [float(amplitude)] * int(length)

def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2 - 1, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]


# -------------------------
# SPCM time-tagging parameters 
# -------------------------
signal_threshold = 0.2   
signal_polarity = "ABOVE" 
time_of_flight = 80
smearing = 0

readout_len = int(1 * u.us)  # 50 us
src_pulse_len = 40   # ns  (10 cycles)
src_amp = 0.3        # output amplitude


config = {
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
            },

            # SPCM signal 
            "analog_inputs": {
                1: {"offset": 0.0},
            },
        }
    },

    "elements": {

        # -------------------------
        # SPCM element 
        # -------------------------
        "SPCM": {
            "singleInput": {"port": ("con1", 2)},  # not used
            "outputs": {"out1": ("con1", 1)},       # analog in 1
            "operations": {
                "readout": "readout_pulse",
            },
            "timeTaggingParameters": {
                "signalThreshold": signal_threshold,
                "signalPolarity": signal_polarity,
                "derivativeThreshold": 1023,
                "derivativePolarity": "BELOW",
            },
            "time_of_flight": time_of_flight,
            "smearing": smearing,
        },

        # -------------------------
        # Simulation-only photon source
        # -------------------------
        "photon_source": {
            "singleInput": {"port": ("con1", 1)},   # analog out 1
            "intermediate_frequency": 0,
            "operations": {
                "pulse": "src_pulse",
            },
        },


    },

    "pulses": {
        # Measurement pulse 
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"single": "zero_wf"},
        },

        # Simulation source pulse (square pulse robust for threshold crossing)
        "src_pulse": {
            "operation": "control",
            "length": src_pulse_len,
            "waveforms": {"single": "src_wf"},
        },
    },

    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "src_wf": {"type": "constant", "sample": src_amp},
        "gauss_wf": {"type": "arbitrary", "samples": gauss(0.2, 0, 10, src_pulse_len)},
        "square_wf": {"type": "arbitrary", "samples": square_pulse(src_amp, src_pulse_len),
},

    },
}

