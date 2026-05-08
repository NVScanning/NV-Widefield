import os
import numpy as np
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qm.octave import *

u = unit(coerce_to_integer=True)

#######################
# QOP CONNECTION      #
#######################
qop_ip = "192.168.88.250"
qop_port = 80

#######################
# AUXILIARY FUNCTIONS #
#######################
def square_pulse(amplitude, length):
    return [float(amplitude)] * int(length)

def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2 - 1, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]


#############
# VARIABLES #
#############

### Pulses lengths
readout_len = 5000 * u.ns  
long_readout_len = 5 * u.ms

src_pulse_len = 80 * u.ns   
src_amp = 0.3        # output amplitude

### Readout parameters
signal_threshold = -5_00  # ADC untis, to convert to volts divide by 4096 (12 bit ADC)

### Delays
detection_delay = 80 * u.ns
laser_delay = 0 * u.ns
src_delay = 0 * u.ns

#############################################
#                  Config                   #
#############################################

config = {
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
            },
            "digital_outputs": {
                1: {},  # SPCM
                2: {},  # Photon Src
            },
            "analog_inputs": {
                1: {"offset": 0.0},  # SPCM 
            },
        }
    },

    "elements": {

        # -------------------------
        # SPCM element 
        # -------------------------
        "SPCM": {
            "singleInput": {"port": ("con1", 1)},  # not used
            "digitalInputs": {  # for visualization in simulation
                "marker": {
                    "port": ("con1", 1),
                    "delay": detection_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "timeTaggingParameters": {
                "signalThreshold": signal_threshold,  # ADC units
                "signalPolarity": "Below",
                "derivativeThreshold": 1_023,
                "derivativePolarity": "Below",
            },
            "time_of_flight": detection_delay,
            "smearing": 0,
        },

        # -------------------------
        # Simulation-only photon source
        # -------------------------
        "photon_source": {
            # "digitalInputs": {
            #     "marker": {
            #         "port": ("con1", 2),
            #         "delay": src_delay,
            #         "buffer": 0,
            #     },
            # },
            "singleInput": {"port": ("con1", 1)},
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
            "digital_marker": "ON",
            "waveforms": {"single": "zero_wf"},
        },

        "long_readout_pulse": {
            "operation": "measurement",
            "length": long_readout_len,
            "digital_marker": "ON",
            "waveforms": {"single": "zero_wf"},
        },

        # Simulation source pulse
        "src_pulse": {
            "operation": "control",
            "length": src_pulse_len,
            # "digital_marker": "ON",
            "waveforms": {"single": "src_wf"},
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "src_wf": {"type": "constant", "sample": src_amp},
        "gauss_wf": {"type": "arbitrary", "samples": gauss(0.2, 0, 10, src_pulse_len)},
        "square_wf": {"type": "arbitrary", "samples": square_pulse(src_amp, src_pulse_len)},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},  # [(on/off, ns)]
        "OFF": {"samples": [(0, 0)]},  # [(on/off, ns)]
    },
}

