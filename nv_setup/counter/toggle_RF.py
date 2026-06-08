import time
import os
from datetime import datetime
from enum import Enum, auto
import numpy as np
import pyvisa

from .. import connection_setup as cs

"""
Toggles RF on for 20 seconds (I think) 

tbh I don't really get the pont of this, maybe it was unfinished
"""



def toggle_rf_at_resonance(sg, amp_dbm: float = -12.0, enable: bool = True):
    f = 2.87e9
    sg.write(f"FREQ {float(f)}")
    sg.write(f"AMPR {amp_dbm}")
    sg.write(f"ENBR {1 if enable else 0}")

def main():
    # ---- parameters ----
    # sg_resource = "TCPIP::169.254.2.7::5025::SOCKET"
    amp_dbm = -12.0

    sg = cs.connect_sg386(cs.sg_resource)
    time.sleep(3)
    toggle_rf_at_resonance(sg, amp_dbm=amp_dbm, enable=True)
    time.sleep(20)
    toggle_rf_at_resonance(sg, amp_dbm=amp_dbm, enable=False)
    print(sg.query("ENBR?\n"))

if __name__ == "__main__":
    main()