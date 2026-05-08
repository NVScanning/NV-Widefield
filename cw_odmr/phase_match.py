import numpy as np
import time
import pyvisa
import matplotlib.pyplot as plt
from tqdm import tqdm
from zhinst.toolkit import Session
from scipy.optimize import curve_fit


#### MFLI connection ####
session = Session('192.168.91.174')  
mfli = session.connect_device('dev5867')
print("Connected to MFLI")

demod = mfli.demods[0]
demod.enable(1)

#### SG386 connection ####
rm = pyvisa.ResourceManager()

sg = rm.open_resource(
    "TCPIP::169.254.2.7::5025::SOCKET",
    write_termination="\n",
    read_termination="\n",
    timeout=5000,  
)
print("Connected to sg386")
         

# time.sleep(4)
sg.write("ENBR 1")       
sg.write("AMPR -12")


#AM modulation
# mod_freq   = 2e3  
# mod_depth  = 100.0  # 100 %

# sg.write("TYPE 0")          # 0 = AM
# sg.write("MFNC 3")          # 3 = Sine
# sg.write(f"RATE {mod_freq}")  # Hz 
# sg.write(f"ADEP {mod_depth}")  # 0–100 %
# sg.write("MODL 1")           # modulation enable

#FM modulation
mod_rate   = 5e3  
mod_dev  = 500e3

sg.write("TYPE 1")          # 1 = FM
sg.write("MFNC 3")          # 3 = Sine
sg.write(f"RATE {mod_rate}")  # Hz 
sg.write(f"FDEV {mod_dev}")  # f_0 +- delta
sg.write("MODL 1")           # modulation enable

iter = 5
f = 2.8e9

for i in tqdm(range(iter), desc="Averaging ODMR sweeps"):

    sg.write(f"FREQ {f}")   
    time.sleep(10)          


sg.write("MODL 0") 
sg.write("ENBR 0") 

