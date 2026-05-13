import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

"""
this file is to read npy files created by cw_odmr

note: measurements previous to 2026-05-13 at 15:02 only have cps data, not frequency stored
^ this means you don't have any x data, and filetype is different
"""



date = "2026-05-13"
time = "15-21-26"
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
directory = os.path.join(project_root, "NVCFM_Data", date, "cw_odmr_" + time)
# directory = "../NVCFM_Data" + "/" + date + "/cw_odmr" + time

data = np.load(directory + ".npz") # npy for old, npz for new measurements

plt.figure(figsize=(8, 5))
plt.plot(data["x"]/10**9,data["y"], "-o", markersize=2) # uncomment for measurements after 2026-05-13 at 15 hours
plt.xlabel("Frequency (GHz)") # uncomment for measurements after 2026-05-13 at 15 hours
# plt.plot(data, "-o", markersize=2) # uncomment for measurements previous to 2026-05-13 at 15 hours
# plt.xlabel("index (frequency isn't saved") # uncomment for measurements previous to 2026-05-13 at 15 hours
plt.ylabel("kcps")
plt.title("ODMR")
plt.grid(True)
plt.show()
