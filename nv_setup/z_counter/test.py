
import sys
import os

sys.path.append(os.path.abspath(".."))

import APT.thorlabs_apt as apt


devs = apt.list_available_devices()
print("Devices:", devs)