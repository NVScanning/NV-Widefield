import os
from datetime import datetime


date_str = '2026_0129'
mode ='AM'
save_dir = os.path.join("ODMR", date_str, mode)

print(save_dir)
# TODO: why tf is it a fixed date if we get the timestamp in the literal next line????

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
print(timestamp)

n_magnet = 5
n_iter = 6
amp_dbm = 12

if mode == 'AM':
    param = "Mf2k"
elif mode == 'FM':
    param = "Mf5k_5M"
elif mode == 'LSR':
    param = "Mf535"
else:
    raise ValueError(f"Unsupported mode: {mode}")




fname = (
    f"odmr_{mode}_{timestamp}"
    f"_P{amp_dbm}dBm"
    f"_{param}"
    f"_B{n_magnet}"
    f"_iter{n_iter-1}.npz"
)

print(fname)