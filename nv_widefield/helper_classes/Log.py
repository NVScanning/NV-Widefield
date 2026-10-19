import time


# other files import this class, then cal Log.log("txt")
# This will print to terminal along with a timestamp only if log.enable=True

"""
Used for debugging with timestamped printouts, call Log.start() to get the logs to print
"""

t0 = 0
enable = False # True to printout all the timestamped Log.log(str)

def log(string):
    if enable:
        print(f"t={time.time()-t0:.3f}s: {string}")


def start():
    global t0
    global enable
    t0 = time.time()
    enable = True