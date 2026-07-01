import time


# other files import this class, then cal Log.log("txt")
# This will print to terminal along with a timestamp only if log.enable=True


t0 = 0
enable = False

def log(string):
    if enable:
        print(f"t={time.time()-t0:.3f}s: {string}")


def start():
    global t0
    global enable
    t0 = time.time()
    enable = True