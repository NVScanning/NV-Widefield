
import pco
import matplotlib.pyplot as plt

"""
A step in the direction of widefield imaging, by using the camera sensor, but binning
all the relevant pixels together into one "brightness" signal, to use in place of the
SPCM used in cw_odmr.py
"""

# TODO:
#   Read an image from the camera
#   Select only the relevant pixels (maybe a square defined in terms of indices)
#   Average the light value from all the pixels (could potentially also make use of uncertainty here)
#   Do all of the above in a loop iterating over the frequencies
#   Use the counts to plot/save/analyze the odmr (same as with SPCM data)

print("widefield binned called")


with pco.Camera() as cam:
    cam.record(mode="sequence")
    image, meta = cam.image()

    plt.imshow(image, cmap='gray')
    plt.show()
    print("Picture taken")

