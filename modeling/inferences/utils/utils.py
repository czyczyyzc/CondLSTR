import random
import colorsys
import numpy as np


def obtain_colors(N, shuffle=False, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if shuffle:
        random.shuffle(colors)
    colors = (np.asarray(colors) * 255).astype(np.uint8)
    return colors
