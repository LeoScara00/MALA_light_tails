import numpy as np

REGION_NAMES = ("A1", "A2", "A3") # radial partition of the state-space 


def classify_regions(x, y, eps):
    
    r_x = np.abs(x)
    r_y = np.abs(y)

    left = (1.0 - eps) * r_x
    right = (1.0 + eps) * r_x

    return {
        "A1": r_y <= left,
        "A2": (r_y > left) & (r_y < right),
        "A3": r_y >= right,
    }
