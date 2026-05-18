REGION_NAMES = ("A1", "A2", "A3")


def classify_regions(x, y, eps):
    if x <= 0:
        raise ValueError("Region diagnostics require x > 0.")

    left = (1.0 - eps) * x
    right = (1.0 + eps) * x

    return {
        "A1": y <= left,
        "A2": (y > left) & (y < right),
        "A3": y >= right,
    }
