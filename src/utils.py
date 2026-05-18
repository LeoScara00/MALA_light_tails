import numpy as np


MAX_LOG_VALUE = 700.0


def subtract_from_logs(log_a, log_b):
    log_a, log_b = np.broadcast_arrays(
        np.asarray(log_a, dtype=float),
        np.asarray(log_b, dtype=float),
    )
    result = np.empty(log_a.shape, dtype=float)
    mask = log_a >= log_b

    result[mask] = np.exp(
        np.clip(log_a[mask], a_min=None, a_max=MAX_LOG_VALUE)
    ) * (-np.expm1(log_b[mask] - log_a[mask]))
    result[~mask] = -np.exp(
        np.clip(log_b[~mask], a_min=None, a_max=MAX_LOG_VALUE)
    ) * (-np.expm1(log_a[~mask] - log_b[~mask]))
    return result


def mean_and_se(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    if values.size <= 1:
        return mean, 0.0
    se = values.std(ddof=1) / np.sqrt(values.size)
    return mean, se


def parse_x_grid(text):
    if ":" in text:
        parts = [part.strip() for part in text.split(":") if part.strip()]
        if len(parts) != 3:
            raise ValueError(
                "Range-style x-grid must have the form 'start:stop:num', for example '0.5:6:30'."
            )
        start, stop, num = float(parts[0]), float(parts[1]), int(float(parts[2]))
        return np.linspace(start, stop, num)

    parts = [part.strip() for part in text.split(",") if part.strip()]
    return np.array([float(part) for part in parts], dtype=float)
