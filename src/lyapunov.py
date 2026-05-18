import numpy as np

from src.utils import MAX_LOG_VALUE, subtract_from_logs


def U(x, p):
    return np.abs(x) ** (2 * p)


def dU(x, p):
    return 2 * p * x * np.abs(x) ** (2 * p - 2)


def V_poly(x, m):
    return 1.0 + np.abs(x) ** m


def V(x, m):
    return V_poly(x, m)


def log_V_exp(x, s, beta):
    return s * np.abs(x) ** beta


def V_exp(x, s, beta):
    return np.exp(np.clip(log_V_exp(x, s, beta), a_min=None, a_max=MAX_LOG_VALUE))


def log_V_U(x, s, q, p):
    return s * U(x, p) ** q


def V_U(x, s, q, p):
    return np.exp(np.clip(log_V_U(x, s, q, p), a_min=None, a_max=MAX_LOG_VALUE))


def build_lyapunov(lyapunov_type, p, m=2.0, s=0.1, beta=1.0, q=1.0):
    if lyapunov_type == "poly":
        return {
            "name": "poly",
            "label": f"V(x)=1+|x|^{m}",
            "V_func": lambda x: V_poly(x, m),
            "log_V_func": None,
            "params": {"m": m},
        }

    if lyapunov_type == "exp":
        return {
            "name": "exp",
            "label": f"V(x)=exp({s}|x|^{beta})",
            "V_func": lambda x: V_exp(x, s, beta),
            "log_V_func": lambda x: log_V_exp(x, s, beta),
            "params": {"s": s, "beta": beta},
        }

    if lyapunov_type == "U":
        return {
            "name": "U",
            "label": f"V(x)=exp({s} U(x)^{q})",
            "V_func": lambda x: V_U(x, s, q, p),
            "log_V_func": lambda x: log_V_U(x, s, q, p),
            "params": {"s": s, "q": q, "p": p},
        }

    raise ValueError(f"Unknown Lyapunov type: {lyapunov_type}")


def evaluate_lyapunov_difference(y, x, V_func, log_V_func=None):
    if log_V_func is not None:
        return subtract_from_logs(log_V_func(y), log_V_func(x))
    return V_func(y) - V_func(x)
