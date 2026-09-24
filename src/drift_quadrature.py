"""Peak-aware log-domain quadrature for the fixed p=2, h=.05, V=1+x² study.

Numerical diagnostics, not certified interval arithmetic or a convergence proof.
The MALA acceptance correction is imported unchanged from diagnostics.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp
from src.diagnostics import delta
from src.lyapunov import dU

P, H = 2, 0.05
LOG_NORM = np.log(H * np.sqrt(2 * np.pi))


def proposal_mean(x):
    return x - 0.5 * H**2 * dU(x, P)


def log_proposal(x, y):
    y = np.asarray(y, dtype=np.longdouble)
    return -0.5 * ((y - proposal_mean(np.longdouble(x))) / H)**2 - LOG_NORM


def log_acceptance(x, y):
    return np.minimum(-delta(np.longdouble(x), np.asarray(y, dtype=np.longdouble), H, P), 0)


def log_integrand(x, y, magnitude=False):
    y = np.asarray(y, dtype=np.longdouble)
    value = log_proposal(x, y) + log_acceptance(x, y)
    if magnitude:
        with np.errstate(divide='ignore'):
            value = value + np.log(np.abs((y - x) * (y + x)))
    return value


def _real_roots(poly):
    return [float(z.real) for z in poly.roots()
            if abs(z.imag) < 1e-6 * (1 + abs(z.real))]


def split_points(x, limit, magnitude=False):
    """All candidate switching and stationary points of the two log branches.

    q alpha = min(q(x,y), exp(x^4-y^4) q(y,x)). For magnitudes,
    stationarity is b'(y)(y²-x²)+2y=0 on each log branch b.
    Extra roots from the inactive branch are harmless subdivisions.
    """
    z = Poly([0, 1])
    mu = proposal_mean(x)
    forward = -(z - mu)**2 / (2 * H**2) - LOG_NORM
    reverse = x**4 - z**4 - (x - z + 2 * H**2 * z**3)**2 / (2 * H**2) - LOG_NORM
    points = [-limit, limit, -abs(x), abs(x), mu, 0.0]
    points += _real_roots(forward - reverse)
    for branch in (forward, reverse):
        derivative = branch.deriv()
        points += _real_roots(derivative * (z*z-x*x) + 2*z if magnitude else derivative)
    anchors = [-limit, limit, -abs(x), abs(x), mu, 0.0]
    # Keep radial boundaries exact even if a polynomial root is very close.
    roots = [v for v in points if all(abs(v-a) > 1e-9 for a in anchors)]
    points = sorted(v for v in roots + anchors if -limit <= v <= limit)
    unique = []
    for v in points:
        if not unique or v - unique[-1] > 1e-9:
            unique.append(v)
    return np.array(unique)


@dataclass
class LogIntegral:
    log_value: float
    log_quad_error: float
    log_omitted_bound: float
    peaks: tuple
    warnings: tuple

    @property
    def relative_error_indicator(self):
        """Quadrature estimate plus omitted-region bound, divided by value."""
        return float(np.exp(np.logaddexp(self.log_quad_error, self.log_omitted_bound) - self.log_value))


def _logsum(values):
    return float(logsumexp(values)) if values else -np.inf


def integrate_component(x, component, *, rtol=1e-8, padding=4.0,
                        drop=50.0, extra_splits=False):
    """Integrate acceptance, inward or outward nonnegative drift magnitude.

    Retain each smooth piece within `drop` log units of its maximum;
    bound (rather than silently zero) the omitted area. Integrate scaled
    weights on [0,1]. Values and errors remain logarithms throughout.
    """
    if x <= 0 or component not in ('acceptance', 'inward', 'outward'):
        raise ValueError('Use x>0 and acceptance/inward/outward.')
    if rtol <= 0 or padding <= 0 or drop <= 0:
        raise ValueError('rtol, padding and drop must be positive.')
    magnitude = component != 'acceptance'
    limit = max(abs(x), abs(proposal_mean(x))) + padding
    points = split_points(x, limit, magnitude)
    if extra_splits:
        points = np.sort(np.r_[points, (points[:-1] + points[1:])/2])
    logs, errors, omitted, peaks, notes = [], [], [], [], []
    f = lambda y: float(log_integrand(x, y, magnitude))
    for left, right in zip(points[:-1], points[1:]):
        mid = (left + right)/2
        if component == 'inward' and abs(mid) >= x:
            continue
        if component == 'outward' and abs(mid) <= x:
            continue
        optimum = minimize_scalar(lambda y: -f(y), bounds=(left, right),
                                  method='bounded', options={'xatol': 1e-13})
        candidates = [left, right, float(optimum.x), mid]
        peak = max(candidates, key=f)
        shift = f(peak)
        peaks.append((peak, shift))
        threshold = shift - drop
        lo = brentq(lambda y: f(y)-threshold, left, peak, xtol=1e-13) if f(left) < threshold and peak > left else left
        hi = brentq(lambda y: f(y)-threshold, peak, right, xtol=1e-13) if f(right) < threshold and peak < right else right
        missing_width = (lo-left) + (right-hi)
        if missing_width > 0:
            omitted.append(np.log(missing_width) + threshold)
        width = hi-lo
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', IntegrationWarning)
            value, err = quad(lambda t: np.exp(f(lo+width*t)-shift), 0, 1,
                              epsabs=rtol*0.01, epsrel=rtol, limit=200)
        notes.extend(str(w.message) for w in caught)
        if value <= 0 or not np.isfinite(value):
            raise ArithmeticError('Scaled integral unresolved; do not report zero.')
        logs.append(shift + np.log(width) + np.log(value))
        if err > 0:
            errors.append(shift + np.log(width) + np.log(err))
    # Outside [-L,L], q alpha <= exp(x^4-y^4)/(h sqrt(2pi)).
    # Convexity gives (L+t)^4 >= L^4+4 L^3 t; integrate the
    # resulting exponential envelope, with y²+x² for drift magnitudes.
    if component != 'inward':
        rate = 4*limit**3
        factor = ((limit**2+x*x)/rate + 2*limit/rate**2 + 2/rate**3) if magnitude else 1/rate
        omitted.append(np.log(2) - LOG_NORM + x**4-limit**4 + np.log(factor))
    return LogIntegral(_logsum(logs), _logsum(errors), _logsum(omitted),
                       tuple(peaks), tuple(notes))


def integrate_drift(x, **kwargs):
    result = {name: integrate_component(x, name, **kwargs)
              for name in ('acceptance', 'inward', 'outward')}
    a, b = result['outward'].log_value, result['inward'].log_value
    if a == b:
        sign, log_abs = 0, -np.inf
    else:
        sign = 1 if a > b else -1
        log_abs = max(a, b) + np.log(-np.expm1(-abs(a-b)))
    log_error = _logsum([v for key in ('inward', 'outward')
                         for v in (result[key].log_quad_error, result[key].log_omitted_bound)])
    result.update(sign=sign, log_abs_drift=log_abs, log_drift_error=log_error,
                  sign_resolved=bool(sign and log_abs > log_error))
    return result


def format_log(log_value, sign=1, digits=4):
    """Never format a finite logarithm as zero through exponent underflow."""
    if sign == 0:
        return 'unresolved'
    if not np.isfinite(log_value):
        return 'unresolved (no finite logarithm)'
    exponent = int(np.floor(log_value / np.log(10)))
    mantissa = np.exp(log_value - exponent*np.log(10))
    if round(mantissa, digits-1) >= 10:
        mantissa /= 10
        exponent += 1
    return f'{"-" if sign < 0 else ""}{mantissa:.{digits-1}f}e{exponent:+d}'
