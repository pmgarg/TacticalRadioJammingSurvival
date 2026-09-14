"""Fixed-size primitives with no dynamic allocation — these map 1:1 to the C build."""
from __future__ import annotations
import math


class Ring:
    """Fixed-capacity circular buffer of floats."""
    __slots__ = ("buf", "cap", "n", "head")

    def __init__(self, cap: int):
        self.cap = cap
        self.buf = [0.0] * cap
        self.n = 0
        self.head = 0

    def push(self, v: float) -> None:
        self.buf[self.head] = v
        self.head = (self.head + 1) % self.cap
        if self.n < self.cap:
            self.n += 1

    def values(self) -> list[float]:
        if self.n < self.cap:
            return self.buf[:self.n]
        return self.buf[self.head:] + self.buf[:self.head]

    def last(self, default: float = 0.0) -> float:
        if self.n == 0:
            return default
        return self.buf[(self.head - 1) % self.cap]

    def mean(self, default: float = 0.0) -> float:
        if self.n == 0:
            return default
        return sum(self.values()) / self.n

    def std(self, default: float = 0.0) -> float:
        if self.n < 2:
            return default
        v = self.values()
        m = sum(v) / len(v)
        return math.sqrt(max(0.0, sum((x - m) ** 2 for x in v) / (len(v) - 1)))

    def min(self, default: float = 0.0) -> float:
        return min(self.values()) if self.n else default

    def max(self, default: float = 0.0) -> float:
        return max(self.values()) if self.n else default

    def slope_per_s(self, dt: float) -> float:
        """Least-squares slope in units/second."""
        v = self.values()
        k = len(v)
        if k < 3 or dt <= 0:
            return 0.0
        xm = (k - 1) / 2.0
        ym = sum(v) / k
        num = sum((i - xm) * (v[i] - ym) for i in range(k))
        den = sum((i - xm) ** 2 for i in range(k))
        if den == 0:
            return 0.0
        return (num / den) / dt


class Ewma:
    __slots__ = ("tau", "v", "init")

    def __init__(self, tau: float):
        self.tau = tau
        self.v = 0.0
        self.init = False

    def update(self, x: float, dt: float) -> float:
        if not self.init:
            self.v, self.init = x, True
            return self.v
        a = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
        self.v += a * (x - self.v)
        return self.v

    @property
    def value(self) -> float:
        return self.v


def pearson(xs: list[float], ys: list[float]) -> float:
    """Correlation coefficient; 0.0 when either side is degenerate."""
    k = min(len(xs), len(ys))
    if k < 4:
        return 0.0
    xs, ys = xs[-k:], ys[-k:]
    mx, my = sum(xs) / k, sum(ys) / k
    sxy = sxx = syy = 0.0
    for i in range(k):
        dx, dy = xs[i] - mx, ys[i] - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    if sxx <= 1e-12 or syy <= 1e-12:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


class Cusum:
    """Two-sided CUSUM on a downward shift (the anomaly gate, design §7.2)."""
    __slots__ = ("k", "h", "s", "fired")

    def __init__(self, k: float, h: float):
        self.k, self.h, self.s, self.fired = k, h, 0.0, False

    def update(self, x: float, baseline: float) -> bool:
        self.s = max(0.0, self.s + (baseline - x - self.k))
        self.fired = self.s > self.h
        return self.fired

    def reset(self) -> None:
        self.s, self.fired = 0.0, False


class RunLength:
    """Tracks below-threshold run lengths -> fade-duration statistics (S7)."""
    __slots__ = ("thresh", "cur", "runs", "cap")

    def __init__(self, thresh: float, cap: int = 32):
        self.thresh, self.cur, self.runs, self.cap = thresh, 0, [], cap

    def update(self, x: float, dt: float) -> None:
        if x < self.thresh:
            self.cur += 1
        elif self.cur > 0:
            self.runs.append(self.cur * dt)
            if len(self.runs) > self.cap:
                self.runs.pop(0)
            self.cur = 0

    def mean(self) -> float:
        return sum(self.runs) / len(self.runs) if self.runs else 0.0

    def p95(self) -> float:
        if not self.runs:
            return 0.0
        s = sorted(self.runs)
        return s[min(len(s) - 1, int(0.95 * len(s)))]

    def duty(self, total_ticks: int) -> float:
        if total_ticks <= 0:
            return 0.0
        return min(1.0, (sum(self.runs) * 10.0 + self.cur) / total_ticks)
