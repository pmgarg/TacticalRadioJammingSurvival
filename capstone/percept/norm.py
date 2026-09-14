"""
Feature normalisation constants — FROZEN.

These are hand-set, not fitted to the training set, and they MUST be identical in
ns-3, in the refsim, and in the ESP32 firmware. If these drift between call sites
the model trains on one distribution and deploys on another (design §9.5).

Mirrored by percept/norm.h for the C build.
"""
N_FEATURES = 56

# (offset, scale) -> normalised = clamp((raw - offset) / scale, -1, 1)
RSSI_OFFSET, RSSI_SCALE = -70.0, 30.0        # dBm
NOISE_OFFSET, NOISE_SCALE = -90.0, 25.0      # dBm
SINR_OFFSET, SINR_SCALE = 15.0, 25.0         # dB
DBM_DELTA_SCALE = 20.0                       # dB deltas
RATE_SCALE = 50.0                            # frames/s
SPEED_SCALE = 20.0                           # m/s
DIST_SCALE = 100.0                           # m
TIME_SCALE = 30.0                            # s (log-scaled elsewhere)
SLOPE_SCALE = 10.0                           # dBm/s

# Percept / decision timing
PERCEPT_HZ = 10.0
WINDOW_T = 16                                # decision window (1.6 s at 10 Hz)
TAU_FAST = 0.5                               # s
TAU_SLOW = 5.0                               # s
TAU_BASE = 60.0                              # s (nominal baseline)
CORR_WINDOW_S = 5.0                          # S1 / S8 / S10 correlation window
TX_SHADOW_TAU_MS = 3.0                       # S4' reactive-jammer reaction window
DEFER_SCALE_MS = 2.0                        # tx_defer_time normalisation
TX_COND_TAU_MS = 2.0                         # S4 "recently transmitted" window

# Detection / classification thresholds
FADE_PDR_THRESH = 0.5                        # below this counts as "in a fade" (S7)
CUSUM_K = 0.02
CUSUM_H = 0.35
JAM_ENERGY_DBM = -80.0


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def nz(x: float, offset: float, scale: float) -> float:
    """Normalise-and-clamp."""
    return clamp((x - offset) / scale)


def unit(x: float) -> float:
    """Already 0..1 -> map to -1..1."""
    return clamp(2.0 * x - 1.0)
