"""
Interactive sleep-quality model and bedtime recommendation.

Times are entered as HH:MM in 24-hour notation, e.g.
  22:30  for 10:30 pm
  07:00  for 7 am

If your desired wake time is on the next day (typical case),
the script will automatically treat it as "tomorrow"
if wake_time <= now_time.

Last caffeine can be entered as HH:MM or the word "none".
"""

import numpy as np

# -------------------------------
# Model parameters (tuned for reasonable behavior)
# -------------------------------

DEFAULT_PARAMS = dict(
    A=0.2,       # mean secretion
    B=0.7,       # circadian amplitude
    K=0.35,      # removal rate [1/h]
    phi=5.5,     # circadian phase [rad], peak during night
    theta=0.35,  # sleep propensity threshold
    k_s=6.0,     # logistic steepness
    lam1=0.6,    # penalty: awakenings
    lam2=0.2,    # penalty: latency
    lam3=0.3,    # bonus: REM alignment
    alpha=0.0,   # light coupling (unused here, L=0)
    c=0.0,       # stress coupling (unused here, C=0)
    T_target=7.5,  # ideal effective sleep duration [h]
    lam4=0.2,      # penalty weight for deviation from T_target
    beta_caff=0.6  # caffeine suppresses melatonin / sleep drive
)

OMEGA = 2.0 * np.pi / 24.0    # 24 h circadian frequency


# -------------------------------
# Helper: time parsing
# -------------------------------

def parse_time_hhmm(s):
    """
    Parse 'HH:MM' (24-hour) into float hours.
    Example: '22:30' -> 22.5
    """
    s = s.strip()
    parts = s.split(':')
    if len(parts) != 2:
        raise ValueError(f"Time '{s}' must be in HH:MM format.")
    hh = int(parts[0])
    mm = int(parts[1])
    if not (0 <= hh < 24) or not (0 <= mm < 60):
        raise ValueError(f"Time '{s}' out of range.")
    return hh + mm / 60.0


def hours_to_hhmm(t):
    """
    Convert float hours (can be >24) to (HH, MM) modulo 24.
    """
    t = t % 24.0
    hh = int(t)
    mm = int(round((t - hh) * 60.0))
    if mm == 60:
        hh = (hh + 1) % 24
        mm = 0
    return hh, mm


# -------------------------------
# ODE RHS and RK4 integrator
# -------------------------------

def melatonin_rhs(t, y, params, t_caff_last):
    """
    Right-hand side of melatonin ODE:
        dy/dt = -K*y + A + B cos(omega t + phi)
                - alpha L(t) - c C(t) - beta_caff U(t)

    Caffeine: U(t) = 1 for 6 h after last caffeine, else 0.
    Light L(t) and stress C(t) are set to 0 in this recommendation mode.
    """
    A = params["A"]
    B = params["B"]
    K = params["K"]
    phi = params["phi"]

    # Light and stress inputs (here fixed to 0)
    L = 0.0
    C = 0.0
    alpha = params.get("alpha", 0.0)
    c = params.get("c", 0.0)

    # Caffeine input: one rectangular pulse
    if t_caff_last is None:
        U = 0.0
    else:
        if t_caff_last <= t < t_caff_last + 6.0:
            U = 1.0
        else:
            U = 0.0

    beta = params.get("beta_caff", 0.6)

    # NOTE: caffeine now SUPPRESSES melatonin (minus sign!)
    forcing = A + B * np.cos(OMEGA * t + phi) - alpha * L - c * C - beta * U
    return -K * y + forcing


def rk4_step(f, t, y, dt, *args):
    """One step of classic 4th-order Runge–Kutta."""
    k1 = f(t,           y,               *args)
    k2 = f(t + 0.5*dt,  y + 0.5*dt*k1,   *args)
    k3 = f(t + 0.5*dt,  y + 0.5*dt*k2,   *args)
    k4 = f(t + dt,      y + dt*k3,       *args)
    return y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


# -------------------------------
# Sleep propensity and Q
# -------------------------------

def logistic(x, theta, k_s):
    """Logistic sleep-propensity function sigma(y - theta)."""
    return 1.0 / (1.0 + np.exp(-k_s * (x - theta)))


def compute_nightly_metrics(t, S, t_bed, t_wake, params):
    """
    Given time grid t and sleep propensity S(t) over [t_bed, t_wake],
    compute latency, number of awakenings, effective sleep time,
    REM-cycle bonus, and quality score Q.
    """
    dt = t[1] - t[0]

    # Sleep onset latency: first time S >= 0.5
    above_half = S >= 0.5
    if np.any(above_half):
        t_onset = t[above_half][0]
        latency = max(0.0, t_onset - t_bed)
    else:
        latency = t_wake - t_bed

    # Awakenings: transitions into low-propensity episodes (S < 0.3)
    awake = S < 0.3
    n_awake = int(np.sum(awake[1:] & (~awake[:-1])))

    # Effective sleep time: integral of S over [t_bed, t_wake]
    T_sleep = float(np.sum(S) * dt)

    # REM-cycle alignment: bonus if (t_wake - t_bed) is near multiple of 1.5 h
    T_window = t_wake - t_bed
    tau_REM = 1.5
    n_star = round(T_window / tau_REM)
    delta = abs(T_window - n_star * tau_REM)

    if delta <= 0.25:
        B_REM = 1.0
    elif delta <= 0.75:
        B_REM = 0.5
    else:
        B_REM = 0.0

    lam1 = params["lam1"]
    lam2 = params["lam2"]
    lam3 = params["lam3"]
    T_target = params.get("T_target", 7.5)
    lam4 = params.get("lam4", 0.2)

    # Penalize deviation from target sleep duration
    duration_penalty = (T_sleep - T_target) ** 2

    Q = (
        T_sleep
        - lam1 * n_awake
        - lam2 * latency
        + lam3 * B_REM
        - lam4 * duration_penalty
    )

    metrics = dict(
        latency=latency,
        n_awake=n_awake,
        T_sleep=T_sleep,
        B_REM=B_REM,
        duration_penalty=duration_penalty
    )
    return Q, metrics


# -------------------------------
# Nightly simulation
# -------------------------------

def simulate_night(t_bed, t_wake, t_caff_last, params=None,
                   dt=1.0/12.0, history_hours=10.0):
    """
    Simulate melatonin and sleep propensity around a single night.

    t_bed, t_wake, t_caff_last are all in hours (can exceed 24).
    """
    if params is None:
        params = DEFAULT_PARAMS

    t_start = max(0.0, t_bed - history_hours)
    t_end = t_wake

    n_steps = int(np.ceil((t_end - t_start) / dt))
    t = t_start + dt * np.arange(n_steps + 1)

    y = np.zeros_like(t)
    y[0] = 0.0  # initial melatonin

    for k in range(n_steps):
        y[k+1] = rk4_step(melatonin_rhs, t[k], y[k], dt, params, t_caff_last)

    # Restrict to [t_bed, t_wake]
    mask = (t >= t_bed) & (t <= t_wake)
    t_night = t[mask]
    y_night = y[mask]

    theta = params["theta"]
    k_s = params["k_s"]
    S = logistic(y_night, theta, k_s)

    Q, metrics = compute_nightly_metrics(t_night, S, t_bed, t_wake, params)
    return t_night, y_night, S, Q, metrics


# -------------------------------
# Bedtime recommendation
# -------------------------------

def recommend_bedtime(t_now, t_wake, t_caff_last,
                      params=None, dt=1.0/12.0):
    """
    Scan candidate bedtimes between "now + 15 minutes" and
    a window that yields between 5 and 9 hours in bed.
    Evaluate Q for each, and return the best.
    """
    if params is None:
        params = DEFAULT_PARAMS

    T_min = 5.0    # minimum time in bed [h]
    T_max = 9.0    # maximum time in bed [h]
    delta_min = 0.25   # must wait at least 15 minutes before bed
    h = 0.25   # step between candidate bedtimes [h]

    # Earliest allowed by "wait at least delta_min"
    t_start_constraint = t_now + delta_min
    # Latest you can go to bed and still get at least T_min hours
    t_end_constraint_min = t_wake - T_min
    # Earliest you can go to bed and not exceed T_max hours in bed
    t_start_max = t_wake - T_max

    t_start = max(t_start_constraint, t_start_max)
    t_end = t_end_constraint_min

    if t_start >= t_end:
        raise ValueError(
            "No feasible bedtime window: wake time too early or constraints too tight.")

    candidate_bed = np.arange(t_start, t_end + 1e-9, h)

    best_Q = -1e9
    best_tbed = None
    best_metrics = None

    for t_bed in candidate_bed:
        _, _, _, Q, metrics = simulate_night(
            t_bed, t_wake, t_caff_last, params, dt=dt
        )
        if Q > best_Q:
            best_Q = Q
            best_tbed = float(t_bed)
            best_metrics = metrics

    return best_tbed, best_Q, best_metrics


# -------------------------------
# Interactive CLI
# -------------------------------

if __name__ == "__main__":
    print("=== High-Quality Sleep Bedtime Recommender ===")
    print("Enter times in 24-hour format, e.g. 22:30 or 07:00\n")

    # Current time
    now_str = input("What time is it now? (HH:MM) : ")
    t_now = parse_time_hhmm(now_str)

    # Desired wake time
    wake_str = input("What time do you want to wake up? (HH:MM) : ")
    t_wake = parse_time_hhmm(wake_str)

    # If wake time is earlier than now, assume it's tomorrow
    if t_wake <= t_now:
        t_wake += 24.0

    # Last caffeine
    caff_str = input(
        "When was your last caffeine? (HH:MM or 'none') : ").strip().lower()
    if caff_str in ("none", ""):
        t_caff_last = None
    else:
        t_caff_last = parse_time_hhmm(caff_str)
        # If caffeine time is "in the future" relative to now (weird input),
        # assume it was actually yesterday:
        if t_caff_last > t_now:
            t_caff_last -= 24.0

    best_tbed, best_Q, metrics = recommend_bedtime(
        t_now, t_wake, t_caff_last, params=DEFAULT_PARAMS
    )

    hh, mm = hours_to_hhmm(best_tbed)

    print("\nRecommended bedtime:")
    print("  -> {:02d}:{:02d}".format(hh, mm))
    print("Predicted quality score Q: {:.3f}".format(best_Q))
    print("Details:")
    print("  Effective sleep time       {:.2f} h".format(metrics["T_sleep"]))
    print("  Sleep onset latency        {:.2f} h".format(metrics["latency"]))
    print("  Number of awakenings       {:d}".format(metrics["n_awake"]))
    print("  REM-cycle alignment bonus  {:.2f}".format(metrics["B_REM"]))
    print("  Duration penalty term      {:.2f}".format(
        metrics["duration_penalty"]))
