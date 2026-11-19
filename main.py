"""gta6-trophy-rarity-forecast-01 entry in Python 3.13 (standard library only).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles on the logit scale.

Model (computed at run time from the case file alone; no network, no fitted
constants smuggled in from outside data):

  A trophy's earn rate is earners / players_tracked. The two move for
  different reasons -- new players dilute every rate, while the players
  already tracked keep earning -- so the program forecasts them separately
  and divides:

  * players(t): log players is linear in log(1 + hours since launch),
    fitted by weighted least squares on the case's reads. That is a power
    law in time, which is how a tracked-player count grows after a launch:
    fast at first, then slower.
  * earners_i(t): for each trophy, log earners is linear in the same
    log-time, fitted the same way. A trophy's slope is shrunk toward its
    tier's median slope, because a single trophy's reads are steppy (Exophase
    refreshes in jumps, and a 2-decimal rate quantises a small earner count)
    while trophies of one tier are earned on a similar schedule.
  * rate_i(target) = 100 * earners_i(target) / players(target), clamped.

  The median is that rate. The spread is the honest predictive uncertainty
  on the logit scale: the fit's prediction variance, plus a jump term for
  the refreshes the rules describe (launch night's first refresh moved
  every rate at once, players 638 -> 3,868, so a read can jump ~1 logit
  unit with no warning), plus a model-risk term that grows with the
  horizon. The quantiles are median +/- z_tau * spread, so a trophy whose
  trend is unsure gets a wide band and one whose history is clean gets a
  narrow one.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so a module shipped beside
main.py imports only after the sys.path line below.

What it does:
  - read every <id>.in in the input directory (the grader runs one case at
    a time, with only that case's file there): JSON
    {"schema": "makefun/gta6-trophies-case/v1", "id", "at",
    "horizon_minutes", "target", "levels", "game", "trophies", "history",
    "notes"}, where `trophies` lists the case's trophies {id, name,
    description, tier, points, secret} and `history` every read before
    `at`, oldest first, each {slot, at, players_tracked, trophies_earned,
    rates}, `rates` in the order of `trophies` (null where that read
    lacked the trophy);
  - write <id>.out to the output directory: {"trophies": [{"id": 1, "q":
    [q05, q25, q50, q75, q95]}, ...]}, one row for each of the case's
    trophies, each id once, no other key, each q five finite numbers from
    0 to 100 (percent), non-decreasing; at most 65,536 bytes. It never
    crashes on a case: a trophy it cannot fit falls back to persistence
    with a wide spread, and a case it cannot read at all still gets a
    valid file, because an invalid case ranks below every all-valid entry
    and takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch-day reference for the log-time axis: 2026-11-19T00:00:00Z, the same
# origin the house curve baseline fits against.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# A refresh can move a rate with no warning (launch night's first refresh
# took every rate down at once as the tracked players jumped 638 -> 3,868),
# so no spread may be narrower than this, in logit units.
JUMP = 0.6
# Model risk grows with the horizon: a trend fitted on days of history and
# read days ahead can bend. Per square-root day, in logit units.
RISK_PER_SQRT_DAY = 0.25
# Hard floor and cap on the spread, in logit units.
SPREAD_FLOOR = 0.6
SPREAD_CAP = 2.5
# Prior strength, in reads, when a trophy's slope is shrunk toward its
# tier's median slope.
SHRINK_READS = 6.0
# A trophy needs at least this many informative reads for its own slope to
# count at all; with fewer it takes its tier's slope.
MIN_READS_FOR_SLOPE = 3


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since_launch(at: str) -> float:
    return max(0.0, (parse_time(at) - LAUNCH).total_seconds() / 3600.0)


def log_time(at: str) -> float:
    """log(1 + hours since launch): the axis every trend is fitted on."""
    return math.log1p(hours_since_launch(at))


def wls(xs, ys, ws):
    """Weighted least squares for y = a + b*x.

    Returns (a, b, resid_var, sxx): intercept, slope, weighted residual
    variance (0 when the fit is exact or degenerate) and the weighted
    sum of squared x deviations, which scales the slope's variance.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    sw = sum(ws)
    if sw <= 0:
        return sum(ys) / n, 0.0, 0.0, 0.0
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid = sum(w * (y - ybar) ** 2 for w, y in zip(ws, ys)) / sw
        return ybar, 0.0, resid, 0.0
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    if n > 2:
        resid *= n / (n - 2)
    return a, b, resid, sxx


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def forecast_case(case: dict, name: str) -> dict:
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = log_time(case["target"])

    # Recency weights: the latest read counts twice the earliest. A launch
    # week's dynamics drift, so the recent past says more about the target
    # than the first night does.
    n_reads = len(history)
    recency = [0.5 + 0.5 * (j + 1) / n_reads for j in range(n_reads)]
    xs = [log_time(r["at"]) for r in history]
    players = [r["players_tracked"] for r in history]
    last_players = players[-1] if players[-1] else players[0] or 1

    # --- players(t): log players linear in log-time. ---
    lp = [math.log(max(1.0, float(p))) for p in players]
    pc0, pc1, presid, psxx = wls(xs, lp, recency)
    xbar_p = sum(w * x for w, x in zip(recency, xs)) / sum(recency)
    # Clamp the exponent before exp(): the fit's slope can be wild on a
    # short history, and exp() of it would raise instead of forecasting.
    p_exp = min(max(pc0 + pc1 * x_target, math.log(max(1.0, float(last_players)))),
                math.log(50.0 * float(last_players)))
    players_target = math.exp(p_exp)
    # Players only grow on this axis in any launch week we have seen; a
    # fitted fall is noise, and a wild rise is an artefact of a short
    # history, so keep the forecast between the last count and 50x it.
    players_target = min(max(players_target, float(last_players)),
                         50.0 * float(last_players))
    players_var = presid * (1.0 + (x_target - xbar_p) ** 2 / psxx) if psxx > 0 else presid

    # --- earners_i(t): log earners linear in log-time, per trophy. ---
    # First pass: each trophy's own fit, keeping only reads that name it.
    own = []  # per trophy: (a, b, resid, sxx, xbar, n_info, last_rate)
    for i, trophy in enumerate(trophies):
        xs_i, ys_i, ws_i = [], [], []
        last_rate = None
        for j, read in enumerate(history):
            rate = read["rates"][i]
            if rate is None:
                continue
            last_rate = rate
            earners = max(rate * float(read["players_tracked"]) / 100.0, 0.5)
            xs_i.append(xs[j])
            ys_i.append(math.log(earners))
            ws_i.append(recency[j])
        if len(xs_i) >= 2:
            a, b, resid, sxx = wls(xs_i, ys_i, ws_i)
            xbar = sum(w * x for w, x in zip(ws_i, xs_i)) / sum(ws_i)
        elif len(xs_i) == 1:
            a, b, resid, sxx, xbar = ys_i[0], 0.0, 0.0, 0.0, xs_i[0]
        else:
            a, b, resid, sxx, xbar = 0.0, 0.0, 0.0, 0.0, xs[-1]
        own.append((a, b, resid, sxx, xbar, len(xs_i), last_rate))

    # Tier median slopes, for shrinkage: trophies of one tier are earned
    # on a similar schedule, so a noisy trophy borrows its tier's slope.
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [own[i][1] for i, t in enumerate(trophies)
                  if t["tier"] == tier and own[i][5] >= MIN_READS_FOR_SLOPE]
        tier_slopes[tier] = median(slopes) if slopes else 0.0
    all_slopes = [own[i][1] for i in range(len(trophies))
                  if own[i][5] >= MIN_READS_FOR_SLOPE]
    global_slope = median(all_slopes) if all_slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        a, b, resid, sxx, xbar, n_info, last_rate = own[i]
        tier_b = tier_slopes.get(trophy["tier"], global_slope)
        if n_info >= MIN_READS_FOR_SLOPE:
            # Shrink the trophy's own slope toward its tier's, harder when
            # its own fit rests on few reads.
            b_use = (n_info * b + SHRINK_READS * tier_b) / (n_info + SHRINK_READS)
        else:
            b_use = tier_b if n_info >= 1 else global_slope
        log_earners_t = a + b_use * x_target
        # Earners cannot exceed the players tracked (the rate tops at
        # 100%), and the exponent is clamped so exp() cannot overflow.
        log_earners_t = min(max(log_earners_t, math.log(0.5)),
                            math.log(max(1.0, players_target)))
        earners_t = math.exp(log_earners_t)
        if last_rate is None:
            # Never seen: borrow the tier's median last rate, else 1%.
            tier_rates = [own[k][6] for k, t in enumerate(trophies)
                          if t["tier"] == trophy["tier"] and own[k][6] is not None]
            last_rate = median(tier_rates) if tier_rates else 1.0
            median_rate = clamp(last_rate)
        else:
            median_rate = 100.0 * earners_t / players_target
            median_rate = clamp(median_rate)
        # A trophy at 100% stays there: everyone tracked already has it.
        if last_rate is not None and last_rate >= 99.99:
            median_rate = 99.99

        # --- spread, in logit units. ---
        # Fit prediction variance (residuals plus slope uncertainty over
        # the extrapolation distance), players-forecast variance, the jump
        # prior for a refresh, and model risk growing with the horizon.
        pred_var = resid
        if sxx > 0:
            pred_var = resid * (1.0 + (x_target - xbar) ** 2 / sxx)
        var = pred_var + players_var + JUMP * JUMP
        var += (RISK_PER_SQRT_DAY ** 2) * horizon_days
        spread = math.sqrt(max(var, 0.0))
        spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)

        lg = logit(median_rate)
        q = [round4(100.0 / (1.0 + math.exp(-(lg + z * spread)))) for z in Z]
        # Rounding can only tie neighbours, never invert them, but guard
        # the contract anyway: five finite numbers, non-decreasing.
        q = [min(max(float(v), 0.0), 100.0) for v in q]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def fallback_case(case: dict) -> dict:
    """A valid forecast for a case the model cannot read: persistence at
    1% with a wide spread. Only used so a run never leaves a case invalid.
    """
    rows = []
    for trophy in case.get("trophies", []):
        lg = logit(1.0)
        q = [round4(100.0 / (1.0 + math.exp(-(lg + z * 1.0)))) for z in Z]
        rows.append({"id": trophy["id"], "q": [float(v) for v in q]})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case, path.name)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = fallback_case(case)
            except Exception:
                continue
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
