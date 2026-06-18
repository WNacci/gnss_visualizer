#!/usr/bin/env python3
"""End-of-trial activity change in Phase-2 sheep test trials.

Trials are 35 min. For each Phase-2 test trial we build two group-level
time-series:
  - group-centroid SPEED  (m/min)  — how fast the flock's centre of mass moves
  - group SPREAD          (m)       — mean distance of sheep to the centroid

We then ask how activity changes over the course of a trial:

  (a) Early vs late: Delta = late - early, where "early" is the first
      EARLY_CUTOFF minutes (5-10 min explored) and "late" is the remainder,
      for both speed and spread.
  (b) Pre- vs post-completion: where the trial is completed (all 3 baited
      sites entered within RADIUS = 0.2 grid = 2 m), completion time is the
      max of the three earliest entry times. Compare mean speed/spread before
      vs after completion.
  (c) Overall slope of centroid speed vs trial time (linear fit).

Per-config mean Delta is bootstrapped (1000 resamples, seed 42) for 95% CIs,
and results are aggregated by assay. The script prints a "FINDINGS:" block.

Conventions match analysis/generate_figures.py:
  - t in minutes, grid units (1 unit = 10 m -> speed x10 gives m/min).
  - Tracks interpolated to a 10 Hz grid via _interp_to_1s.
  - Phase-2 cohort: config in {A,B,C,D}, integer assay, date >= 2026-02-17,
    group_num not in {9, 14}.

Safe to run repeatedly. Figures -> analysis/figures/.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from gps_analysis import (
    build_trials,
    build_tracks_cache,
    load_trial_tracks,
    detect_site_visits,
    BAITED_CANONICAL,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RADIUS = 0.2            # 0.2 grid units = 2 m (strict, for completion)
MIN_DWELL_S = 0.0
TEST_CONFIGS = {"A", "B", "C", "D"}
PHASE2_START = "2026-02-17"
_CTRL_GROUPS = {9, 14}
EARLY_CUTOFF = 7.5      # minutes; boundary between "early" and "late" windows
                        # (mid-point of the requested 5-10 min early window)
EARLY_LO = 5.0          # robustness: also report 5-min and 10-min cutoffs
EARLY_HI = 10.0
DUR_MIN = 35            # trial length (minutes)
SPEED_SMOOTH_S = 60     # centroid-speed smoothing window (samples at 10 Hz)
N_BOOT = 1000
SEED = 42

PALETTE = {
    "blue": "#3B7DD8", "orange": "#E8823A", "green": "#4AAD5B",
    "purple": "#8B6DAF", "red": "#D64550", "grey": "#888888", "gold": "#E8B83D",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Track interpolation (matches generate_figures.py _interp_to_1s)
# ---------------------------------------------------------------------------
def _interp_to_1s(tracks, dur_min=DUR_MIN):
    """Interpolate every sheep track onto a shared 10 Hz grid (0..dur_min)."""
    t_grid = np.arange(0, dur_min, 1 / 600)
    result = {}
    for sid, trk in tracks.items():
        if len(trk["t"]) < 2:
            continue
        order = np.argsort(trk["t"])
        t_s, gx_s, gy_s = trk["t"][order], trk["gx"][order], trk["gy"][order]
        mask = t_grid <= t_s.max()
        tg = t_grid[mask]
        if len(tg) < 2:
            continue
        result[sid] = {
            "gx": np.interp(tg, t_s, gx_s),
            "gy": np.interp(tg, t_s, gy_s),
            "t": tg,
        }
    return result


def _group_series(tracks):
    """Return (t_speed, speed_m_min, t_spread, spread_m) group time-series.

    speed = smoothed group-centroid speed in m/min (aligned with t_speed).
    spread = mean distance of sheep to the centroid in m (aligned with t_spread).
    Returns None if fewer than 2 sheep have usable tracks.
    """
    interp = _interp_to_1s(tracks)
    sids = sorted(interp.keys())
    if len(sids) < 2:
        return None
    T = min(len(interp[s]["gx"]) for s in sids)
    if T < 2:
        return None

    GX = np.column_stack([interp[s]["gx"][:T] for s in sids])
    GY = np.column_stack([interp[s]["gy"][:T] for s in sids])
    t = interp[sids[0]]["t"][:T]

    cx, cy = GX.mean(axis=1), GY.mean(axis=1)

    # Centroid speed: 10 Hz -> dt = 1/600 min; x10 -> m/min.
    dt = 1.0 / 600.0
    raw_speed = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2) / dt * 10.0
    win = min(SPEED_SMOOTH_S, max(1, len(raw_speed)))
    kernel = np.ones(win) / win
    speed = np.convolve(raw_speed, kernel, mode="same")
    t_speed = t[1:]  # speed aligns with the later edge of each diff

    spread = np.mean(np.sqrt((GX - cx[:, None]) ** 2
                             + (GY - cy[:, None]) ** 2), axis=1) * 10.0

    return t_speed, speed, t, spread


def _completion_time(tracks, field):
    """Completion = max of the 3 earliest baited-site entry times (within 2 m).

    Returns the time in minutes, or None if the trial never completes.
    """
    visits = detect_site_visits(tracks, field, RADIUS, MIN_DWELL_S)
    earliest = {}
    for site in BAITED_CANONICAL:
        vlist = visits.get(site, [])
        if vlist:
            earliest[site] = min(v[1] for v in vlist)
    if len(earliest) == len(BAITED_CANONICAL):
        return max(earliest.values())
    return None


def _window_mean(t, y, lo, hi):
    """Mean of y over t in [lo, hi). NaN if no samples."""
    m = (t >= lo) & (t < hi)
    return float(np.mean(y[m])) if m.any() else np.nan


# ---------------------------------------------------------------------------
# Bootstrap CI for a per-config mean of a quantity (resample trials within
# config, average per-config means, take percentile CI of the grand mean).
# ---------------------------------------------------------------------------
def _bootstrap_perconfig_mean(df, value_col, n_boot=N_BOOT, seed=SEED):
    """Bootstrap the grand mean of per-config means of `value_col`.

    Resampling is done within each config (trials are the resampling unit),
    so configs are weighted equally regardless of trial count. Returns
    (observed, lo95, hi95, n_trials, n_configs).
    """
    d = df.dropna(subset=[value_col])
    if len(d) == 0:
        return np.nan, np.nan, np.nan, 0, 0
    configs = sorted(d["config"].unique())
    by_cfg = {c: d.loc[d["config"] == c, value_col].to_numpy() for c in configs}

    def grand(sampler):
        means = []
        for c in configs:
            vals = by_cfg[c]
            if len(vals) == 0:
                continue
            means.append(np.mean(sampler(vals)))
        return np.mean(means) if means else np.nan

    observed = grand(lambda v: v)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = grand(lambda v: v[rng.integers(0, len(v), len(v))])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(observed), float(lo), float(hi), len(d), len(configs)


# ===========================================================================
# Load data and select cohort
# ===========================================================================
print("Loading data...")
trials = build_trials()
tracks_cache = build_tracks_cache(trials)

test_trials = [
    t for t in trials
    if t["config"] in TEST_CONFIGS
    and t["assay"] is not None
    and isinstance(t["assay"], int)
    and t["date"] >= PHASE2_START
    and t["group_num"] not in _CTRL_GROUPS
]
print(f"Total trials: {len(trials)}, Phase-2 test trials: {len(test_trials)}")


# ===========================================================================
# Per-trial activity metrics
# ===========================================================================
print("Computing per-trial activity time-series...")
records = []
n_skipped = 0
for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks or len(tracks) < 2:
        n_skipped += 1
        continue
    series = _group_series(tracks)
    if series is None:
        n_skipped += 1
        continue
    t_speed, speed, t_spread, spread = series
    if not (np.isfinite(speed).any() and np.isfinite(spread).any()):
        n_skipped += 1
        continue

    # (a) Early vs late (Delta = late - early), at the 7.5-min boundary,
    # with 5- and 10-min robustness variants.
    rec = {
        "name": trial["name"],
        "config": trial["config"],
        "assay": trial["assay"],
        "group_num": trial["group_num"],
        "n_sheep": len(tracks),
        "mean_speed": float(np.nanmean(speed)),
        "mean_spread": float(np.nanmean(spread)),
    }
    for tag, cut in [("", EARLY_CUTOFF), ("_5", EARLY_LO), ("_10", EARLY_HI)]:
        sp_e = _window_mean(t_speed, speed, 0.0, cut)
        sp_l = _window_mean(t_speed, speed, cut, DUR_MIN)
        spr_e = _window_mean(t_spread, spread, 0.0, cut)
        spr_l = _window_mean(t_spread, spread, cut, DUR_MIN)
        rec[f"d_speed{tag}"] = sp_l - sp_e
        rec[f"d_spread{tag}"] = spr_l - spr_e
        if tag == "":
            rec["speed_early"] = sp_e
            rec["speed_late"] = sp_l
            rec["spread_early"] = spr_e
            rec["spread_late"] = spr_l

    # (c) Overall slope of speed vs trial-time (m/min per min).
    fin = np.isfinite(t_speed) & np.isfinite(speed)
    if fin.sum() >= 2:
        sl = stats.linregress(t_speed[fin], speed[fin])
        rec["speed_slope"] = float(sl.slope)
    else:
        rec["speed_slope"] = np.nan

    # (b) Pre- vs post-completion.
    ct = _completion_time(tracks, trial["field"])
    rec["completion_time"] = ct
    if ct is not None and 0.0 < ct < DUR_MIN:
        sp_pre = _window_mean(t_speed, speed, 0.0, ct)
        sp_post = _window_mean(t_speed, speed, ct, DUR_MIN)
        spr_pre = _window_mean(t_spread, spread, 0.0, ct)
        spr_post = _window_mean(t_spread, spread, ct, DUR_MIN)
        rec["dc_speed"] = sp_post - sp_pre
        rec["dc_spread"] = spr_post - spr_pre
    else:
        rec["dc_speed"] = np.nan
        rec["dc_spread"] = np.nan

    records.append(rec)

df = pd.DataFrame(records)
print(f"  Usable trials: {len(df)} (skipped {n_skipped} for <2 sheep / empty tracks)")
n_complete = int(df["completion_time"].notna().sum()) if len(df) else 0
print(f"  Trials completing all 3 baited sites within 2 m: {n_complete}")

if len(df) == 0:
    print("\nFINDINGS: No usable Phase-2 trials with >=2 sheep were found; "
          "cannot assess end-of-trial activity change.")
    raise SystemExit(0)


# ===========================================================================
# Bootstrap per-config mean Deltas with 95 % CIs
# ===========================================================================
print("\nBootstrapping per-config mean Deltas (1000 resamples, seed 42)...")
boot_specs = [
    ("d_speed", "Delta speed (late-early, m/min)"),
    ("d_spread", "Delta spread (late-early, m)"),
    ("dc_speed", "Delta speed (post-pre completion, m/min)"),
    ("dc_spread", "Delta spread (post-pre completion, m)"),
    ("speed_slope", "Speed slope (m/min per min)"),
]
boot_results = {}
print(f"  {'metric':40s} {'obs':>8s}  {'95% CI':>20s}   n  k")
for col, label in boot_specs:
    obs, lo, hi, n, k = _bootstrap_perconfig_mean(df, col)
    boot_results[col] = (obs, lo, hi, n, k)
    if np.isfinite(obs):
        print(f"  {label:40s} {obs:8.3f}  [{lo:7.3f}, {hi:7.3f}]  {n:3d} {k}")
    else:
        print(f"  {label:40s} {'n/a':>8s}  {'(no data)':>20s}  {n:3d} {k}")


# Paired tests on early vs late (and pre vs post completion) across trials.
def _paired(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())
    w = stats.wilcoxon(a[m], b[m])
    return float(np.median(b[m] - a[m])), float(w.pvalue), int(m.sum())

print("\nPaired early-vs-late and pre-vs-post-completion (Wilcoxon):")
for label, a, b in [
    ("speed early vs late", df["speed_early"].to_numpy(), df["speed_late"].to_numpy()),
    ("spread early vs late", df["spread_early"].to_numpy(), df["spread_late"].to_numpy()),
]:
    med, p, n = _paired(a, b)
    print(f"  {label:32s} median Delta={med:+.3f}  p={p:.3f}  n={n}")
comp = df.dropna(subset=["completion_time"])
if len(comp) >= 3:
    for label, col in [("speed post-pre completion", "dc_speed"),
                       ("spread post-pre completion", "dc_spread")]:
        vals = comp[col].dropna().to_numpy()
        if len(vals) >= 3:
            w = stats.wilcoxon(vals)
            print(f"  {label:32s} median Delta={np.median(vals):+.3f}  "
                  f"p={w.pvalue:.3f}  n={len(vals)}")


# ===========================================================================
# Aggregate by assay
# ===========================================================================
print("\nPer-assay summary (median Deltas):")
assays = sorted(df["assay"].unique())
print(f"  {'assay':>5s} {'n':>3s} {'dSpeed':>8s} {'dSpread':>8s} "
      f"{'slope':>8s} {'dcSpeed':>8s} {'dcSpread':>8s}")
assay_rows = []
for a in assays:
    sub = df[df["assay"] == a]
    row = {
        "assay": a, "n": len(sub),
        "d_speed": sub["d_speed"].median(),
        "d_spread": sub["d_spread"].median(),
        "speed_slope": sub["speed_slope"].median(),
        "dc_speed": sub["dc_speed"].median(),
        "dc_spread": sub["dc_spread"].median(),
    }
    assay_rows.append(row)
    print(f"  {a:5d} {len(sub):3d} {row['d_speed']:8.3f} {row['d_spread']:8.3f} "
          f"{row['speed_slope']:8.4f} {row['dc_speed']:8.3f} {row['dc_spread']:8.3f}")
assay_df = pd.DataFrame(assay_rows)
assay_df.to_csv(FIGDIR / "end_of_trial_activity_by_assay.csv", index=False)


# ===========================================================================
# Figures
# ===========================================================================
print("\nGenerating figures...")

# Fig 1: early-vs-late paired dot plots for speed and spread.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
rng = np.random.default_rng(SEED)
for ax, e_col, l_col, ylabel, title, col in [
    (ax1, "speed_early", "speed_late", "Centroid speed (m/min)",
     f"a  Speed: first {EARLY_CUTOFF:g} min vs remainder", PALETTE["blue"]),
    (ax2, "spread_early", "spread_late", "Group spread (m)",
     f"b  Spread: first {EARLY_CUTOFF:g} min vs remainder", PALETTE["orange"]),
]:
    e = df[e_col].to_numpy()
    l = df[l_col].to_numpy()
    m = np.isfinite(e) & np.isfinite(l)
    for ev, lv in zip(e[m], l[m]):
        ax.plot([1, 2], [ev, lv], color=col, alpha=0.25, lw=0.6, zorder=1)
    ax.scatter(np.full(m.sum(), 1) + rng.uniform(-0.04, 0.04, m.sum()),
               e[m], s=14, color=col, alpha=0.6, zorder=2)
    ax.scatter(np.full(m.sum(), 2) + rng.uniform(-0.04, 0.04, m.sum()),
               l[m], s=14, color=col, alpha=0.6, zorder=2)
    ax.plot([1, 2], [np.nanmedian(e[m]), np.nanmedian(l[m])],
            color="k", lw=2, marker="o", zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["early", "late"])
    ax.set_xlim(0.6, 2.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
fig.tight_layout()
fig.savefig(FIGDIR / "end_of_trial_early_vs_late.png")
fig.savefig(FIGDIR / "end_of_trial_early_vs_late.pdf")
plt.close(fig)
print("  -> end_of_trial_early_vs_late")

# Fig 2: mean activity time-course across trials (binned), with completion marker.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
bins = np.arange(0, DUR_MIN + 0.5, 0.5)
centres = 0.5 * (bins[:-1] + bins[1:])
speed_stack, spread_stack = [], []
for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks or len(tracks) < 2:
        continue
    series = _group_series(tracks)
    if series is None:
        continue
    t_sp, sp, t_spr, spr = series
    sp_b, _, _ = stats.binned_statistic(t_sp, sp, "mean", bins=bins)
    spr_b, _, _ = stats.binned_statistic(t_spr, spr, "mean", bins=bins)
    speed_stack.append(sp_b)
    spread_stack.append(spr_b)
speed_stack = np.array(speed_stack)
spread_stack = np.array(spread_stack)
for ax, stack, ylabel, title, col in [
    (ax1, speed_stack, "Centroid speed (m/min)", "a  Mean speed over trial", PALETTE["blue"]),
    (ax2, spread_stack, "Group spread (m)", "b  Mean spread over trial", PALETTE["orange"]),
]:
    # A late time-bin can be all-NaN if every trial's tracks end before it;
    # suppress the resulting empty-slice warnings (those bins plot as NaN gaps).
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(stack, axis=0)
            sem = np.nanstd(stack, axis=0)
    n_at = np.sum(np.isfinite(stack), axis=0)
    sem = sem / np.sqrt(np.maximum(n_at, 1))
    ax.plot(centres, mean, color=col, lw=1.5)
    ax.fill_between(centres, mean - sem, mean + sem, color=col, alpha=0.2)
    ax.axvline(EARLY_CUTOFF, color="#999", ls=":", lw=0.8)
    med_ct = df["completion_time"].median()
    if np.isfinite(med_ct):
        ax.axvline(med_ct, color=PALETTE["red"], ls="--", lw=0.8)
        ax.text(med_ct, ax.get_ylim()[1], " median\n completion",
                color=PALETTE["red"], fontsize=5.5, va="top")
    ax.set_xlabel("Trial time (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
fig.tight_layout()
fig.savefig(FIGDIR / "end_of_trial_timecourse.png")
fig.savefig(FIGDIR / "end_of_trial_timecourse.pdf")
plt.close(fig)
print("  -> end_of_trial_timecourse")


# ===========================================================================
# FINDINGS
# ===========================================================================
def _dir(obs, lo, hi):
    if not np.isfinite(obs):
        return "n/a"
    sig = "" if (lo <= 0 <= hi) else " (CI excludes 0)"
    return f"{obs:+.2f}{sig}"

ds = boot_results["d_speed"]
dspr = boot_results["d_spread"]
dcs = boot_results["dc_speed"]
slp = boot_results["speed_slope"]

# Is the early window the most active? Compare to whole-trial late mean speed.
early_more_active = np.nanmedian(df["speed_early"]) > np.nanmedian(df["speed_late"])

print("\n" + "=" * 72)
print("FINDINGS: end-of-trial activity change (Phase-2 test trials)")
print("=" * 72)
print(f"Cohort: {len(df)} usable trials across {df['config'].nunique()} configs, "
      f"{df['assay'].nunique()} assay levels; {n_complete} completed all 3 "
      f"baited sites within 2 m.")
print(f"(a) Early-vs-late (boundary {EARLY_CUTOFF:g} min): "
      f"Delta speed = {_dir(*ds[:3])} m/min, "
      f"Delta spread = {_dir(*dspr[:3])} m "
      f"(per-config bootstrap mean, 95% CI).")
print(f"    First {EARLY_CUTOFF:g} min is "
      f"{'MORE' if early_more_active else 'NOT more'} active than the remainder "
      f"(median centroid speed early "
      f"{np.nanmedian(df['speed_early']):.2f} vs late "
      f"{np.nanmedian(df['speed_late']):.2f} m/min).")
print(f"(b) Post- minus pre-completion: speed Delta = {_dir(*dcs[:3])} m/min "
      f"(n={dcs[3]} completed trials). Negative => group winds down after "
      f"finding all reward.")
print(f"(c) Overall speed slope = {_dir(*slp[:3])} m/min per min of trial time. "
      f"Negative => activity declines toward trial end.")

wind_down = (np.isfinite(slp[0]) and slp[0] < 0) or (np.isfinite(ds[0]) and ds[0] < 0)
print("SUMMARY: The flock "
      + ("DOES" if wind_down else "does NOT clearly")
      + " wind down toward the end of the trial"
      + ("; the first window is the most active." if early_more_active
         else "; the early window is not the most active window."))
print("=" * 72)
