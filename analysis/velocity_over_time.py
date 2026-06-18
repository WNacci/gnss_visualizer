#!/usr/bin/env python3
"""Velocity across time: group-centroid and per-sheep speed vs trial time.

For each Phase-2 test trial we compute, as a function of trial time (0-35 min,
1-min bins):
  - the group-centroid speed (speed of the flock's centre of mass), and
  - the mean per-sheep speed (average over individuals of each sheep's speed).

These per-trial speed-time curves are aggregated by assay level into
mean +/- SEM speed-time profiles.

We then report:
  (a) overall mean speed by assay with a trend test --
      Spearman(assay, mean_speed) using a within-group assay-shuffle null
      (1000 perms, two-sided empirical p), and
  (b) the shape of the speed-time profile (early-trial vs late-trial speed).

Speeds are reported in m/min (1 grid unit = 10 m).  Plain script, mirrors the
style and speed computation of analysis/generate_figures.py.  Safe to run
repeatedly.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.signal import savgol_filter

from gps_analysis import (
    build_trials,
    build_tracks_cache,
    load_trial_tracks,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# Empty / all-NaN bins are expected (a bin where no sheep has data) and are
# handled as missing downstream; silence the resulting nanmean warnings.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "legend.frameon": False,
    "legend.fontsize": 7,
})

PALETTE = {
    "blue": "#3B7DD8",
    "orange": "#E8823A",
    "green": "#4AAD5B",
    "purple": "#8B6DAF",
    "red": "#D64550",
    "grey": "#888888",
}

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TEST_CONFIGS = {"A", "B", "C", "D"}
PHASE2_START = "2026-02-17"
_CTRL_GROUPS = {9, 14}          # control groups excluded from Phase-2 cohort
TRIAL_DUR_MIN = 35.0            # trials run for a fixed 35-min window
BIN_WIDTH_MIN = 1.0            # 1-min time bins
SAMPLE_HZ = 1.0                # resample to 1 s grid before computing speed
SAVGOL_WIN_S = 11             # smoothing window (seconds) for GPS jitter
SAVGOL_POLY = 3
SPEED_CLIP_M_MIN = 400.0      # physiologically implausible above this; clip jitter spikes
N_PERM = 1000
SEED = 42

# Behaviourally meaningful analysis window.  The raw 35-min profile is bracketed
# by two non-foraging transients that would otherwise dominate any summary:
#   - a release/startle spike in minute 0 (animals enter and bolt), and
#   - an end-of-trial handling spike from ~minute 30 onward (animals are herded
#     out / recovered), which is monotone and large (15 -> 38 m/min).
# We therefore characterise within-trial speed dynamics over WIN_LO..WIN_HI and
# compute the overall mean speed (for the assay trend) over the same window, so
# the trend reflects foraging movement rather than handling artefacts.  The full
# 0-35 min profile is still plotted for transparency.
WIN_LO = 1.0
WIN_HI = 30.0

N_BINS = int(round(TRIAL_DUR_MIN / BIN_WIDTH_MIN))
BIN_EDGES = np.arange(0, TRIAL_DUR_MIN + BIN_WIDTH_MIN / 2, BIN_WIDTH_MIN)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
WIN_MASK = (BIN_CENTERS >= WIN_LO) & (BIN_CENTERS < WIN_HI)


# ===========================================================================
# Helpers
# ===========================================================================
def _resample_smooth(trk, dur_min=TRIAL_DUR_MIN, hz=SAMPLE_HZ):
    """Interpolate a single track onto a regular grid and Savitzky-Golay smooth.

    Returns (t_grid_min, gx, gy) on a 1/hz-second grid covering the part of the
    trial the sheep was actually recorded for, or None if too few samples.
    """
    t = np.asarray(trk["t"], dtype=float)
    gx = np.asarray(trk["gx"], dtype=float)
    gy = np.asarray(trk["gy"], dtype=float)
    if len(t) < 5:
        return None
    order = np.argsort(t)
    t, gx, gy = t[order], gx[order], gy[order]
    # keep unique time stamps (interp requires strictly increasing x)
    uniq = np.concatenate([[True], np.diff(t) > 0])
    t, gx, gy = t[uniq], gx[uniq], gy[uniq]
    if len(t) < 5:
        return None

    step_min = (1.0 / hz) / 60.0
    t_max = min(t.max(), dur_min)
    if t_max <= t.min():
        return None
    tg = np.arange(max(t.min(), 0.0), t_max, step_min)
    if len(tg) < 5:
        return None
    gxg = np.interp(tg, t, gx)
    gyg = np.interp(tg, t, gy)

    n = len(gxg)
    w = min(SAVGOL_WIN_S, n if n % 2 == 1 else n - 1)
    if w >= SAVGOL_POLY + 2:
        gxg = savgol_filter(gxg, w, SAVGOL_POLY)
        gyg = savgol_filter(gyg, w, SAVGOL_POLY)
    return tg, gxg, gyg


def _speed_m_min(tg, gx, gy):
    """Per-sample speed (m/min) at the midpoints, with matching time stamps.

    Distance in grid units / dt in minutes * 10 -> m/min.  GPS jitter spikes
    are clipped at SPEED_CLIP_M_MIN.
    """
    dt = np.diff(tg)
    dt[dt <= 0] = np.nan
    dist_gu = np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2)
    speed = dist_gu / dt * 10.0
    speed = np.clip(speed, 0, SPEED_CLIP_M_MIN)
    t_mid = (tg[:-1] + tg[1:]) / 2.0
    return t_mid, speed


def _bin_speed(t_mid, speed):
    """Bin a speed time series into 1-min bins -> array length N_BINS (NaN if empty)."""
    out = np.full(N_BINS, np.nan)
    finite = np.isfinite(speed)
    if not finite.any():
        return out
    idx = np.clip(np.floor(t_mid[finite] / BIN_WIDTH_MIN).astype(int), 0, N_BINS - 1)
    sp = speed[finite]
    for b in range(N_BINS):
        sel = sp[idx == b]
        if len(sel):
            out[b] = np.nanmean(sel)
    return out


# ===========================================================================
# Load data  (tracks cache only -- do NOT build the 2 GB gps cache)
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
# Per-trial speed-time curves
# ===========================================================================
records = []                 # one row per trial: overall mean speeds + meta
centroid_curves = {}         # trial_name -> binned centroid speed (N_BINS,)
persheep_curves = {}         # trial_name -> binned mean per-sheep speed (N_BINS,)
n_skipped = 0

for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks:
        n_skipped += 1
        continue

    # --- per-sheep speeds on a common regular grid ---
    resampled = {}
    for sid, trk in tracks.items():
        rs = _resample_smooth(trk)
        if rs is not None:
            resampled[sid] = rs
    if len(resampled) < 2:        # need >=2 sheep for a sensible group trial
        n_skipped += 1
        continue

    persheep_binned = []
    for sid, (tg, gx, gy) in resampled.items():
        t_mid, sp = _speed_m_min(tg, gx, gy)
        persheep_binned.append(_bin_speed(t_mid, sp))
    persheep_binned = np.vstack(persheep_binned)
    # mean across sheep within each bin (ignore sheep with no data in a bin)
    persheep_curve = np.nanmean(persheep_binned, axis=0)

    # --- group-centroid speed on a shared time grid ---
    # Restrict to the overlap window so the centroid is well defined.
    t_lo = max(rs[0][0] for rs in resampled.values())
    t_hi = min(rs[0][-1] for rs in resampled.values())
    centroid_curve = np.full(N_BINS, np.nan)
    if t_hi > t_lo:
        step_min = (1.0 / SAMPLE_HZ) / 60.0
        tg_c = np.arange(t_lo, t_hi, step_min)
        if len(tg_c) >= 5:
            gxs = np.vstack([np.interp(tg_c, rs[0], rs[1]) for rs in resampled.values()])
            gys = np.vstack([np.interp(tg_c, rs[0], rs[2]) for rs in resampled.values()])
            cx = gxs.mean(axis=0)
            cy = gys.mean(axis=0)
            t_mid_c, sp_c = _speed_m_min(tg_c, cx, cy)
            centroid_curve = _bin_speed(t_mid_c, sp_c)

    centroid_curves[trial["name"]] = centroid_curve
    persheep_curves[trial["name"]] = persheep_curve

    records.append({
        "name": trial["name"],
        "group_num": trial["group_num"],
        "assay": trial["assay"],
        "n_sheep": len(resampled),
        # Overall mean speed over the behavioural window (excludes the release
        # and end-of-trial handling transients).  nan-safe: a trial contributes
        # only if it has >=1 finite bin in the window.
        "mean_centroid_speed": float(np.nanmean(centroid_curve[WIN_MASK]))
        if np.isfinite(centroid_curve[WIN_MASK]).any() else np.nan,
        "mean_persheep_speed": float(np.nanmean(persheep_curve[WIN_MASK]))
        if np.isfinite(persheep_curve[WIN_MASK]).any() else np.nan,
    })

df = pd.DataFrame(records)
print(f"Trials used: {len(df)}, skipped (empty / <2 sheep): {n_skipped}")
if df.empty:
    raise SystemExit("No usable trials -- aborting.")

assays_sorted = sorted(df["assay"].unique())


# ===========================================================================
# (a) Overall mean speed by assay + within-group assay-shuffle trend test
# ===========================================================================
def assay_shuffle_trend(frame, col, n_perm=N_PERM, seed=SEED):
    """Spearman(assay, col) with a within-group assay-shuffle null.

    Returns (rho_obs, p_two_sided, null_mean, lo95, hi95, n).
    """
    d = frame.dropna(subset=[col]).reset_index(drop=True)
    if len(d) < 3 or d["assay"].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, len(d)
    assay = d["assay"].to_numpy(dtype=float)
    vals = d[col].to_numpy(dtype=float)
    rho_obs = stats.spearmanr(assay, vals).statistic
    rng = np.random.default_rng(seed)
    groups = [np.asarray(idx) for idx in d.groupby("group_num").groups.values()]
    null = np.empty(n_perm)
    for k in range(n_perm):
        shuffled = assay.copy()
        for pos in groups:
            shuffled[pos] = rng.permutation(assay[pos])
        null[k] = stats.spearmanr(shuffled, vals).statistic
    p = float(np.mean(np.abs(null) >= abs(rho_obs)))
    lo, hi = np.percentile(null, [2.5, 97.5])
    return rho_obs, p, float(np.mean(null)), float(lo), float(hi), len(d)


trend = {}
for col in ["mean_centroid_speed", "mean_persheep_speed"]:
    trend[col] = assay_shuffle_trend(df, col)


# ===========================================================================
# (b) Speed-time profile shape, aggregated by assay (mean +/- SEM)
# ===========================================================================
def _sem(a, axis=0):
    a = np.asarray(a, dtype=float)
    n = np.sum(np.isfinite(a), axis=axis)
    sd = np.nanstd(a, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = np.where(n > 1, sd / np.sqrt(n), np.nan)
    return sem


def profile_by_assay(curves):
    """Stack per-trial binned curves by assay -> {assay: (mean[N_BINS], sem[N_BINS], n)}."""
    out = {}
    for a in assays_sorted:
        names = df[df["assay"] == a]["name"].tolist()
        stack = np.vstack([curves[n] for n in names if n in curves])
        out[a] = (np.nanmean(stack, axis=0), _sem(stack, axis=0), stack.shape[0])
    return out


centroid_profiles = profile_by_assay(centroid_curves)
persheep_profiles = profile_by_assay(persheep_curves)

# Overall (all assays pooled) profile for shape description.
all_centroid = np.vstack(list(centroid_curves.values()))
all_persheep = np.vstack(list(persheep_curves.values()))
overall_centroid_profile = np.nanmean(all_centroid, axis=0)
overall_persheep_profile = np.nanmean(all_persheep, axis=0)

# Within-trial shape over the behavioural window: compare the first 5 min of the
# window (WIN_LO..WIN_LO+5) against the remainder of the window.  Both endpoints
# are clear of the release (min 0) and handling (>=min 30) transients.
EARLY_BINS = (BIN_CENTERS >= WIN_LO) & (BIN_CENTERS < WIN_LO + 5.0)
LATE_BINS = (BIN_CENTERS >= WIN_HI - 5.0) & (BIN_CENTERS < WIN_HI)
early_per_trial = np.nanmean(np.where(EARLY_BINS, all_centroid, np.nan), axis=1)
late_per_trial = np.nanmean(np.where(LATE_BINS, all_centroid, np.nan), axis=1)
paired = np.isfinite(early_per_trial) & np.isfinite(late_per_trial)
early_mean = float(np.nanmean(early_per_trial))
late_mean = float(np.nanmean(late_per_trial))
if paired.sum() >= 3:
    w_stat, w_p = stats.wilcoxon(early_per_trial[paired], late_per_trial[paired])
else:
    w_stat, w_p = np.nan, np.nan

# Spearman of centroid speed vs time bin within the window (monotone trend).
win_finite = WIN_MASK & np.isfinite(overall_centroid_profile)
rho_time, p_time = stats.spearmanr(
    BIN_CENTERS[win_finite], overall_centroid_profile[win_finite]
)


# ===========================================================================
# Figures
# ===========================================================================
cmap = plt.colormaps["viridis"]
n_assay = len(assays_sorted)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for i, a in enumerate(assays_sorted):
    col = cmap(i / max(n_assay - 1, 1))
    m, s, n = centroid_profiles[a]
    ax1.plot(BIN_CENTERS, m, color=col, lw=1.4, label=f"assay {a} (n={n})")
    ax1.fill_between(BIN_CENTERS, m - s, m + s, color=col, alpha=0.15, linewidth=0)
ax1.set_xlabel("Trial time (min)")
ax1.set_ylabel("Group-centroid speed (m/min)")
ax1.set_title("a  Centroid speed vs time, by assay")
ax1.set_xlim(0, TRIAL_DUR_MIN)
ax1.legend(ncol=2, fontsize=6)

for i, a in enumerate(assays_sorted):
    col = cmap(i / max(n_assay - 1, 1))
    m, s, n = persheep_profiles[a]
    ax2.plot(BIN_CENTERS, m, color=col, lw=1.4, label=f"assay {a} (n={n})")
    ax2.fill_between(BIN_CENTERS, m - s, m + s, color=col, alpha=0.15, linewidth=0)
ax2.set_xlabel("Trial time (min)")
ax2.set_ylabel("Mean per-sheep speed (m/min)")
ax2.set_title("b  Per-sheep speed vs time, by assay")
ax2.set_xlim(0, TRIAL_DUR_MIN)
ax2.legend(ncol=2, fontsize=6)

fig.tight_layout()
fig.savefig(FIGDIR / "velocity_over_time_profiles.pdf")
fig.savefig(FIGDIR / "velocity_over_time_profiles.png")
plt.close(fig)
print("  -> velocity_over_time_profiles")

# Overall pooled profile + overall mean speed by assay.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
ax1.plot(BIN_CENTERS, overall_centroid_profile, color=PALETTE["blue"], lw=1.6,
         label="centroid")
ax1.plot(BIN_CENTERS, overall_persheep_profile, color=PALETTE["orange"], lw=1.6,
         label="per-sheep")
ax1.axvspan(WIN_LO, WIN_HI, color="0.85", alpha=0.5, zorder=0,
            label=f"analysis window ({WIN_LO:.0f}-{WIN_HI:.0f} min)")
ax1.set_xlabel("Trial time (min)")
ax1.set_ylabel("Speed (m/min)")
ax1.set_title("a  Pooled speed-time profile (all assays)")
ax1.set_xlim(0, TRIAL_DUR_MIN)
ax1.legend()

mean_by_assay = [df[df["assay"] == a]["mean_centroid_speed"].values for a in assays_sorted]
ax2.boxplot(mean_by_assay, tick_labels=[str(a) for a in assays_sorted],
            showfliers=False, widths=0.55,
            medianprops=dict(color="k", lw=1.2))
for i, d in enumerate(mean_by_assay):
    jit = np.random.default_rng(SEED + i).uniform(-0.15, 0.15, len(d))
    ax2.scatter(np.full(len(d), i + 1) + jit, d, s=12, alpha=0.55,
                color=PALETTE["blue"], edgecolors="none", zorder=3)
ax2.set_xlabel("Assay level")
ax2.set_ylabel("Mean centroid speed (m/min)")
ax2.set_title("b  Overall mean speed by assay")
rho, p, *_ = trend["mean_centroid_speed"]
ax2.text(0.97, 0.95, f"$\\rho$={rho:+.2f}, p={p:.3f}",
         transform=ax2.transAxes, ha="right", va="top", fontsize=7, style="italic")

fig.tight_layout()
fig.savefig(FIGDIR / "velocity_over_time_summary.pdf")
fig.savefig(FIGDIR / "velocity_over_time_summary.png")
plt.close(fig)
print("  -> velocity_over_time_summary")


# ===========================================================================
# FINDINGS
# ===========================================================================
# Peak/trough within the behavioural window (avoid the transient end bins).
win_prof = np.where(WIN_MASK, overall_centroid_profile, np.nan)
peak_bin = int(np.nanargmax(win_prof))
trough_bin = int(np.nanargmin(win_prof))

print()
print("=" * 72)
print("FINDINGS: velocity across time")
print("=" * 72)
print(f"Cohort: {len(df)} Phase-2 test trials "
      f"(assays {assays_sorted}), {n_skipped} skipped (empty / <2 sheep).")
print(f"Analysis window {WIN_LO:.0f}-{WIN_HI:.0f} min (excludes minute-0 release "
      "spike and the end-of-trial handling spike from ~min 30).")
print(f"Window mean centroid speed = {df['mean_centroid_speed'].mean():.1f} m/min; "
      f"mean per-sheep speed = {df['mean_persheep_speed'].mean():.1f} m/min.")
print(f"(For context, the raw profile shows ~{overall_centroid_profile[0]:.0f} m/min "
      f"in minute 0 and rises to ~{np.nanmax(overall_centroid_profile):.0f} m/min "
      "at trial end -- both non-foraging transients.)")
print()
print("(a) Trend across assays (Spearman, within-group assay-shuffle null, "
      f"{N_PERM} perms, two-sided; window mean speed):")
for col, lbl in [("mean_centroid_speed", "centroid"),
                 ("mean_persheep_speed", "per-sheep")]:
    rho, p, nm, lo, hi, n = trend[col]
    print(f"  {lbl:9s} rho={rho:+.3f}  p={p:.3f}  "
          f"null mean={nm:+.3f} [95% {lo:+.3f}, {hi:+.3f}]  n={n}")
print()
print("(b) Within-trial speed-time profile shape (pooled across assays, window):")
print(f"  Early ({WIN_LO:.0f}-{WIN_LO + 5:.0f} min) centroid speed = {early_mean:.1f} "
      f"m/min; late ({WIN_HI - 5:.0f}-{WIN_HI:.0f} min) = {late_mean:.1f} m/min "
      f"(Wilcoxon p={w_p:.3f}, n={int(paired.sum())}).")
print(f"  Peak centroid speed at ~{BIN_CENTERS[peak_bin]:.0f} min "
      f"({overall_centroid_profile[peak_bin]:.1f} m/min); "
      f"minimum at ~{BIN_CENTERS[trough_bin]:.0f} min "
      f"({overall_centroid_profile[trough_bin]:.1f} m/min).")
print(f"  Spearman(speed, time bin within window) = {rho_time:+.3f} (p={p_time:.3f}) "
      "-- negative => speed declines through the trial.")
_direction = ("higher early then declining" if early_mean > late_mean
              else "lower early then rising")
print(f"  Summary: within-trial speed is {_direction}; "
      f"across assays the centroid-speed trend is "
      f"{'significant' if trend['mean_centroid_speed'][1] < 0.05 else 'not significant'} "
      f"(p={trend['mean_centroid_speed'][1]:.3f}).")
print("=" * 72)
