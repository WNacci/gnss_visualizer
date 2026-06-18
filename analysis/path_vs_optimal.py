#!/usr/bin/env python3
"""Search-path efficiency: actual path vs. the optimal baited-site tour.

For Phase-2 test trials in which all three baited sites are found (strict 2 m
radius), this script asks how close each sheep's search path comes to the
shortest possible route that visits the three baited sites, and whether that
efficiency improves with experience (assay).

Definitions
-----------
* ACTUAL path length — for each sheep, the cumulative path length
  (``cumulative_path_length`` x10 -> metres) interpolated to the trial
  COMPLETION TIME. Completion time is a trial-level quantity: the maximum over
  the three baited sites of each site's earliest entry time across all sheep
  (i.e. the moment the last baited site is first reached, strict 2 m radius).
* OPTIMAL path length — the shortest tour that STARTS at the sheep's entry
  position (its first track point) and visits all three baited canonical sites
  {A1, A2, A3} (positions from ``SITE_GRID``). Brute-force all 3! = 6 orderings,
  sum Euclidean leg distances (x10 -> metres), take the minimum.
* EFFICIENCY RATIO = optimal_length / actual_length, in (0, 1]; 1 == perfectly
  efficient (the sheep walked the shortest possible tour).

Per-sheep vs. per-trial
-----------------------
The efficiency ratio is computed PER SHEEP (each sheep has its own entry point
and its own walked path up to the shared completion time), then AVERAGED within
a trial to give one ratio per trial. The trial-level ratio is what enters the
across-assay trend test. This mirrors the "mean path to completion" trial
metric in generate_figures.py while giving every sheep its own optimal baseline.

Trend test
----------
Spearman(assay, trial efficiency ratio) with a within-group assay-shuffle null
(1000 permutations; assay labels permuted only within each group_num so the
group composition is preserved). Two-sided empirical p from |rho|.

Outputs a per-assay summary, a permutation-null result, a figure, and a final
block beginning "FINDINGS:". Safe to run repeatedly.
"""
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from gps_analysis import (
    build_trials,
    build_tracks_cache,
    load_trial_tracks,
    detect_site_visits,
    cumulative_path_length,
    SITE_GRID,
    BAITED_CANONICAL,
)

# ---------------------------------------------------------------------------
# Parameters (kept consistent with generate_figures.py "PATH LENGTH" section)
# ---------------------------------------------------------------------------
RADIUS = 0.2  # 0.2 grid units = 2 m (strict, for completion / baited discovery)
MIN_DWELL_S = 0.0
GRID_TO_M = 10.0  # 1 grid unit = 10 m
TEST_CONFIGS = {"A", "B", "C", "D"}
PHASE2_START = "2026-02-17"
_CTRL_GROUPS = {9, 14}
N_PERM = 1000
SEED = 42

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# Canonical baited-site positions (oriented frame). Sorted for determinism.
BAITED_SORTED = sorted(BAITED_CANONICAL)
BAITED_POS = np.array([SITE_GRID[s] for s in BAITED_SORTED], dtype=float)


# ---------------------------------------------------------------------------
# Optimal tour
# ---------------------------------------------------------------------------
def optimal_tour_length_m(start_xy):
    """Shortest tour (in metres) starting at start_xy and visiting all baited sites.

    Brute-forces all 3! orderings of the three baited canonical sites, summing
    Euclidean leg distances; the start point is fixed (the sheep's entry point)
    and is not itself a site to return to.
    """
    start = np.asarray(start_xy, dtype=float)
    best = np.inf
    for order in permutations(range(len(BAITED_POS))):
        prev = start
        total = 0.0
        for idx in order:
            total += float(np.hypot(*(BAITED_POS[idx] - prev)))
            prev = BAITED_POS[idx]
        best = min(best, total)
    return best * GRID_TO_M


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
trials = build_trials()
tracks_cache = build_tracks_cache(trials)

test_trials = [
    t
    for t in trials
    if t["config"] in TEST_CONFIGS
    and t["assay"] is not None
    and isinstance(t["assay"], int)
    and t["date"] >= PHASE2_START
    and t["group_num"] not in _CTRL_GROUPS
]
print(f"Total trials: {len(trials)}, Phase 2 test trials: {len(test_trials)}")


# ---------------------------------------------------------------------------
# Compute per-trial efficiency
# ---------------------------------------------------------------------------
print("Computing path-vs-optimal efficiency...")
records = []
n_no_tracks = 0
n_lt2_sheep = 0
n_incomplete = 0  # all 3 baited not found at 2 m

for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks:
        n_no_tracks += 1
        continue
    if len(tracks) < 2:
        n_lt2_sheep += 1
        continue

    visits = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)

    # Earliest entry time per baited site, across all sheep (strict 2 m).
    baited_disc = {}
    for site in BAITED_CANONICAL:
        vlist = visits.get(site, [])
        if vlist:
            baited_disc[site] = min(v[1] for v in vlist)

    if len(baited_disc) < 3:
        n_incomplete += 1
        continue

    # Completion time = moment the last baited site is first reached.
    completion_time = max(baited_disc.values())

    sheep_ratios = []
    for sid, trk in tracks.items():
        gx, gy, t = trk["gx"], trk["gy"], trk["t"]
        if len(t) < 2:
            continue
        cpl_m = cumulative_path_length(gx, gy) * GRID_TO_M
        # Actual path walked up to completion time (interpolated).
        actual_m = float(np.interp(completion_time, t, cpl_m))
        if not np.isfinite(actual_m) or actual_m <= 0:
            continue
        # Optimal tour from this sheep's entry position (first track point).
        optimal_m = optimal_tour_length_m((gx[0], gy[0]))
        ratio = optimal_m / actual_m
        # Clip to (0, 1]: a sheep cannot beat the optimal tour from its entry
        # point, but interpolation/discretisation can nudge the ratio just over
        # 1.0 in rare cases; cap so the metric stays interpretable.
        ratio = min(ratio, 1.0)
        sheep_ratios.append(ratio)

    if not sheep_ratios:
        continue

    records.append(
        {
            "group_num": trial["group_num"],
            "assay": trial["assay"],
            "completion_time": completion_time,
            "n_sheep_used": len(sheep_ratios),
            "efficiency": float(np.mean(sheep_ratios)),
        }
    )

df = pd.DataFrame(records)
print(
    f"  skipped: no tracks={n_no_tracks}, <2 sheep={n_lt2_sheep}, "
    f"all-3-baited not found={n_incomplete}"
)
print(f"  trials with all 3 baited sites found and usable: n={len(df)}")

if df.empty:
    print("\nFINDINGS: no Phase-2 test trials reached all three baited sites at "
          "the strict 2 m radius, so search-path efficiency cannot be computed.")
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Per-assay summary
# ---------------------------------------------------------------------------
assays_sorted = sorted(df["assay"].unique())
print("\nPer-assay efficiency (optimal / actual; 1 = perfectly efficient):")
print(f"  {'assay':>5}  {'n':>3}  {'median':>7}  {'mean':>6}  {'IQR':>16}")
for a in assays_sorted:
    e = df[df["assay"] == a]["efficiency"].to_numpy()
    iqr = f"[{np.quantile(e, 0.25):.3f}, {np.quantile(e, 0.75):.3f}]"
    print(f"  {a:>5}  {len(e):>3}  {np.median(e):>7.3f}  {np.mean(e):>6.3f}  {iqr:>16}")


# ---------------------------------------------------------------------------
# Trend test: Spearman(assay, efficiency) with within-group assay-shuffle null
# ---------------------------------------------------------------------------
def assay_shuffle_p(frame, col, n_perm=N_PERM, seed=SEED):
    """Spearman(assay, col) vs. a within-group assay-shuffle null.

    Returns (rho_obs, p_two_sided, null_mean, null_lo95, null_hi95, n).
    """
    d = frame.dropna(subset=[col]).reset_index(drop=True)
    if len(d) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, len(d)
    rho_obs = stats.spearmanr(d["assay"], d[col]).statistic
    rng = np.random.default_rng(seed)
    groups = [np.asarray(idx) for idx in d.groupby("group_num").groups.values()]
    assay = d["assay"].to_numpy(dtype=float)
    col_vals = d[col].to_numpy(dtype=float)
    null = np.empty(n_perm)
    for k in range(n_perm):
        shuffled = assay.copy()
        for pos in groups:
            shuffled[pos] = rng.permutation(assay[pos])
        null[k] = stats.spearmanr(shuffled, col_vals).statistic
    p = float(np.mean(np.abs(null) >= abs(rho_obs)))
    lo, hi = np.percentile(null, [2.5, 97.5])
    return rho_obs, p, float(np.mean(null)), float(lo), float(hi), len(d)


rho_obs, p_perm, null_mean, null_lo, null_hi, n_used = assay_shuffle_p(df, "efficiency")
print("\nTrend test (assay vs. efficiency ratio):")
print(f"  observed Spearman rho = {rho_obs:+.3f}  (n={n_used} trials)")
print(
    f"  within-group assay-shuffle null ({N_PERM} perms): "
    f"mean={null_mean:+.3f}, 95% CI [{null_lo:+.3f}, {null_hi:+.3f}]"
)
print(f"  two-sided empirical p = {p_perm:.3f}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)
COLOUR = "#3B7DD8"

fig, ax = plt.subplots(figsize=(5.5, 3.8))
data_e = [df[df["assay"] == a]["efficiency"].to_numpy() for a in assays_sorted]
bp = ax.boxplot(
    data_e,
    tick_labels=[str(a) for a in assays_sorted],
    patch_artist=True,
    widths=0.55,
    showfliers=False,
    medianprops=dict(color="k", lw=1.2),
    whiskerprops=dict(color="#333", lw=0.8),
    capprops=dict(color="#333", lw=0.8),
)
for box in bp["boxes"]:
    box.set_facecolor(COLOUR)
    box.set_alpha(0.35)
    box.set_edgecolor(COLOUR)
    box.set_linewidth(0.8)
for i, e in enumerate(data_e):
    jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(e))
    ax.scatter(np.full(len(e), i + 1) + jitter, e, s=12, alpha=0.55,
               color=COLOUR, edgecolors="none", zorder=3)
    ax.text(i + 1, 0.01, f"n={len(e)}", ha="center", va="bottom", fontsize=5.5,
            color="#777", transform=ax.get_xaxis_transform())

ax.axhline(1.0, color="#999", ls=":", lw=0.7, zorder=0)
ax.text(0.02, 1.0, "optimal", fontsize=6, color="#999", va="bottom",
        transform=ax.get_yaxis_transform())
ax.set_ylim(0, 1.08)
ax.set_xlabel("Assay level")
ax.set_ylabel("Search-path efficiency (optimal / actual)")
ax.set_title("Path efficiency vs. baited-site tour")
pstr = f"p = {p_perm:.3f}" if p_perm >= 0.001 else "p < 0.001"
ax.text(0.97, 0.06, f"Spearman $\\rho$ = {rho_obs:+.2f}\n{pstr} (perm.)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
        color="#333", style="italic")

fig.tight_layout()
fig.savefig(FIGDIR / "path_vs_optimal.pdf")
fig.savefig(FIGDIR / "path_vs_optimal.png")
plt.close(fig)
print(f"  -> {FIGDIR / 'path_vs_optimal.png'}")


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
overall_med = float(df["efficiency"].median())
overall_mean = float(df["efficiency"].mean())
first_a, last_a = assays_sorted[0], assays_sorted[-1]
med_first = float(df[df["assay"] == first_a]["efficiency"].median())
med_last = float(df[df["assay"] == last_a]["efficiency"].median())
direction = "improves" if rho_obs > 0 else "declines"
sig = "significant" if p_perm < 0.05 else "not significant"

print("\n" + "=" * 70)
print(
    "FINDINGS: Across {n} Phase-2 trials where all three baited sites were "
    "reached (2 m radius), search paths were far from optimal — the median "
    "efficiency ratio (optimal tour / actual path to completion) was "
    "{med:.2f} (mean {mean:.2f}), i.e. sheep walked roughly {x:.1f}x the "
    "shortest baited tour. Efficiency {dirn} with experience: Spearman "
    "rho(assay, efficiency) = {rho:+.2f}, two-sided permutation p = {p:.3f} "
    "({sig}); median efficiency at assay {fa} was {mf:.2f} vs. {ml:.2f} at "
    "assay {la}. The within-group assay-shuffle null centred at {nm:+.2f} "
    "(95% CI [{lo:+.2f}, {hi:+.2f}]).".format(
        n=len(df),
        med=overall_med,
        mean=overall_mean,
        x=(1.0 / overall_med if overall_med > 0 else float("nan")),
        dirn=direction,
        rho=rho_obs,
        p=p_perm,
        sig=sig,
        fa=first_a,
        mf=med_first,
        ml=med_last,
        la=last_a,
        nm=null_mean,
        lo=null_lo,
        hi=null_hi,
    )
)
print("=" * 70)
