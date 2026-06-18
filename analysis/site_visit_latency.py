#!/usr/bin/env python3
"""Latency to the Nth distinct site visit, per Phase-2 test trial.

For each trial we ask: how quickly does the *group* (any sheep) first reach its
1st, 2nd, 3rd, ... distinct reward site?  We take the earliest entry time across
all sheep for each site (the group-level first-arrival time), then sort those
arrival times ascending.  The k-th value is the group's latency to its k-th
distinct site.

Two site sets are analysed:
  - ALL sites at the generous 5 m (0.5 grid-unit) explore radius — awareness of
    any of the 12 sites.
  - BAITED sites {A1, A2, A3} at the strict 2 m (0.2 grid-unit) radius — actually
    reaching a reward.

Outputs (analysis/figures/):
  - site_visit_latency_curves.{pdf,png} — median latency-to-Nth-site curves
    (all sites and baited-only), with IQR bands.
  - site_visit_latency_by_assay.{pdf,png} — latency to 1st/2nd/3rd baited site
    by assay level.

Trend test: Spearman(assay, latency_to_3rd_baited) with a within-group
assay-shuffle permutation null (1000 perms, two-sided empirical p).

FINDINGS are printed at the end.
"""
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
# Nature-style plot defaults (matched to analysis/generate_figures.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,
    "axes.labelsize": 9,
    "axes.labelpad": 5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
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
    "gold": "#E8B83D",
}

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RADIUS_EXPLORE = 0.5   # 0.5 grid units = 5 m (generous, any-site exploration)
RADIUS_BAITED = 0.2    # 0.2 grid units = 2 m (strict, must reach reward)
MIN_DWELL_S = 0.0
TEST_CONFIGS = {"A", "B", "C", "D"}
PHASE2_START = "2026-02-17"
_CTRL_GROUPS = {9, 14}
N_PERM = 1000
SEED = 42
TRIAL_DUR = 35.0       # minutes; latencies must fall within [0, TRIAL_DUR]
N_BAITED = len(BAITED_CANONICAL)  # 3
N_ALL = 12             # total sites in the canonical grid

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
trials = build_trials()
tracks_cache = build_tracks_cache(trials)

test_trials = [t for t in trials
               if t["config"] in TEST_CONFIGS
               and t["assay"] is not None
               and isinstance(t["assay"], int)
               and t["date"] >= PHASE2_START
               and t["group_num"] not in _CTRL_GROUPS]

print(f"Total trials: {len(trials)}, Phase 2 test trials: {len(test_trials)}")


# ===========================================================================
# Helpers
# ===========================================================================
def group_first_arrivals(visits):
    """Per site, earliest entry_min across sheep; returns sorted ascending list.

    `visits` is detect_site_visits() output: site -> [(sid, entry, exit), ...].
    Sites never visited contribute nothing.  The returned list's k-th element
    (0-indexed) is the group latency to the (k+1)-th distinct site.
    """
    arrivals = []
    for vlist in visits.values():
        if vlist:
            arrivals.append(min(v[1] for v in vlist))
    return sorted(arrivals)


def nth_latency(sorted_arrivals, n):
    """Latency to the n-th distinct site (1-indexed); NaN if not reached."""
    if len(sorted_arrivals) >= n:
        return sorted_arrivals[n - 1]
    return np.nan


# ===========================================================================
# 1. PER-TRIAL LATENCY TO Nth SITE
# ===========================================================================
print("Computing per-trial latency-to-Nth-site...")
records = []
n_skipped = 0
for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    # Robustness: skip empty tracks / fewer than 2 sheep (mislabel-safe: we never
    # index a specific sheep id, only aggregate across whatever ids are present).
    if not tracks or len(tracks) < 2:
        n_skipped += 1
        continue

    visits_all = detect_site_visits(tracks, trial["field"], RADIUS_EXPLORE, MIN_DWELL_S)
    visits_baited_full = detect_site_visits(tracks, trial["field"], RADIUS_BAITED, MIN_DWELL_S)
    # Restrict the strict-radius detection to the canonical baited triplet.
    visits_baited = {s: v for s, v in visits_baited_full.items()
                     if s in BAITED_CANONICAL}

    arr_all = group_first_arrivals(visits_all)
    arr_baited = group_first_arrivals(visits_baited)

    rec = {
        "group_num": trial["group_num"],
        "assay": trial["assay"],
        "config": trial["config"],
        "n_sites_all": len(arr_all),
        "n_baited": len(arr_baited),
    }
    for n in range(1, N_ALL + 1):
        rec[f"all_{n}"] = nth_latency(arr_all, n)
    for n in range(1, N_BAITED + 1):
        rec[f"baited_{n}"] = nth_latency(arr_baited, n)
    records.append(rec)

df = pd.DataFrame(records)
print(f"  Trials used: {len(df)} (skipped {n_skipped} empty / <2 sheep)")
assays_sorted = sorted(df["assay"].unique())


# ===========================================================================
# 2. LATENCY-TO-Nth-SITE CURVES (median + IQR)
# ===========================================================================
def latency_curve(frame, prefix, n_max):
    """Return (ns, median, q25, q75, counts) over n = 1..n_max."""
    ns, med, q25, q75, cnt = [], [], [], [], []
    for n in range(1, n_max + 1):
        col = frame[f"{prefix}_{n}"].dropna()
        ns.append(n)
        cnt.append(len(col))
        if len(col) > 0:
            med.append(col.median())
            q25.append(col.quantile(0.25))
            q75.append(col.quantile(0.75))
        else:
            med.append(np.nan); q25.append(np.nan); q75.append(np.nan)
    return (np.array(ns), np.array(med), np.array(q25),
            np.array(q75), np.array(cnt))

ns_all, med_all, q25_all, q75_all, cnt_all = latency_curve(df, "all", N_ALL)
ns_b, med_b, q25_b, q75_b, cnt_b = latency_curve(df, "baited", N_BAITED)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

ax1.fill_between(ns_all, q25_all, q75_all, color=PALETTE["blue"], alpha=0.2,
                 linewidth=0)
ax1.plot(ns_all, med_all, "-o", color=PALETTE["blue"], ms=4, lw=1.4)
ax1.set_xlabel("Nth distinct site reached")
ax1.set_ylabel("Group latency (min)")
ax1.set_title("a  Latency to Nth site (any, 5 m)")
ax1.set_xticks(ns_all)
for n, m, c in zip(ns_all, med_all, cnt_all):
    if np.isfinite(m):
        ax1.text(n, m, f" n={c}", fontsize=4.5, color="#888", va="bottom")

ax2.fill_between(ns_b, q25_b, q75_b, color=PALETTE["gold"], alpha=0.25,
                 linewidth=0)
ax2.plot(ns_b, med_b, "-o", color=PALETTE["orange"], ms=5, lw=1.6)
ax2.set_xlabel("Nth distinct baited site reached")
ax2.set_ylabel("Group latency (min)")
ax2.set_title("b  Latency to Nth baited site (2 m)")
ax2.set_xticks(ns_b)
for n, m, c in zip(ns_b, med_b, cnt_b):
    if np.isfinite(m):
        ax2.text(n, m, f" n={c}", fontsize=5, color="#888", va="bottom")

fig.tight_layout()
fig.savefig(FIGDIR / "site_visit_latency_curves.pdf")
fig.savefig(FIGDIR / "site_visit_latency_curves.png")
plt.close(fig)
print("  -> site_visit_latency_curves")


# ===========================================================================
# 3. LATENCY TO Nth BAITED SITE BY ASSAY
# ===========================================================================
fig, axes = plt.subplots(1, N_BAITED, figsize=(11, 3.6), sharey=True)
colours = [PALETTE["green"], PALETTE["orange"], PALETTE["red"]]
ordinals = {1: "1st", 2: "2nd", 3: "3rd"}
for n, ax, col in zip(range(1, N_BAITED + 1), axes, colours):
    data = [df[df["assay"] == a][f"baited_{n}"].dropna().values
            for a in assays_sorted]
    bp = ax.boxplot(data, tick_labels=[str(a) for a in assays_sorted],
                    patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="k", lw=1.2),
                    whiskerprops=dict(color="#333", lw=0.8),
                    capprops=dict(color="#333", lw=0.8))
    for box in bp["boxes"]:
        box.set_facecolor(col); box.set_alpha(0.35)
        box.set_edgecolor(col); box.set_linewidth(0.8)
    for i, d in enumerate(data):
        jit = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(d))
        ax.scatter(np.full(len(d), i + 1) + jit, d, s=10, alpha=0.55,
                   color=col, edgecolors="none", zorder=3)
        ax.text(i + 1, 0.01, f"n={len(d)}", ha="center", va="bottom",
                fontsize=5.5, color="#777", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Assay level")
    ax.set_title(f"Latency to {ordinals[n]} baited site")
    # Spearman annotation (raw, descriptive)
    sub = df.dropna(subset=[f"baited_{n}"])
    if len(sub) >= 3:
        rho, p_sp = stats.spearmanr(sub["assay"], sub[f"baited_{n}"])
        pstr = f"p = {p_sp:.2e}" if p_sp < 0.001 else f"p = {p_sp:.3f}"
        ax.text(0.97, 0.96, f"$\\rho$ = {rho:.2f}, {pstr}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6, color="#333", style="italic")
axes[0].set_ylabel("Group latency (min)")
fig.tight_layout()
fig.savefig(FIGDIR / "site_visit_latency_by_assay.pdf")
fig.savefig(FIGDIR / "site_visit_latency_by_assay.png")
plt.close(fig)
print("  -> site_visit_latency_by_assay")


# ===========================================================================
# 4. AGGREGATE BY ASSAY (median latency tables)
# ===========================================================================
print("\nMedian latency to Nth baited site, by assay (min; n in parens):")
agg_rows = []
for a in assays_sorted:
    sub = df[df["assay"] == a]
    row = {"assay": a, "N_trials": len(sub)}
    for n in range(1, N_BAITED + 1):
        col = sub[f"baited_{n}"].dropna()
        row[f"baited_{n}_med"] = col.median() if len(col) else np.nan
        row[f"baited_{n}_n"] = len(col)
    agg_rows.append(row)
agg_df = pd.DataFrame(agg_rows)
for _, r in agg_df.iterrows():
    parts = []
    for n in range(1, N_BAITED + 1):
        m = r[f"baited_{n}_med"]
        parts.append(f"{ordinals[n]}={m:5.1f} (n={int(r[f'baited_{n}_n'])})"
                     if np.isfinite(m) else f"{ordinals[n]}=  NaN (n=0)")
    print(f"  assay {int(r['assay'])}: " + "  ".join(parts))

print("\nMedian latency to Nth site of ANY type, by assay (min):")
for a in assays_sorted:
    sub = df[df["assay"] == a]
    meds = []
    for n in range(1, N_ALL + 1):
        col = sub[f"all_{n}"].dropna()
        meds.append(f"{col.median():.1f}" if len(col) else "NaN")
    print(f"  assay {int(a)}: " + " ".join(f"{v:>5s}" for v in meds))


# ===========================================================================
# 5. TREND TEST: Spearman(assay, latency_to_3rd_baited)
#    with within-group assay-shuffle null
# ===========================================================================
def assay_shuffle_test(frame, col, n_perm=N_PERM, seed=SEED):
    """Spearman rho of (assay, col) with within-group assay-shuffle null.

    Returns (rho_obs, p_two_sided, null_mean, null_lo95, null_hi95, n).
    """
    d = frame.dropna(subset=[col]).reset_index(drop=True)
    if len(d) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, len(d)
    assay = d["assay"].to_numpy(dtype=float)
    target = d[col].to_numpy(dtype=float)
    rho_obs = stats.spearmanr(assay, target).statistic
    rng = np.random.default_rng(seed)
    groups = [np.asarray(idx) for idx in d.groupby("group_num").groups.values()]
    null = np.empty(n_perm)
    for k in range(n_perm):
        shuffled = assay.copy()
        for pos in groups:
            shuffled[pos] = rng.permutation(assay[pos])
        null[k] = stats.spearmanr(shuffled, target).statistic
    p_two = float(np.mean(np.abs(null) >= abs(rho_obs)))
    lo, hi = np.percentile(null, [2.5, 97.5])
    return rho_obs, p_two, float(np.mean(null)), float(lo), float(hi), len(d)

print("\nTrend test: Spearman(assay, latency) with within-group assay-shuffle null")
print(f"  ({N_PERM} perms, two-sided empirical p, seed={SEED})")
trend_results = {}
for col, label in [("baited_3", "latency_to_3rd_baited"),
                   ("baited_1", "latency_to_1st_baited"),
                   ("baited_2", "latency_to_2nd_baited")]:
    rho, p, nm, lo, hi, n = assay_shuffle_test(df, col)
    trend_results[col] = (rho, p, n)
    if np.isfinite(rho):
        print(f"  {label:22s} rho={rho:+.3f}  p={p:.3f}  "
              f"null={nm:+.3f} [95% {lo:+.3f}, {hi:+.3f}]  n={n}")
    else:
        print(f"  {label:22s} insufficient data (n={n})")


# ===========================================================================
# 6. FINDINGS
# ===========================================================================
print("\n" + "=" * 70)
print("FINDINGS:")

# Successive-site pacing (all-site curve).
reached = [(n, m) for n, m in zip(ns_all, med_all) if np.isfinite(m)]
if reached:
    n1, m1 = reached[0]
    nk, mk = reached[-1]
    print(f"  Groups reach their 1st distinct site (any, 5 m) in a median "
          f"{m1:.1f} min, and progress to the {nk}th distinct site by a "
          f"median {mk:.1f} min; latency rises with site rank.")

# Baited progression.
b_meds = [med_b[n - 1] for n in range(1, N_BAITED + 1)]
n_reach_3 = int(cnt_b[-1]) if len(cnt_b) else 0
if np.isfinite(b_meds[0]):
    msg = (f"  Median latency to the 1st baited site is {b_meds[0]:.1f} min")
    if np.isfinite(b_meds[1]):
        msg += f", 2nd {b_meds[1]:.1f} min"
    if np.isfinite(b_meds[2]):
        msg += f", 3rd {b_meds[2]:.1f} min"
    msg += f" ({n_reach_3}/{len(df)} trials reach all 3 baited sites)."
    print(msg)

# Trend direction.
rho3, p3, n3 = trend_results.get("baited_3", (np.nan, np.nan, 0))
if np.isfinite(rho3):
    if p3 < 0.05:
        direction = "decreases" if rho3 < 0 else "increases"
        print(f"  Latency to the 3rd baited site {direction} significantly with "
              f"assay (Spearman rho={rho3:+.3f}, within-group shuffle p={p3:.3f}, "
              f"n={n3}): groups {'speed up' if rho3 < 0 else 'slow down'} across "
              f"assays.")
    else:
        print(f"  No significant assay trend in latency to the 3rd baited site "
              f"(Spearman rho={rho3:+.3f}, within-group shuffle p={p3:.3f}, "
              f"n={n3}); learning is not detectable in 3rd-site arrival time.")
else:
    print(f"  Too few trials reach all 3 baited sites for a reliable assay trend "
          f"test (n={n3}).")
print("=" * 70)
