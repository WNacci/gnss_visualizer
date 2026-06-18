#!/usr/bin/env python3
"""Field A vs Field B comparison of Phase-2 test trials.

Question
--------
Do sheep perform differently between the two physical fields (A vs B)?  Both
fields use an identical reward-site grid, but they sit in different locations
with different distal/field-specific visual cues.  A reliable A-vs-B difference
in search efficiency or baited-site preference would be consistent with sheep
using field-specific landmarks for localisation rather than relying purely on
local cues.

Metrics (per trial, Phase-2 cohort, fields A and B)
---------------------------------------------------
1. Completion time and path-to-completion for the 3 baited sites.
   - completion = max of the three earliest baited-site entry times
     (strict 2 m radius = 0.2 grid units); NaN if not all 3 found.
   - path-to-completion = mean across sheep of cumulative path length
     (x10 -> metres) interpolated at the completion time.
2. Baited-preference fraction = time spent within radius 0.5 of any baited
   canonical site {A1,A2,A3} / time within radius 0.5 of ANY of the 12 sites.
   Uses oriented tracks (apply_orient=True).
3. Top-leader frontal share = fraction of moving frames for which a single
   sheep is the front-most along the centroid velocity (the LEADERSHIP method
   from generate_figures.py section 4).

Statistics
----------
For every metric: Mann-Whitney U (two-sided) plus a label-permutation test
(1000 permutations, seed 42) that shuffles the A/B field labels and recomputes
the difference in medians.  Report observed difference, the permutation null
interval, the two-sided empirical p-value and the MWU p-value, with n per field.
Results are UNCORRECTED and EXPLORATORY.

This is a standalone script (run repeatedly, safe).  It adds no files outside
analysis/figures/ and edits no existing module.
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
    cumulative_path_length,
    SITE_GRID,
    BAITED_CANONICAL,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters (matched to generate_figures.py)
# ---------------------------------------------------------------------------
RADIUS = 0.2          # 0.2 grid units = 2 m (strict, for baited completion)
RADIUS_PREF = 0.5     # 0.5 grid units = 5 m (generous, for preference dwell)
MIN_DWELL_S = 0.0
TEST_CONFIGS = {"A", "B", "C", "D"}
PHASE2_START = "2026-02-17"
_CTRL_GROUPS = {9, 14}
SMOOTH_WIN = 15
SPEED_THRESH = 0.000833  # 5 m/min at 10 Hz (np.gradient units: gu/sample)
N_PERM = 1000
SEED = 42

PALETTE = {"A": "#3B7DD8", "B": "#E8823A"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interp_to_1s(tracks, dur_min=25):
    """Interpolate each sheep track onto a regular 10 Hz grid (matches raw rate)."""
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


def _completion_and_path(tracks, field):
    """Return (completion_time_min, mean_path_to_completion_m).

    completion = max of the three earliest baited-site entry times (2 m);
    NaN if fewer than 3 baited sites are reached.  path-to-completion is the
    mean across sheep of cumulative path length (metres) at the completion time.
    """
    visits = detect_site_visits(tracks, field, RADIUS, MIN_DWELL_S)
    disc_times = {s: min(v[1] for v in vl) for s, vl in visits.items() if vl}
    baited_disc = {s: disc_times[s] for s in BAITED_CANONICAL if s in disc_times}
    if len(baited_disc) < 3:
        return np.nan, np.nan
    completion = max(baited_disc.values())

    paths = []
    for trk in tracks.values():
        if len(trk["t"]) == 0:
            continue
        cpl = cumulative_path_length(trk["gx"], trk["gy"]) * 10.0
        paths.append(float(np.interp(completion, trk["t"], cpl)))
    mean_path = float(np.nanmean(paths)) if paths else np.nan
    return completion, mean_path


def _baited_preference(tracks):
    """Fraction of any-site dwell-samples that fall at a baited canonical site.

    For each track sample, count it as "near a baited site" if it is within
    RADIUS_PREF of any of {A1,A2,A3}, and "near any site" if within RADIUS_PREF
    of any of the 12 canonical sites.  Returns near_baited / near_any over all
    sheep, or NaN if no sample is near any site.
    """
    baited_pos = [SITE_GRID[s] for s in BAITED_CANONICAL]
    all_pos = list(SITE_GRID.values())
    near_baited = 0
    near_any = 0
    for trk in tracks.values():
        gx, gy = trk["gx"], trk["gy"]
        if len(gx) == 0:
            continue
        in_baited = np.zeros(len(gx), dtype=bool)
        for sx, sy in baited_pos:
            in_baited |= (gx - sx) ** 2 + (gy - sy) ** 2 <= RADIUS_PREF ** 2
        in_any = np.zeros(len(gx), dtype=bool)
        for sx, sy in all_pos:
            in_any |= (gx - sx) ** 2 + (gy - sy) ** 2 <= RADIUS_PREF ** 2
        near_baited += int(in_baited.sum())
        near_any += int(in_any.sum())
    if near_any == 0:
        return np.nan
    return near_baited / near_any


def _top_leader_share(tracks):
    """Fraction of moving frames where a single sheep is the front-most leader.

    Mirrors the LEADERSHIP computation in generate_figures.py section 4:
    smooth centroid velocity, mark moving frames, project each sheep onto the
    unit velocity direction, take the front-most (argmax) per frame, and return
    the dominant sheep's share of all moving frames.  NaN if <2 sheep or no
    motion.
    """
    interp = _interp_to_1s(tracks)
    if len(interp) < 2:
        return np.nan
    sids = sorted(interp.keys())
    n_sheep = len(sids)
    T = min(len(interp[s]["gx"]) for s in sids)
    if T < 2:
        return np.nan

    GX = np.column_stack([interp[s]["gx"][:T] for s in sids])
    GY = np.column_stack([interp[s]["gy"][:T] for s in sids])
    cx, cy = GX.mean(axis=1), GY.mean(axis=1)

    kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    vx = np.convolve(np.gradient(cx), kernel, mode="same")
    vy = np.convolve(np.gradient(cy), kernel, mode="same")
    speed = np.sqrt(vx ** 2 + vy ** 2)
    moving = speed > SPEED_THRESH
    vnx = np.where(moving, vx / speed, 0.0)
    vny = np.where(moving, vy / speed, 0.0)

    projections = np.zeros((T, n_sheep))
    for i in range(n_sheep):
        projections[:, i] = (GX[:, i] - cx) * vnx + (GY[:, i] - cy) * vny

    leader_idx = np.argmax(projections, axis=1)
    total_moving = int(moving.sum())
    if total_moving == 0:
        return np.nan
    counts = np.array([np.sum((leader_idx == i) & moving) for i in range(n_sheep)])
    return float(counts.max() / total_moving)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _perm_test(vals_a, vals_b, n_perm=N_PERM, seed=SEED):
    """Label-permutation test on the difference of medians (A - B).

    Returns dict with observed diff, null 2.5/97.5 percentile interval,
    two-sided empirical p (|null| >= |observed|), MWU p, and n per field.
    """
    a = np.asarray(vals_a, dtype=float)
    b = np.asarray(vals_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n_a, n_b = len(a), len(b)
    out = {
        "n_a": n_a, "n_b": n_b,
        "median_a": np.nan, "median_b": np.nan,
        "obs_diff": np.nan, "null_lo": np.nan, "null_hi": np.nan,
        "p_perm": np.nan, "p_mwu": np.nan,
    }
    if n_a < 2 or n_b < 2:
        return out

    out["median_a"] = float(np.median(a))
    out["median_b"] = float(np.median(b))
    obs = out["median_a"] - out["median_b"]
    out["obs_diff"] = float(obs)

    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(pooled)
        null[k] = np.median(perm[:n_a]) - np.median(perm[n_a:])
    out["null_lo"] = float(np.percentile(null, 2.5))
    out["null_hi"] = float(np.percentile(null, 97.5))
    # +1 smoothing keeps the empirical p strictly within (0, 1].
    out["p_perm"] = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    out["p_mwu"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return out


def _report(name, unit, res):
    """Pretty-print one metric's A-vs-B comparison."""
    print(f"\n  {name}")
    print(f"    n: A={res['n_a']}  B={res['n_b']}")
    if not np.isfinite(res["obs_diff"]):
        print("    insufficient finite data in one or both fields -> skipped")
        return
    print(f"    median A = {res['median_a']:.3f} {unit}   "
          f"median B = {res['median_b']:.3f} {unit}")
    print(f"    observed diff (A - B) = {res['obs_diff']:+.3f} {unit}")
    print(f"    permutation null 95% interval = "
          f"[{res['null_lo']:+.3f}, {res['null_hi']:+.3f}] {unit}")
    print(f"    permutation p (two-sided) = {res['p_perm']:.3f}")
    print(f"    Mann-Whitney U p (two-sided) = {res['p_mwu']:.3f}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading data (tracks_cache ~687 MB)...")
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

    records = []
    for trial in test_trials:
        field = trial["field"]
        if field not in ("A", "B"):
            continue  # ignore Unknown / control fields
        try:
            tracks = load_trial_tracks(trial, tracks_cache=tracks_cache,
                                       apply_orient=True)
        except Exception as exc:  # robustness: never crash on one bad trial
            print(f"  WARN: skipping {trial['name']}: {exc}")
            continue
        if not tracks or len(tracks) < 2:
            continue

        completion, path_comp = _completion_and_path(tracks, field)
        pref = _baited_preference(tracks)
        leader = _top_leader_share(tracks)

        records.append({
            "name": trial["name"],
            "field": field,
            "group_num": trial["group_num"],
            "assay": trial["assay"],
            "completion_time": completion,
            "path_to_completion": path_comp,
            "baited_pref": pref,
            "top_leader_share": leader,
        })

    df = pd.DataFrame(records)
    n_a = int((df["field"] == "A").sum())
    n_b = int((df["field"] == "B").sum())
    print(f"\nTrials with >=2 sheep: {len(df)}  (Field A: {n_a}, Field B: {n_b})")

    metrics = [
        ("completion_time", "Completion time (min)", "min"),
        ("path_to_completion", "Path to completion (m)", "m"),
        ("baited_pref", "Baited-preference fraction", "frac"),
        ("top_leader_share", "Top-leader frontal share", "frac"),
    ]

    results = {}
    print("\n" + "=" * 70)
    print("FIELD A vs FIELD B  (Mann-Whitney U + label-permutation test)")
    print(f"  {N_PERM} permutations, seed {SEED}; uncorrected / exploratory")
    print("=" * 70)
    for col, label, unit in metrics:
        a_vals = df.loc[df["field"] == "A", col].to_numpy()
        b_vals = df.loc[df["field"] == "B", col].to_numpy()
        res = _perm_test(a_vals, b_vals)
        results[col] = (label, unit, res)
        _report(label, unit, res)

    _make_figure(df, metrics, results)

    # -------------------------------------------------------------------
    # FINDINGS
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    sig = []
    for col, (label, unit, res) in results.items():
        if np.isfinite(res["p_perm"]) and (res["p_perm"] < 0.05 or res["p_mwu"] < 0.05):
            sig.append((label, res))
    print("FINDINGS: Field A vs Field B (Phase-2 test trials)")
    print(f"  Sample sizes: Field A n={n_a}, Field B n={n_b} "
          f"(total {n_a + n_b}).")
    if not sig:
        print("  No metric shows a significant A-vs-B difference "
              "(all permutation and MWU p >= 0.05, uncorrected).")
        print("  -> No detectable behavioural difference between fields; the "
              "data do not support field-specific (distal-cue) localisation.")
    else:
        print("  Significant difference(s) (uncorrected, exploratory):")
        for label, res in sig:
            print(f"    - {label}: median A={res['median_a']:.3f} vs "
                  f"B={res['median_b']:.3f}, diff={res['obs_diff']:+.3f}, "
                  f"perm p={res['p_perm']:.3f}, MWU p={res['p_mwu']:.3f}")
        print("  -> A reliable A-vs-B difference is consistent with sheep using "
              "distal / field-specific cues for localisation.")
    print("  NOTE: p-values are uncorrected and exploratory; many trials never "
          "complete (NaN completion handled per metric).")
    print("=" * 70)


def _make_figure(df, metrics, results):
    """Box plot per metric, Field A vs B, with permutation/MWU p annotations."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.8))
    for ax, (col, label, unit) in zip(np.atleast_1d(axes), metrics):
        a = df.loc[df["field"] == "A", col].dropna().to_numpy()
        b = df.loc[df["field"] == "B", col].dropna().to_numpy()
        data = [a, b]
        bp = ax.boxplot(data, tick_labels=["A", "B"], patch_artist=True,
                        widths=0.55, showfliers=False,
                        medianprops=dict(color="k", lw=1.2))
        for box, fld in zip(bp["boxes"], ["A", "B"]):
            box.set_facecolor(PALETTE[fld])
            box.set_alpha(0.35)
            box.set_edgecolor(PALETTE[fld])
        for i, (d, fld) in enumerate(zip(data, ["A", "B"])):
            if len(d) == 0:
                continue
            jit = np.random.default_rng(SEED + i).uniform(-0.15, 0.15, len(d))
            ax.scatter(np.full(len(d), i + 1) + jit, d, s=12, alpha=0.55,
                       color=PALETTE[fld], edgecolors="none", zorder=3)
            ax.text(i + 1, 0.01, f"n={len(d)}", ha="center", va="bottom",
                    fontsize=6, color="#777", transform=ax.get_xaxis_transform())
        ax.set_ylabel(f"{label}")
        ax.set_xlabel("Field")
        ax.set_title(label, fontsize=9)
        res = results[col][2]
        if np.isfinite(res["p_perm"]):
            ax.text(0.97, 0.97,
                    f"perm p={res['p_perm']:.3f}\nMWU p={res['p_mwu']:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color="#333", style="italic")
    fig.suptitle("Field A vs Field B (Phase-2 test trials, exploratory)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "field_comparison.pdf")
    fig.savefig(FIGDIR / "field_comparison.png")
    plt.close(fig)
    print(f"\n  -> figure: {FIGDIR / 'field_comparison.png'}")


if __name__ == "__main__":
    main()
