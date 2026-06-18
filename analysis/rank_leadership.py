#!/usr/bin/env python3
"""Rank-based leadership ("following order") analysis for sheep GPS trials.

A "following order" metric for Phase-2 test trials: for each visited reward
site, sheep are ranked by first-arrival time (rank 1 = pioneer). Each sheep's
arrival rank is averaged across the sites it visited in a trial; a low average
rank marks a consistent leader.

The script then:
  (a) reports the distribution of per-sheep average arrival rank,
  (b) tests whether average rank is non-uniform within groups via a
      permutation null that shuffles arrival order within each site
      (statistic = variance of per-sheep mean rank vs. null),
  (c) correlates rank-based leadership with frontal-leadership share
      (centroid-velocity argmax projection, as in generate_figures.py §4),
  (d) measures rank consistency across trials within a group (Spearman of
      per-sheep mean rank between assay pairs).

Plain headless script (like analysis/generate_figures.py); prints a FINDINGS
block. Safe to run repeatedly.
"""
import numpy as np
import pandas as pd
from scipy import stats

from gps_analysis import (
    build_trials,
    build_tracks_cache,
    load_trial_tracks,
    detect_site_visits,
)

# ---------------------------------------------------------------------------
# Parameters (mirror generate_figures.py where shared)
# ---------------------------------------------------------------------------
RADIUS = 0.5            # 0.5 grid units = 5 m (site arrival, per spec)
MIN_DWELL_S = 0.0
TEST_CONFIGS = {"A", "B", "C", "D"}
PHASE2_START = "2026-02-17"
CTRL_GROUPS = {9, 14}
SMOOTH_WIN = 15
SPEED_THRESH = 0.000833  # 5 m/min at 10 Hz (gradient units: gu/sample)
N_PERM = 2000
SEED = 42

RNG = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interp_to_1s(tracks, dur_min=25):
    """Interpolate per-sheep tracks onto a shared 10 Hz grid (matches §4)."""
    t_grid = np.arange(0, dur_min, 1 / 600)
    result = {}
    for sid, trk in tracks.items():
        order = np.argsort(trk["t"])
        t_s, gx_s, gy_s = trk["t"][order], trk["gx"][order], trk["gy"][order]
        if len(t_s) == 0:
            continue
        mask = t_grid <= t_s.max()
        tg = t_grid[mask]
        if len(tg) == 0:
            continue
        result[sid] = {
            "gx": np.interp(tg, t_s, gx_s),
            "gy": np.interp(tg, t_s, gy_s),
            "t": tg,
        }
    return result


def _frontal_shares(tracks):
    """Per-sheep frontal-leadership share via centroid-velocity argmax.

    Replicates generate_figures.py §4 LEADERSHIP: project each sheep onto the
    moving frame's forward axis; the frontmost sheep while the group moves is
    the instantaneous leader. Returns {sheep_id: fraction of moving frames led}.
    """
    interp = _interp_to_1s(tracks)
    sids = sorted(interp.keys())
    n_sheep = len(sids)
    if n_sheep < 2:
        return {}
    T = min(len(interp[s]["gx"]) for s in sids)
    if T < 2:
        return {}

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
        return {}
    shares = {}
    for i, sid in enumerate(sids):
        shares[sid] = float(np.sum((leader_idx == i) & moving)) / total_moving
    return shares


def _arrival_ranks(visits):
    """Per-site arrival ranks from site visits.

    Returns a list of dicts, one per visited site, mapping sheep_id -> rank
    (1 = first to arrive). Each sheep is counted once per site, by its first
    entry time. Sites visited by a single sheep are kept (rank 1). Ties get
    the average rank.
    """
    site_ranks = []
    for site, vlist in visits.items():
        if not vlist:
            continue
        first_entry = {}
        for sid, entry_min, _exit_min in vlist:
            if sid not in first_entry or entry_min < first_entry[sid]:
                first_entry[sid] = entry_min
        ids = list(first_entry.keys())
        times = np.array([first_entry[s] for s in ids])
        ranks = stats.rankdata(times, method="average")
        site_ranks.append({s: float(r) for s, r in zip(ids, ranks)})
    return site_ranks


def _mean_rank_per_sheep(site_ranks):
    """Average arrival rank per sheep across sites it visited."""
    acc = {}
    for sr in site_ranks:
        for sid, r in sr.items():
            acc.setdefault(sid, []).append(r)
    return {sid: float(np.mean(rs)) for sid, rs in acc.items()}


def _perm_variance_p(site_ranks, observed_means):
    """Permutation test: is the variance of per-sheep mean rank non-uniform?

    Null shuffles arrival order within each site (re-assigns the rank vector to
    the participating sheep at random), preserving each site's rank multiset and
    each sheep's site-participation. Statistic = variance of per-sheep mean rank.

    Returns (observed_stat, two_sided_p, (null_mean, (lo95, hi95))).
    """
    sheep = sorted(observed_means.keys())
    if len(sheep) < 2 or len(site_ranks) == 0:
        return np.nan, np.nan, (np.nan, (np.nan, np.nan))
    obs_stat = float(np.var(list(observed_means.values())))

    site_data = []
    for sr in site_ranks:
        ids = list(sr.keys())
        vals = np.array([sr[s] for s in ids])
        site_data.append((ids, vals))

    null = np.empty(N_PERM)
    for k in range(N_PERM):
        acc = {}
        for ids, vals in site_data:
            shuffled = RNG.permutation(vals)
            for s, r in zip(ids, shuffled):
                acc.setdefault(s, []).append(r)
        means = [np.mean(acc[s]) for s in sheep if s in acc]
        null[k] = np.var(means)

    null_mean = float(null.mean())
    p = float(np.mean(np.abs(null - null_mean) >= abs(obs_stat - null_mean)))
    lo, hi = np.percentile(null, [2.5, 97.5])
    return obs_stat, p, (null_mean, (float(lo), float(hi)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data (trials + tracks cache ~687MB)...")
    trials = build_trials()
    tracks_cache = build_tracks_cache(trials)

    test_trials = [
        t for t in trials
        if t["config"] in TEST_CONFIGS
        and isinstance(t["assay"], int)
        and t["date"] >= PHASE2_START
        and t["group_num"] not in CTRL_GROUPS
    ]
    print(f"Total trials: {len(trials)}, Phase 2 test trials: {len(test_trials)}")

    rows = []
    per_trial = []
    n_trials_used = 0
    n_skipped = 0

    for trial in test_trials:
        tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
        if not tracks or len(tracks) < 2:
            n_skipped += 1
            continue

        visits = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)
        site_ranks = _arrival_ranks(visits)
        if not site_ranks:
            n_skipped += 1
            continue

        mean_ranks = _mean_rank_per_sheep(site_ranks)
        if len(mean_ranks) < 2:
            n_skipped += 1
            continue

        frontal = _frontal_shares(tracks)
        n_trials_used += 1
        per_trial.append({
            "group_num": trial["group_num"],
            "assay": trial["assay"],
            "site_ranks": site_ranks,
            "mean_ranks": mean_ranks,
        })
        for sid, mr in mean_ranks.items():
            rows.append({
                "group_num": trial["group_num"],
                "assay": trial["assay"],
                "trial_name": trial["name"],
                "sheep_id": sid,
                "mean_rank": mr,
                "n_sites": sum(1 for sr in site_ranks if sid in sr),
                "frontal_share": frontal.get(sid, np.nan),
                "group_size_obs": len(mean_ranks),
            })

    df = pd.DataFrame(rows)
    print(f"Trials used: {n_trials_used}, skipped: {n_skipped}, "
          f"per-sheep records: {len(df)}")

    if df.empty:
        print("\nFINDINGS:\n  No usable trials — analysis degenerate.")
        return

    # -----------------------------------------------------------------------
    # (a) Distribution of per-sheep average rank
    # -----------------------------------------------------------------------
    mr = df["mean_rank"].to_numpy()
    a_n = len(mr)
    a_mean = float(np.mean(mr))
    a_med = float(np.median(mr))
    a_q = np.percentile(mr, [25, 75])

    print("\n(a) Per-sheep average arrival rank distribution:")
    print(f"    n={a_n} (trial,sheep) records  mean={a_mean:.2f}  "
          f"median={a_med:.2f}  IQR=[{a_q[0]:.2f}, {a_q[1]:.2f}]")

    # -----------------------------------------------------------------------
    # (b) Permutation test: non-uniform mean rank within groups.
    #     Pool all per-sheep mean ranks within each group (across trials) and
    #     test their variance against a within-site shuffled null.
    # -----------------------------------------------------------------------
    print("\n(b) Permutation test (arrival order shuffled within each site):")
    pooled_site_ranks = []
    for pt in per_trial:
        g = pt["group_num"]
        for sr in pt["site_ranks"]:
            pooled_site_ranks.append({(g, s): r for s, r in sr.items()})
    pooled_means = _mean_rank_per_sheep(pooled_site_ranks)
    grp_obs_stat, grp_p, (grp_nmean, (grp_lo, grp_hi)) = _perm_variance_p(
        pooled_site_ranks, pooled_means)

    if np.isfinite(grp_obs_stat):
        print(f"    Pooled within-group test (n={len(pooled_means)} group-sheep):")
        print(f"      observed var(mean rank) = {grp_obs_stat:.4f}")
        print(f"      null mean = {grp_nmean:.4f}  95% CI [{grp_lo:.4f}, {grp_hi:.4f}]")
        print(f"      two-sided empirical p = {grp_p:.4f}")

    # Per-trial tests for context.
    trial_p = []
    for pt in per_trial:
        obs_stat, p, _ = _perm_variance_p(pt["site_ranks"], pt["mean_ranks"])
        if np.isfinite(p):
            trial_p.append(p)
    if trial_p:
        frac_sig = float(np.mean(np.array(trial_p) < 0.05))
        print(f"    Per-trial tests: {len(trial_p)} trials, "
              f"{frac_sig*100:.0f}% with p<0.05; median p = {np.median(trial_p):.3f}")

    # -----------------------------------------------------------------------
    # (c) Rank leadership vs frontal-leadership share.
    #     Low mean rank = strong arrival leader; high frontal share = strong
    #     frontal leader. Agreement => NEGATIVE Spearman rho.
    # -----------------------------------------------------------------------
    print("\n(c) Rank leadership vs frontal-leadership share:")
    cc = df.dropna(subset=["frontal_share"])
    c_rho = c_p = np.nan
    if len(cc) >= 3:
        c_rho, c_p = stats.spearmanr(cc["mean_rank"], cc["frontal_share"])
        print(f"    (trial,sheep) level: Spearman rho = {c_rho:+.3f}, "
              f"p = {c_p:.4f}, n = {len(cc)}")

    sheep_pool = (cc.groupby("sheep_id")
                    .agg(mean_rank=("mean_rank", "mean"),
                         frontal_share=("frontal_share", "mean"),
                         n=("mean_rank", "size"))
                    .reset_index())
    cs_rho = cs_p = np.nan
    if len(sheep_pool) >= 3:
        cs_rho, cs_p = stats.spearmanr(sheep_pool["mean_rank"],
                                       sheep_pool["frontal_share"])
        print(f"    per-sheep pooled:    Spearman rho = {cs_rho:+.3f}, "
              f"p = {cs_p:.4f}, n = {len(sheep_pool)} sheep")

    # -----------------------------------------------------------------------
    # (d) Rank consistency across trials within a group: Spearman of per-sheep
    #     mean rank between all trial pairs in a group with >=3 shared sheep.
    # -----------------------------------------------------------------------
    print("\n(d) Rank consistency across trials within group (trial-pair Spearman):")
    pair_rhos = []
    for g, gdf in df.groupby("group_num"):
        trial_keys = gdf["trial_name"].unique()
        for i in range(len(trial_keys)):
            for j in range(i + 1, len(trial_keys)):
                a = gdf[gdf["trial_name"] == trial_keys[i]].set_index("sheep_id")["mean_rank"]
                b = gdf[gdf["trial_name"] == trial_keys[j]].set_index("sheep_id")["mean_rank"]
                common = a.index.intersection(b.index)
                if len(common) >= 3:
                    rho, _ = stats.spearmanr(a.loc[common], b.loc[common])
                    if np.isfinite(rho):
                        pair_rhos.append(rho)
    pair_rhos = np.array(pair_rhos)
    d_mean = d_med = d_p = np.nan
    if len(pair_rhos):
        d_mean = float(np.mean(pair_rhos))
        d_med = float(np.median(pair_rhos))
        if len(pair_rhos) >= 6 and np.any(pair_rhos != 0):
            try:
                _, d_p = stats.wilcoxon(pair_rhos, alternative="greater")
            except ValueError:
                d_p = np.nan
        print(f"    {len(pair_rhos)} trial-pairs (>=3 shared sheep), "
              f"mean rho = {d_mean:+.3f}, median = {d_med:+.3f}")
        if np.isfinite(d_p):
            print(f"    Wilcoxon (rho>0) p = {d_p:.4f}")
    else:
        print("    No trial-pairs with >=3 shared sheep; consistency undefined.")

    # -----------------------------------------------------------------------
    # FINDINGS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINDINGS:")
    print(f"  Phase-2 trials analysed: {n_trials_used} (skipped {n_skipped}); "
          f"{a_n} (trial,sheep) rank records.")
    print(f"  (a) Per-sheep mean arrival rank: median {a_med:.2f} "
          f"(IQR [{a_q[0]:.2f}, {a_q[1]:.2f}]).")
    if np.isfinite(grp_obs_stat):
        nonuniform = ("non-uniform" if (np.isfinite(grp_p) and grp_p < 0.05)
                      else "NOT distinguishable from uniform")
        print(f"  (b) Within-group arrival order is {nonuniform}: "
              f"var(mean rank)={grp_obs_stat:.3f} vs null "
              f"{grp_nmean:.3f} [{grp_lo:.3f}, {grp_hi:.3f}], p={grp_p:.4f}.")
    if np.isfinite(cs_rho):
        agree = ("agree (consistent leaders arrive first)"
                 if (cs_rho < 0 and np.isfinite(cs_p) and cs_p < 0.05)
                 else "do NOT significantly agree")
        print(f"  (c) Rank vs frontal leadership {agree}: per-sheep Spearman "
              f"rho={cs_rho:+.3f}, p={cs_p:.4f} (n={len(sheep_pool)} sheep); "
              f"trial-level rho={c_rho:+.3f}, p={c_p:.4f}.")
        print(f"      (Negative rho expected: low rank = early arrival = leader.)")
    if len(pair_rhos):
        consistent = ("reliable across trials"
                      if (np.isfinite(d_p) and d_p < 0.05)
                      else "weak/uncertain")
        pstr = f"{d_p:.4f}" if np.isfinite(d_p) else "n/a"
        print(f"  (d) Cross-trial rank consistency is {consistent}: "
              f"mean pairwise Spearman rho={d_mean:+.3f} (median {d_med:+.3f}), "
              f"Wilcoxon p={pstr}, {len(pair_rhos)} pairs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
