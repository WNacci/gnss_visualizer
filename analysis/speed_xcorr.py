#!/usr/bin/env python3
"""Speed cross-correlation leadership analysis.

Who leads changes in SCALAR speed (the magnitude |d(position)/dt|), as
distinct from velocity-VECTOR cross-correlation (leader_follower.py).

Per Phase-2 test trial:
  - interpolate each sheep onto a common 1 Hz time grid,
  - compute scalar speed = |d(position)/dt| per sheep,
  - smooth with a ~15 s moving average,
  - for each ordered pair (i, j) compute the lagged cross-correlation of the
    two speed series over lags +/-30 s. The lag of peak cross-correlation
    indicates who leads speed changes.

Per-trial per-pair asymmetry = peak_xcorr(positive lag) - peak_xcorr(negative
lag). A positive asymmetry for ordered pair (i, j) means i's speed changes lead
j's (i.e. j's speed best matches i's *past*).

Null model: circular-shift each sheep's speed series by a uniform random lag
> 5 min before pairing. This preserves each series' autocorrelation but
destroys cross-sheep alignment. 2000 permutations; two-sided empirical p on the
per-trial mean |asymmetry|.

Also reports, per group, whether a consistent sheep leads speed across trials
(net leadership score = mean over partners of asymmetry where the sheep is the
leading member of the ordered pair).

Outputs a figure to analysis/figures/ and prints a FINDINGS block.

Safe to run repeatedly.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from gps_analysis import (
    build_trials,
    build_tracks_cache,
    load_trial_tracks,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

TEST_CONFIGS = {"A", "B", "C", "D"}
PHASE2_START = "2026-02-17"
_CTRL_GROUPS = {9, 14}

DUR_MIN = 25            # analysis window (minutes)
FS_HZ = 1.0            # common time-grid sample rate (1 Hz)
SMOOTH_S = 15          # speed smoothing window (seconds)
MAX_LAG_S = 30         # cross-correlation lag range (+/- seconds)
MIN_SHIFT_S = 300     # null circular-shift minimum (> 5 minutes)
N_PERM = 2000
SEED = 42

# Grid units: 1 unit = 10 m, t in minutes -> convert speed to m/s for reporting.
GRID_TO_M = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interp_to_grid(tracks, dur_min=DUR_MIN, fs_hz=FS_HZ):
    """Interpolate each sheep onto a common regular time grid (1 Hz)."""
    dt = 1.0 / fs_hz / 60.0  # minutes per sample
    t_grid = np.arange(0, dur_min, dt)
    result = {}
    for sid, trk in tracks.items():
        t = np.asarray(trk["t"], dtype=float)
        gx = np.asarray(trk["gx"], dtype=float)
        gy = np.asarray(trk["gy"], dtype=float)
        if t.size < 2:
            continue
        order = np.argsort(t)
        t, gx, gy = t[order], gx[order], gy[order]
        mask = t_grid <= t.max()
        tg = t_grid[mask]
        if tg.size < 2:
            continue
        result[sid] = {
            "gx": np.interp(tg, t, gx),
            "gy": np.interp(tg, t, gy),
            "t": tg,
        }
    return result


def _scalar_speed(gx, gy, fs_hz=FS_HZ, smooth_s=SMOOTH_S):
    """Scalar speed |d(position)/dt| in m/s, smoothed with a moving average."""
    # np.gradient gives per-sample displacement; dt = 1/fs seconds.
    dt_s = 1.0 / fs_hz
    dgx = np.gradient(gx) * GRID_TO_M / dt_s
    dgy = np.gradient(gy) * GRID_TO_M / dt_s
    speed = np.sqrt(dgx ** 2 + dgy ** 2)
    win = max(1, int(round(smooth_s * fs_hz)))
    if win > 1 and speed.size >= win:
        kernel = np.ones(win) / win
        speed = np.convolve(speed, kernel, mode="same")
    return speed


def _xcorr_peaks(a, b, max_lag):
    """Lagged Pearson cross-correlation of equal-length series a, b.

    Returns (lags, corr) over lags in [-max_lag, +max_lag] (in samples).
    Positive lag k correlates a[t] with b[t+k]: a leads b at positive lag.

    Each lag is normalised by the full-series std and length so that peaks are
    directly comparable across lags (correlation-coefficient scale).
    """
    a = a - a.mean()
    b = b - b.mean()
    sa = a.std()
    sb = b.std()
    n = a.size
    lags = np.arange(-max_lag, max_lag + 1)
    if sa < 1e-12 or sb < 1e-12:
        return lags, np.full(lags.size, np.nan)
    denom = sa * sb * n
    # corr[k] = (1/denom) * sum_t a[t] * b[t+k] for k in [-max_lag, max_lag].
    # Compute only the needed lags directly (cheaper than a full correlation).
    corr = np.empty(lags.size)
    for idx, k in enumerate(lags):
        if k >= 0:
            corr[idx] = np.dot(a[: n - k] if k > 0 else a, b[k:]) / denom
        else:
            corr[idx] = np.dot(a[-k:], b[: n + k]) / denom
    return lags, corr


def _pair_asymmetry(a, b, max_lag):
    """Asymmetry for ordered pair: peak corr at positive lag - at negative lag.

    Positive lag => a leads b. So a large positive asymmetry means a's speed
    changes precede b's.
    """
    lags, corr = _xcorr_peaks(a, b, max_lag)
    if np.all(np.isnan(corr)):
        return np.nan
    pos = corr[lags > 0]
    neg = corr[lags < 0]
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    return float(np.nanmax(pos) - np.nanmax(neg))


def _circular_shift(x, shift):
    return np.roll(x, shift)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def main():
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

    rng = np.random.default_rng(SEED)
    max_lag = int(round(MAX_LAG_S * FS_HZ))
    min_shift = int(round(MIN_SHIFT_S * FS_HZ))

    trial_records = []        # per-trial summary
    # group_num -> sheep_id -> list of net leadership scores (one per trial)
    group_leader = {}

    n_skipped = 0
    for trial in test_trials:
        tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
        if not tracks or len(tracks) < 2:
            n_skipped += 1
            continue

        interp = _interp_to_grid(tracks)
        sids = sorted(interp.keys())
        if len(sids) < 2:
            n_skipped += 1
            continue

        T = min(len(interp[s]["gx"]) for s in sids)
        if T <= 2 * max_lag + 2:
            n_skipped += 1
            continue
        if T <= min_shift:
            # Need a series long enough to allow a >5 min circular shift.
            n_skipped += 1
            continue

        speeds = {
            s: _scalar_speed(interp[s]["gx"][:T], interp[s]["gy"][:T]) for s in sids
        }
        # Drop sheep with degenerate (flat) speed series.
        sids = [s for s in sids if speeds[s].std() > 1e-9]
        if len(sids) < 2:
            n_skipped += 1
            continue
        n_sheep = len(sids)

        # Observed: asymmetry for every ordered pair, plus per-sheep net score.
        asym = {}
        for i in sids:
            for j in sids:
                if i == j:
                    continue
                asym[(i, j)] = _pair_asymmetry(speeds[i], speeds[j], max_lag)

        obs_abs = [abs(v) for v in asym.values() if not np.isnan(v)]
        if not obs_abs:
            n_skipped += 1
            continue
        obs_stat = float(np.mean(obs_abs))

        # Per-sheep net leadership: mean asymmetry over ordered pairs where the
        # sheep is the (leading) first member.
        net_score = {}
        for s in sids:
            vals = [asym[(s, j)] for j in sids if j != s and not np.isnan(asym[(s, j)])]
            net_score[s] = float(np.mean(vals)) if vals else np.nan

        # Null distribution: independently circular-shift each sheep's speed by
        # a uniform random lag > 5 min, then recompute mean |asymmetry|.
        # asymmetry is antisymmetric (asym(j,i) = -asym(i,j)), so |asym| is the
        # same for both orderings; iterate unordered pairs only.
        upairs = [
            (sids[i], sids[j])
            for i in range(n_sheep)
            for j in range(i + 1, n_sheep)
        ]
        null_stats = np.empty(N_PERM)
        for p in range(N_PERM):
            shifted = {
                s: _circular_shift(speeds[s], int(rng.integers(min_shift, T)))
                for s in sids
            }
            vals = []
            for i, j in upairs:
                a = _pair_asymmetry(shifted[i], shifted[j], max_lag)
                if not np.isnan(a):
                    vals.append(abs(a))
            null_stats[p] = np.mean(vals) if vals else np.nan

        null_valid = null_stats[~np.isnan(null_stats)]
        if null_valid.size == 0:
            n_skipped += 1
            continue
        null_mean = float(null_valid.mean())
        # Two-sided empirical p: how extreme is |obs - null_mean| vs the null.
        centered_null = np.abs(null_valid - null_mean)
        centered_obs = abs(obs_stat - null_mean)
        p_val = float((np.sum(centered_null >= centered_obs) + 1) / (null_valid.size + 1))

        trial_records.append(
            {
                "group_num": trial["group_num"],
                "assay": trial["assay"],
                "n_sheep": n_sheep,
                "obs_stat": obs_stat,
                "null_mean": null_mean,
                "null_lo": float(np.percentile(null_valid, 2.5)),
                "null_hi": float(np.percentile(null_valid, 97.5)),
                "p_val": p_val,
            }
        )

        gl = group_leader.setdefault(trial["group_num"], {})
        for s, sc in net_score.items():
            if not np.isnan(sc):
                gl.setdefault(s, []).append(sc)

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    n_trials = len(trial_records)
    if n_trials == 0:
        print("\nFINDINGS: no usable trials after filtering; analysis degenerate.")
        return

    obs_arr = np.array([r["obs_stat"] for r in trial_records])
    null_arr = np.array([r["null_mean"] for r in trial_records])
    p_arr = np.array([r["p_val"] for r in trial_records])

    # Across-trial summary: mean observed vs mean null, fraction significant.
    mean_obs = float(obs_arr.mean())
    mean_null = float(null_arr.mean())
    frac_sig = float(np.mean(p_arr < 0.05))
    # Combine per-trial p-values (Fisher) for a group-level statement.
    eps = 1e-12
    fisher_stat = -2.0 * np.sum(np.log(np.clip(p_arr, eps, 1.0)))
    from scipy import stats as _sp

    fisher_p = float(_sp.chi2.sf(fisher_stat, df=2 * n_trials))

    # Per-group consistent leader: sheep with highest mean net score across
    # trials, and how consistently it ranks #1.
    print("\nPer-group consistent speed leaders:")
    group_lines = []
    for g in sorted(group_leader.keys()):
        gl = group_leader[g]
        # Require a sheep to appear in >=2 trials for a "consistent" claim.
        means = {s: np.mean(v) for s, v in gl.items() if len(v) >= 2}
        n_obs = {s: len(v) for s, v in gl.items()}
        if not means:
            continue
        top = max(means, key=means.get)
        line = (
            f"  Group {g:2d}: top speed leader = {top} "
            f"(mean net asym = {means[top]:+.4f}, n_trials = {n_obs[top]})"
        )
        print(line)
        group_lines.append((g, top, means[top], n_obs[top]))

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.scatter(
        null_arr, obs_arr, s=22, alpha=0.6, color="#3B7DD8", edgecolors="none"
    )
    lim = max(obs_arr.max(), null_arr.max()) * 1.05
    ax1.plot([0, lim], [0, lim], color="#888", ls="--", lw=0.8)
    ax1.set_xlabel("Null mean |asymmetry|")
    ax1.set_ylabel("Observed mean |asymmetry|")
    ax1.set_title("a  Observed vs null per trial")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)

    ax2.hist(p_arr, bins=20, range=(0, 1), color="#E8823A", alpha=0.8)
    ax2.axvline(0.05, color="#D64550", ls="--", lw=0.8, label="p = 0.05")
    ax2.set_xlabel("Per-trial two-sided empirical p")
    ax2.set_ylabel("Number of trials")
    ax2.set_title("b  Distribution of per-trial p-values")
    ax2.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(FIGDIR / "speed_xcorr_leadership.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGDIR / "speed_xcorr_leadership.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> figures/speed_xcorr_leadership.png/.pdf")

    # -----------------------------------------------------------------------
    # FINDINGS
    # -----------------------------------------------------------------------
    n_consistent = 0
    for g, top, m, nt in group_lines:
        gl = group_leader[g]
        # consistent if the top sheep's mean net score is positive and it leads
        # in a majority of its trials.
        scores = gl[top]
        if m > 0 and np.mean(np.array(scores) > 0) >= 0.5:
            n_consistent += 1

    print("\n" + "=" * 70)
    print("FINDINGS: Speed cross-correlation leadership (scalar speed)")
    print("=" * 70)
    print(f"  Usable trials: {n_trials}  (skipped {n_skipped})")
    print(
        f"  Mean observed mean|asymmetry| = {mean_obs:.4f}; "
        f"mean null = {mean_null:.4f}"
    )
    print(
        f"  Per-trial significant (p<0.05): {frac_sig*100:.1f}% "
        f"({int(round(frac_sig*n_trials))}/{n_trials})"
    )
    print(
        f"  Combined (Fisher) across trials: chi2 = {fisher_stat:.1f}, "
        f"df = {2*n_trials}, p = {fisher_p:.2e}"
    )
    print(
        f"  Groups with a consistent speed leader (positive net asym, "
        f"leads in >=50% of trials): {n_consistent}/{len(group_lines)}"
    )
    direction = (
        "evidence FOR directional speed leadership beyond shared group motion"
        if (frac_sig > 0.1 and mean_obs > mean_null)
        else "no clear directional speed leadership beyond shared group motion"
    )
    print(f"  Interpretation: {direction}.")
    print("=" * 70)


if __name__ == "__main__":
    main()
