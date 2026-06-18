#!/usr/bin/env python3
"""Multi-timescale group cohesion analysis.

Complements scripts/flocking_dynamics.py (single-timescale NND/spread time
series + early-vs-late) by asking *how cohesion behaves across temporal
resolutions* and *how long cohesion states persist*.

Per Phase-2 test trial we compute, at every common-grid timestep (~1 s):
  - NND     : nearest-neighbour distance = min pairwise Euclidean distance (m)
  - spread  : mean distance of sheep to the group centroid (m)

We then ask two questions:

(a) Multi-resolution variance/mean.
    Aggregate the raw (~1 s) NND/spread series to 10 s and 60 s windows
    (block-mean) and report how the per-trial mean and standard deviation of
    each metric change with timescale. Window-averaging leaves the mean
    essentially unchanged but smooths fast fluctuations, so the SD falls as
    the window grows; the *rate* of that fall encodes how much cohesion
    variance lives at sub-window timescales.

(b) Autocorrelation persistence timescale.
    For the raw series we compute the autocorrelation function (ACF) of NND
    and spread and report tau_e = the lag at which the ACF first crosses 1/e.
    This is "how long a cohesion state persists".

Everything is aggregated by assay and by config. A within-group assay-shuffle
permutation null (1000 perms, seed 42) tests whether the cohesion timescale or
magnitude trends with assay — i.e. whether learning reshapes group structure
or only navigation.

Plain script (run: ``uv run python analysis/cohesion_timescales.py``); writes
figures to analysis/figures/ and prints a FINDINGS block. Safe to re-run.
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
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

GRID_TO_M = 10.0            # 1 grid unit = 10 m
RAW_DT_MIN = 1 / 60.0       # common grid resolution: ~1 s (in minutes)
TIMESCALES_S = [1, 10, 60]  # resolutions to compare (seconds)
PHASE2_START = "2026-02-17"
TEST_CONFIGS = {"A", "B", "C", "D"}
CTRL_GROUPS = {9, 14}       # control groups excluded from Phase-2 test cohort
N_PERM = 1000
SEED = 42
ACF_INV_E = 1.0 / np.e

PALETTE = {
    "blue": "#3B7DD8", "orange": "#E8823A", "green": "#4AAD5B",
    "purple": "#8B6DAF", "red": "#D64550", "grey": "#888888",
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Per-trial cohesion series at raw (~1 s) resolution
# ---------------------------------------------------------------------------
def compute_raw_cohesion(trial, tracks_cache):
    """Return raw-resolution NND & spread series (metres) for one trial.

    Returns None when fewer than 2 sheep have usable tracks (NND undefined)
    or when no overlapping time samples exist.
    """
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    # Keep only sheep with non-empty tracks.
    tracks = {sid: trk for sid, trk in tracks.items() if len(trk["t"]) >= 2}
    if len(tracks) < 2:
        return None

    sheep_ids = sorted(tracks.keys())
    n_sheep = len(sheep_ids)

    # Common time grid spanning the overlap of all sheep (so interpolation never
    # extrapolates past where an animal has data).
    t_min = max(float(tracks[s]["t"].min()) for s in sheep_ids)
    t_max = min(float(tracks[s]["t"].max()) for s in sheep_ids)
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max - t_min < RAW_DT_MIN:
        return None
    t_common = np.arange(t_min, t_max + RAW_DT_MIN, RAW_DT_MIN)
    n_t = len(t_common)
    if n_t < 2:
        return None

    gx = np.empty((n_sheep, n_t))
    gy = np.empty((n_sheep, n_t))
    for ci, sid in enumerate(sheep_ids):
        trk = tracks[sid]
        order = np.argsort(trk["t"])
        gx[ci] = np.interp(t_common, trk["t"][order], trk["gx"][order])
        gy[ci] = np.interp(t_common, trk["t"][order], trk["gy"][order])

    # Spread: mean distance to centroid, in metres.
    cx = gx.mean(axis=0)
    cy = gy.mean(axis=0)
    spread = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2).mean(axis=0) * GRID_TO_M

    # NND: min pairwise distance at each timestep, in metres.
    pw = []
    for i in range(n_sheep):
        for j in range(i + 1, n_sheep):
            pw.append(np.sqrt((gx[i] - gx[j]) ** 2 + (gy[i] - gy[j]) ** 2))
    nnd = np.min(np.asarray(pw), axis=0) * GRID_TO_M

    return {
        "t": t_common,
        "nnd": nnd,
        "spread": spread,
        "n_sheep": n_sheep,
        "dt_min": RAW_DT_MIN,
    }


# ---------------------------------------------------------------------------
# Timescale aggregation & autocorrelation
# ---------------------------------------------------------------------------
def block_aggregate(series, dt_min, window_s):
    """Block-mean a series to a coarser ``window_s`` resolution.

    Drops a trailing partial block so every output point averages a full
    window. Returns the raw series unchanged when the window is <= the
    native resolution.
    """
    block = max(int(round(window_s / 60.0 / dt_min)), 1)
    if block <= 1:
        return series.copy()
    n_full = len(series) // block
    if n_full < 1:
        return np.array([series.mean()])
    return series[: n_full * block].reshape(n_full, block).mean(axis=1)


def acf_crossing_timescale(series, dt_min):
    """Lag (in seconds) at which the ACF of ``series`` first crosses 1/e.

    Uses the biased (1/N-normalised) autocorrelation, which decays to zero and
    so always yields a finite crossing for a stationary-ish, mean-reverting
    signal. Returns np.nan if the series is too short or has ~zero variance
    (a degenerate, perfectly cohesive trial). Linear interpolation between the
    two bracketing lags gives a sub-step estimate.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 4:
        return np.nan
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var <= 1e-12:
        return np.nan
    # Full autocorrelation via direct correlate (series are short).
    acov = np.correlate(x, x, mode="full")[n - 1:] / n
    acf = acov / acov[0]
    # First lag where ACF drops below 1/e.
    below = np.flatnonzero(acf < ACF_INV_E)
    if below.size == 0:
        return np.nan  # never decorrelates within the observed window
    k = below[0]
    if k == 0:
        return 0.0
    # Linear interpolation between lag k-1 (>=1/e) and lag k (<1/e).
    a0, a1 = acf[k - 1], acf[k]
    frac = (a0 - ACF_INV_E) / (a0 - a1) if a0 != a1 else 0.0
    lag_steps = (k - 1) + frac
    return lag_steps * dt_min * 60.0  # seconds


# ---------------------------------------------------------------------------
# Within-group assay-shuffle permutation null
# ---------------------------------------------------------------------------
def assay_shuffle_p(df, value_col, n_perm=N_PERM, seed=SEED):
    """Two-sided empirical p for a Spearman trend of value vs assay.

    Assay labels are permuted *within* each group_num (preserving group
    identity), matching the convention used elsewhere in the repo.
    Returns (rho_obs, p, n).
    """
    d = df.dropna(subset=[value_col]).reset_index(drop=True)
    if len(d) < 3 or d["assay"].nunique() < 2:
        return np.nan, np.nan, len(d)
    assay = d["assay"].to_numpy(dtype=float)
    y = d[value_col].to_numpy(dtype=float)
    rho_obs = stats.spearmanr(assay, y).statistic
    if not np.isfinite(rho_obs):
        return np.nan, np.nan, len(d)
    rng = np.random.default_rng(seed)
    groups = [np.asarray(idx) for idx in d.groupby("group_num").indices.values()]
    null = np.empty(n_perm)
    for k in range(n_perm):
        shuf = assay.copy()
        for idx in groups:
            shuf[idx] = assay[rng.permutation(idx)]
        null[k] = stats.spearmanr(shuf, y).statistic
    p = float(np.mean(np.abs(null) >= abs(rho_obs)))
    return float(rho_obs), p, len(d)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading trials & tracks cache (data/tracks_cache.pkl ~687MB)…")
    trials = build_trials()
    tracks_cache = build_tracks_cache(trials)

    test_trials = [
        t for t in trials
        if t["config"] in TEST_CONFIGS
        and isinstance(t["assay"], int)
        and t["date"] >= PHASE2_START
        and t["group_num"] not in CTRL_GROUPS
    ]
    print(f"Phase-2 test trials matching cohort filter: {len(test_trials)}")

    records = []
    skipped = 0
    for trial in test_trials:
        c = compute_raw_cohesion(trial, tracks_cache)
        if c is None:
            skipped += 1
            continue

        rec = {
            "name": trial["name"],
            "assay": trial["assay"],
            "config": trial["config"],
            "group_num": trial["group_num"],
            "group_size": trial["group_size"],
            "n_sheep": c["n_sheep"],
            "n_raw": len(c["nnd"]),
        }

        # (a) mean & SD at each timescale.
        for metric in ("nnd", "spread"):
            for ws in TIMESCALES_S:
                agg = block_aggregate(c[metric], c["dt_min"], ws)
                rec[f"{metric}_mean_{ws}s"] = float(np.mean(agg))
                rec[f"{metric}_sd_{ws}s"] = float(np.std(agg)) if len(agg) > 1 else np.nan

        # (b) autocorrelation persistence timescale (raw series).
        rec["nnd_tau_e_s"] = acf_crossing_timescale(c["nnd"], c["dt_min"])
        rec["spread_tau_e_s"] = acf_crossing_timescale(c["spread"], c["dt_min"])

        records.append(rec)

    if not records:
        print("FINDINGS: no usable trials after filtering — cannot compute cohesion "
              "timescales. (All candidate trials had <2 sheep or no overlapping data.)")
        return

    df = pd.DataFrame(records)
    print(f"Computed cohesion timescales for {len(df)} trials "
          f"({skipped} skipped for <2 sheep / no overlap).")

    assays = sorted(df["assay"].unique())
    configs = sorted(df["config"].unique())

    # -------------------------------------------------------------------
    # Timescale variance-reduction summary (cohort-wide)
    # -------------------------------------------------------------------
    print("\nVariance / mean vs timescale (cohort medians):")
    ts_summary = []
    for metric, label in (("nnd", "NND"), ("spread", "spread")):
        for ws in TIMESCALES_S:
            ts_summary.append({
                "metric": label,
                "window_s": ws,
                "median_mean_m": float(df[f"{metric}_mean_{ws}s"].median()),
                "median_sd_m": float(df[f"{metric}_sd_{ws}s"].median()),
            })
    ts_df = pd.DataFrame(ts_summary)
    print(ts_df.to_string(index=False))

    # SD retained at 60 s relative to 1 s — how much variance survives coarsening.
    sd_ratio = {}
    for metric, label in (("nnd", "NND"), ("spread", "spread")):
        r1 = df[f"{metric}_sd_1s"]
        r60 = df[f"{metric}_sd_60s"]
        valid = r1 > 1e-9
        sd_ratio[label] = float(np.median((r60[valid] / r1[valid]))) if valid.any() else np.nan
    print(f"\nMedian SD(60 s)/SD(1 s):  NND {sd_ratio['NND']:.2f}, "
          f"spread {sd_ratio['spread']:.2f}  "
          f"(lower → more cohesion variance lives at fast sub-minute timescales)")

    # -------------------------------------------------------------------
    # Persistence timescales (cohort-wide)
    # -------------------------------------------------------------------
    nnd_tau = df["nnd_tau_e_s"].dropna()
    sp_tau = df["spread_tau_e_s"].dropna()
    print("\nAutocorrelation persistence (1/e crossing):")
    print(f"  NND    tau_e median {nnd_tau.median():.1f} s "
          f"(IQR {nnd_tau.quantile(.25):.1f}–{nnd_tau.quantile(.75):.1f}, "
          f"n={len(nnd_tau)})")
    print(f"  spread tau_e median {sp_tau.median():.1f} s "
          f"(IQR {sp_tau.quantile(.25):.1f}–{sp_tau.quantile(.75):.1f}, "
          f"n={len(sp_tau)})")

    # -------------------------------------------------------------------
    # Trend tests vs assay (within-group shuffle null)
    # -------------------------------------------------------------------
    print(f"\nAssay-trend tests (within-group shuffle, {N_PERM} perms, seed {SEED}):")
    trend_cols = {
        "nnd_mean_1s": "mean NND (1 s)",
        "spread_mean_1s": "mean spread (1 s)",
        "nnd_sd_1s": "NND SD (1 s)",
        "spread_sd_1s": "spread SD (1 s)",
        "nnd_tau_e_s": "NND persistence tau_e",
        "spread_tau_e_s": "spread persistence tau_e",
    }
    trend_results = {}
    for col, label in trend_cols.items():
        rho, p, n = assay_shuffle_p(df, col)
        trend_results[col] = (rho, p, n)
        rho_s = f"{rho:+.3f}" if np.isfinite(rho) else "  nan"
        p_s = f"{p:.3f}" if np.isfinite(p) else "  nan"
        print(f"  {label:24s} rho={rho_s}  p={p_s}  n={n}")

    # -------------------------------------------------------------------
    # Aggregate-by-assay and by-config tables
    # -------------------------------------------------------------------
    def _agg(group_col, keys):
        rows = []
        for k in keys:
            sub = df[df[group_col] == k]
            rows.append({
                group_col: k, "n": len(sub),
                "NND_mean_1s": sub["nnd_mean_1s"].median(),
                "NND_sd_1s": sub["nnd_sd_1s"].median(),
                "NND_tau_e_s": sub["nnd_tau_e_s"].median(),
                "spread_mean_1s": sub["spread_mean_1s"].median(),
                "spread_sd_1s": sub["spread_sd_1s"].median(),
                "spread_tau_e_s": sub["spread_tau_e_s"].median(),
            })
        return pd.DataFrame(rows)

    by_assay = _agg("assay", assays)
    by_config = _agg("config", configs)
    print("\nMedian cohesion metrics by assay:")
    print(by_assay.round(2).to_string(index=False))
    print("\nMedian cohesion metrics by config:")
    print(by_config.round(2).to_string(index=False))

    # ===================================================================
    # FIGURES
    # ===================================================================
    # Fig 1: SD vs timescale (per metric), one line per trial + cohort median.
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (metric, label, colour) in zip(
        axes1, (("nnd", "NND", PALETTE["blue"]), ("spread", "Spread", PALETTE["orange"]))
    ):
        xs = np.array(TIMESCALES_S, dtype=float)
        sd_mat = np.column_stack([df[f"{metric}_sd_{ws}s"].to_numpy() for ws in TIMESCALES_S])
        for row in sd_mat:
            ax.plot(xs, row, color=colour, alpha=0.12, lw=0.7)
        ax.plot(xs, np.nanmedian(sd_mat, axis=0), color="k", lw=2, marker="o",
                label="cohort median")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(w)) for w in xs])
        ax.set_xlabel("Aggregation window (s)")
        ax.set_ylabel(f"Within-trial SD of {label} (m)")
        ax.set_title(f"{label}: fluctuation amplitude vs timescale")
        ax.legend()
    fig1.suptitle("Cohesion variance shrinks as timescale coarsens", fontsize=11)
    fig1.tight_layout()
    fig1.savefig(FIGDIR / "cohesion_timescales_variance.png")
    plt.close(fig1)
    print("\n  -> cohesion_timescales_variance.png")

    # Fig 2: persistence timescale (tau_e) by assay, box + points.
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, label, colour) in zip(
        axes2,
        (("nnd_tau_e_s", "NND", PALETTE["blue"]),
         ("spread_tau_e_s", "Spread", PALETTE["orange"])),
    ):
        data = [df[df["assay"] == a][col].dropna().values for a in assays]
        ax.boxplot(data, tick_labels=[str(a) for a in assays], showfliers=False,
                   medianprops=dict(color="k", lw=1.2))
        for i, d in enumerate(data):
            if len(d):
                jit = np.random.default_rng(SEED + i).uniform(-0.15, 0.15, len(d))
                ax.scatter(np.full(len(d), i + 1) + jit, d, s=12, alpha=0.5,
                           color=colour, edgecolors="none")
        rho, p, _ = trend_results[col]
        ax.set_xlabel("Assay")
        ax.set_ylabel(f"{label} persistence tau_e (s)")
        ax.set_title(f"{label} cohesion persistence by assay")
        if np.isfinite(rho):
            ax.text(0.97, 0.95, f"rho={rho:+.2f}, p={p:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, style="italic", color="#333")
    fig2.suptitle("How long do cohesion states persist? (1/e ACF crossing)", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(FIGDIR / "cohesion_persistence_by_assay.png")
    plt.close(fig2)
    print("  -> cohesion_persistence_by_assay.png")

    # Fig 3: magnitude (mean NND / spread, 1 s) by assay.
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, label, colour) in zip(
        axes3,
        (("nnd_mean_1s", "Mean NND", PALETTE["blue"]),
         ("spread_mean_1s", "Mean spread", PALETTE["orange"])),
    ):
        data = [df[df["assay"] == a][col].dropna().values for a in assays]
        ax.boxplot(data, tick_labels=[str(a) for a in assays], showfliers=False,
                   medianprops=dict(color="k", lw=1.2))
        for i, d in enumerate(data):
            if len(d):
                jit = np.random.default_rng(SEED + 100 + i).uniform(-0.15, 0.15, len(d))
                ax.scatter(np.full(len(d), i + 1) + jit, d, s=12, alpha=0.5,
                           color=colour, edgecolors="none")
        rho, p, _ = trend_results[col]
        ax.set_xlabel("Assay")
        ax.set_ylabel(f"{label} (m)")
        ax.set_title(f"{label} by assay")
        if np.isfinite(rho):
            ax.text(0.97, 0.95, f"rho={rho:+.2f}, p={p:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, style="italic", color="#333")
    fig3.suptitle("Cohesion magnitude by assay", fontsize=11)
    fig3.tight_layout()
    fig3.savefig(FIGDIR / "cohesion_magnitude_by_assay.png")
    plt.close(fig3)
    print("  -> cohesion_magnitude_by_assay.png")

    # ===================================================================
    # FINDINGS
    # ===================================================================
    # Characteristic timescale = cohort-median tau_e.
    tau_nnd = nnd_tau.median()
    tau_sp = sp_tau.median()

    def _verdict(col):
        rho, p, _ = trend_results[col]
        if not np.isfinite(p):
            return "untestable"
        return f"rho={rho:+.2f}, p={p:.3f} ({'trend' if p < 0.05 else 'no trend'})"

    print("\n" + "=" * 70)
    print("FINDINGS:")
    print(f"- Across {len(df)} Phase-2 test trials, cohesion has a characteristic "
          f"persistence timescale of ~{tau_nnd:.0f} s for NND and ~{tau_sp:.0f} s "
          f"for spread (median lag at which the ACF first crosses 1/e). Cohesion "
          f"states are thus sustained over tens of seconds, not instantaneous.")
    print(f"- Coarsening the series from ~1 s to 60 s windows leaves the mean "
          f"essentially unchanged but shrinks within-trial SD to "
          f"{sd_ratio['NND']*100:.0f}% (NND) and {sd_ratio['spread']*100:.0f}% "
          f"(spread) of its raw value, i.e. much of the cohesion fluctuation "
          f"lives at sub-minute timescales.")
    print(f"- Cohesion MAGNITUDE vs assay: mean NND {_verdict('nnd_mean_1s')}; "
          f"mean spread {_verdict('spread_mean_1s')}.")
    print(f"- Cohesion PERSISTENCE vs assay: NND tau_e {_verdict('nnd_tau_e_s')}; "
          f"spread tau_e {_verdict('spread_tau_e_s')}.")
    sig = [c for c in trend_cols
           if np.isfinite(trend_results[c][1]) and trend_results[c][1] < 0.05]
    if sig:
        print(f"- INTERPRETATION: {len(sig)} cohesion metric(s) trend with assay "
              f"({', '.join(trend_cols[c] for c in sig)}), so learning appears to "
              f"reshape group structure, not only navigation.")
    else:
        print("- INTERPRETATION: no cohesion metric (magnitude or persistence) "
              "trends significantly with assay, consistent with learning changing "
              "navigation while leaving group structure stable across assays.")
    print("=" * 70)


if __name__ == "__main__":
    main()
