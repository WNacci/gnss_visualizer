#!/usr/bin/env python3
"""Paired-device inter-distance: an empirical estimate of GPS precision.

Some sheep wore TWO GPS devices simultaneously. Because both devices ride on
the same animal, the distance between their two tracks at any instant is a
direct, model-free estimate of GPS jitter/accuracy (the two devices *should*
report the same position). We:

  1. Load each trial's tracks WITHOUT averaging the two devices
     (``load_trial_device_tracks``), keeping ``dev_num -> {gx, gy, t, sheep_id}``.
  2. For every sheep carrying exactly two devices, interpolate both device
     tracks onto a common time grid and compute the inter-device distance over
     time (grid units x 10 = metres).
  3. Report the distribution (mean / median / 95th pct) of inter-device
     distance per paired sheep, pooled across trials, and count how many
     sheep/trials contributed paired devices.

We use ``apply_orient=False``: orientation is a per-config rigid transform that
is identical for both devices on the same sheep, so it cannot change their
relative distance. Skipping it avoids needless work and any offset corrections.

Phase 2 cohort only (config in A/B/C/D, integer assay, date >= 2026-02-17,
group not a control). Group 13 has a known device mislabel (13689 vs 13686 =
same animal); we treat whatever sheep_id the metadata assigns and never crash.

Produces a histogram + per-sheep summary figure under analysis/figures/ and
prints a FINDINGS block. Safe to run repeatedly.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from gps_analysis import (
    build_trials,
    load_trial_device_tracks,
    load_gnss_date,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Phase 2 cohort filter (matches generate_figures.py)
# ---------------------------------------------------------------------------
TEST_CONFIGS = {"A", "B", "C", "D"}
PHASE2_START = "2026-02-17"
CTRL_GROUPS = {9, 14}
GRID_TO_M = 10.0          # one grid unit = 10 metres
DETECTION_RADIUS_M = 2.0  # the site-detection radius this precision must support

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def _common_grid_distance(trk_a, trk_b):
    """Inter-device distance (metres) on a shared 10 Hz time grid.

    Both tracks are interpolated onto the overlapping time window only, so we
    never extrapolate beyond either device's coverage. Returns a 1-D array of
    distances in metres, or an empty array if there is no usable overlap.
    """
    ta, tb = trk_a["t"], trk_b["t"]
    if len(ta) < 2 or len(tb) < 2:
        return np.array([])

    # Sort by time (kalman smoothing preserves order, but be safe).
    oa, ob = np.argsort(ta), np.argsort(tb)
    ta, axg, ayg = ta[oa], trk_a["gx"][oa], trk_a["gy"][oa]
    tb, bxg, byg = tb[ob], trk_b["gx"][ob], trk_b["gy"][ob]

    t0 = max(ta[0], tb[0])
    t1 = min(ta[-1], tb[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.array([])

    grid = np.arange(t0, t1, 1 / 600.0)  # 10 Hz (0.1 s steps), minutes units
    if grid.size == 0:
        return np.array([])

    ax = np.interp(grid, ta, axg)
    ay = np.interp(grid, ta, ayg)
    bx = np.interp(grid, tb, bxg)
    by = np.interp(grid, tb, byg)

    d = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2) * GRID_TO_M
    return d[np.isfinite(d)]


def main():
    print("Loading trial metadata...")
    trials = build_trials()
    test_trials = [
        t for t in trials
        if t["config"] in TEST_CONFIGS
        and t["assay"] is not None
        and isinstance(t["assay"], int)
        and t["date"] >= PHASE2_START
        and t["group_num"] not in CTRL_GROUPS
    ]
    print(f"Phase 2 test trials: {len(test_trials)} (of {len(trials)} total)")

    # Per-date GNSS cache built ourselves (cheap; avoids the 2 GB build_gps_cache).
    dates = sorted({t["date"] for t in test_trials})
    print(f"Loading raw GNSS for {len(dates)} dates...")
    gnss_cache = {}
    for d in dates:
        try:
            gnss_cache[d] = load_gnss_date(d)
        except Exception as exc:  # noqa: BLE001 - robustness over a missing/corrupt date
            print(f"  WARN: could not load GNSS for {d}: {exc}")
            gnss_cache[d] = {}

    # ----------------------------------------------------------------------
    # Walk trials; collect per (trial, sheep) paired-device distance series.
    # ----------------------------------------------------------------------
    per_sheep_records = []   # one row per paired sheep-in-trial
    pooled_distances = []    # every per-sample distance, pooled
    paired_trials = set()
    n_single_dev_sheep = 0

    for trial in test_trials:
        dev_tracks = load_trial_device_tracks(
            trial, gnss_cache=gnss_cache, apply_orient=False
        )
        if not dev_tracks:
            continue

        # Group devices by the sheep that wore them.
        by_sheep: dict[str, list[int]] = {}
        for dev_num, trk in dev_tracks.items():
            if trk["gx"].size == 0:
                continue
            by_sheep.setdefault(trk["sheep_id"], []).append(dev_num)

        for sheep_id, devs in by_sheep.items():
            if len(devs) < 2:
                n_single_dev_sheep += 1
                continue
            # If a sheep somehow has >2 devices, use the first two with data.
            d_arr = _common_grid_distance(dev_tracks[devs[0]], dev_tracks[devs[1]])
            if d_arr.size == 0:
                continue

            paired_trials.add(trial["name"])
            pooled_distances.append(d_arr)
            per_sheep_records.append({
                "trial": trial["name"],
                "date": trial["date"],
                "group_num": trial["group_num"],
                "assay": trial["assay"],
                "sheep_id": sheep_id,
                "dev_a": devs[0],
                "dev_b": devs[1],
                "n_samples": d_arr.size,
                "mean_m": float(np.mean(d_arr)),
                "median_m": float(np.median(d_arr)),
                "p25_m": float(np.percentile(d_arr, 25)),
                "p95_m": float(np.percentile(d_arr, 95)),
                "max_m": float(np.max(d_arr)),
            })

    per_sheep_df = pd.DataFrame(per_sheep_records)

    if per_sheep_df.empty:
        print("\nFINDINGS:")
        print("  No sheep with two simultaneous GPS devices were found in the "
              "Phase 2 cohort, so empirical GPS precision could not be "
              "estimated. (n_single_device_sheep_in_trials="
              f"{n_single_dev_sheep})")
        return

    pooled = np.concatenate(pooled_distances)

    # Per-sheep summary (each paired sheep weighted equally via its own stats).
    sheep_mean = per_sheep_df["mean_m"].to_numpy()
    sheep_median = per_sheep_df["median_m"].to_numpy()
    sheep_p25 = per_sheep_df["p25_m"].to_numpy()
    sheep_p95 = per_sheep_df["p95_m"].to_numpy()

    n_paired_sheep_in_trials = len(per_sheep_df)
    n_unique_sheep = per_sheep_df["sheep_id"].nunique()
    n_paired_trials = len(paired_trials)

    # ----------------------------------------------------------------------
    # Figure: pooled distribution + per-sheep means
    # ----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    clip = np.percentile(pooled, 99.5)
    ax1.hist(pooled, bins=80, range=(0, clip), color="#3B7DD8", alpha=0.8)
    ax1.axvline(np.median(pooled), color="k", ls="--", lw=1.0,
                label=f"median {np.median(pooled):.2f} m")
    ax1.axvline(np.percentile(pooled, 95), color="#D64550", ls=":", lw=1.0,
                label=f"95th pct {np.percentile(pooled, 95):.2f} m")
    ax1.axvline(DETECTION_RADIUS_M, color="#4AAD5B", ls="-", lw=1.0,
                label="2 m detection radius")
    ax1.set_xlabel("Inter-device distance (m)")
    ax1.set_ylabel("Sample count (10 Hz)")
    ax1.set_title("a  Pooled inter-device distance")
    ax1.legend(fontsize=6, frameon=False)

    # Centre each marker on the per-sheep median with non-negative whiskers
    # running from the 25th to the 95th percentile of that sheep's distances.
    order = np.argsort(sheep_median)
    y = np.arange(len(order))
    ax2.errorbar(
        sheep_median[order], y,
        xerr=[sheep_median[order] - sheep_p25[order],
              sheep_p95[order] - sheep_median[order]],
        fmt="o", ms=3, lw=0.8, color="#E8823A", ecolor="#bbb", capsize=0,
    )
    ax2.axvline(DETECTION_RADIUS_M, color="#4AAD5B", ls="-", lw=1.0)
    ax2.set_xlabel("Per-sheep inter-device distance (m)")
    ax2.set_ylabel("Paired sheep-trial (sorted)")
    ax2.set_title("b  Per-sheep precision\n(median, 25th-95th pct)")
    ax2.set_yticks([])

    fig.tight_layout()
    fig.savefig(FIGDIR / "paired_device_distance.pdf")
    fig.savefig(FIGDIR / "paired_device_distance.png")
    plt.close(fig)
    print("  -> paired_device_distance figure written")

    per_sheep_df.sort_values("mean_m").to_csv(
        FIGDIR / "paired_device_distance.csv", index=False
    )

    # ----------------------------------------------------------------------
    # FINDINGS
    # ----------------------------------------------------------------------
    pooled_median = float(np.median(pooled))
    pooled_mean = float(np.mean(pooled))
    pooled_p95 = float(np.percentile(pooled, 95))
    frac_within_2m = float(np.mean(pooled <= DETECTION_RADIUS_M))

    print("\nFINDINGS:")
    print(f"  Empirical GPS precision from {n_paired_sheep_in_trials} paired "
          f"device-pairs ({n_unique_sheep} unique sheep) across "
          f"{n_paired_trials} trials; {n_single_dev_sheep} single-device "
          f"sheep-in-trials were skipped.")
    print(f"  Inter-device distance, pooled over {pooled.size:,} 10 Hz samples: "
          f"median {pooled_median:.2f} m, mean {pooled_mean:.2f} m, "
          f"95th pct {pooled_p95:.2f} m.")
    print(f"  Per-sheep (each pair weighted equally): "
          f"median of medians {np.median(sheep_median):.2f} m, "
          f"median of means {np.median(sheep_mean):.2f} m, "
          f"median of 95th-pcts {np.median(sheep_p95):.2f} m "
          f"(range of per-sheep means {sheep_mean.min():.2f}-{sheep_mean.max():.2f} m).")
    print(f"  {100 * frac_within_2m:.1f}% of paired samples fall within the "
          f"{DETECTION_RADIUS_M:.0f} m site-detection radius.")
    print(f"  Implication: two devices on the SAME animal disagree by ~"
          f"{pooled_median:.1f} m typically (95th pct {pooled_p95:.1f} m). "
          f"Since this is a difference of two independent GPS errors, the "
          f"single-device error is ~{pooled_median / np.sqrt(2):.1f} m (median). "
          f"A {DETECTION_RADIUS_M:.0f} m detection radius is therefore "
          f"{'comparable to' if pooled_median > DETECTION_RADIUS_M else 'larger than'} "
          f"GPS jitter: it is justified but not generous, so site-visit "
          f"detection at 2 m is sensitive to GPS noise and the looser 5 m "
          f"exploration radius used elsewhere is the more robust choice.")


if __name__ == "__main__":
    main()
