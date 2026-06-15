"""Worked example: load trials, plot one trial in arena coordinates.

Run from the package root:
    python example_usage.py
"""
import matplotlib.pyplot as plt
import pandas as pd

from sheep_gnss_loader import (
    SITE_GRID,
    load_trial_tracks,
    load_trials,
)


def main():
    trials = load_trials("trial_metadata.csv")
    print(f"Loaded {len(trials)} trials")

    # Pick the first trial that actually has GNSS data on disk.
    trial = next(t for t in trials if t["devices"])
    print(f"First trial: {trial['name']}")
    print(f"  field={trial['field']}, config={trial['config']}, "
          f"assay={trial['assay']}, n_devices={len(trial['devices'])}")

    tracks = load_trial_tracks(
        trial,
        gnss_root="gnss/",
        reward_sites_csv="fitted_reward_sites.csv",
        apply_orient=False,
    )
    if not tracks:
        print("  No tracks loaded (GNSS data missing for this trial).")
        return

    n_samples = {sid: len(t["t"]) for sid, t in tracks.items()}
    print(f"  Tracks loaded: {len(tracks)} sheep, samples: {n_samples}")

    fig, ax = plt.subplots(figsize=(8, 8))
    for sid, t in tracks.items():
        ax.plot(t["gx"], t["gy"], lw=0.5, alpha=0.7, label=f"sheep {sid}")

    # Overlay the 12 reward sites.
    for label, (sx, sy) in SITE_GRID.items():
        ax.scatter([sx], [sy], s=120, marker="o", facecolors="none",
                   edgecolors="black", lw=1.2, zorder=5)
        ax.annotate(label, (sx, sy), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Arena x (grid units; 1 unit = 10 m)")
    ax.set_ylabel("Arena y (grid units; 1 unit = 10 m)")
    ax.set_title(trial["name"], fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    out = "trial_overview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"  Saved plot: {out}")


if __name__ == "__main__":
    main()
