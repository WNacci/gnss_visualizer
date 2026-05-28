#!/usr/bin/env python3
"""Generate MP4 videos combining success tracking and leadership analysis
side-by-side. Output is scrubable video at 30fps.

Left panel: trajectory buildup + site discovery
Right panel: leadership metric (velocity arrow, projections, leader highlight)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter

from gps_analysis import (
    build_trials, build_tracks_cache, load_trial_tracks,
    detect_site_visits, SITE_GRID,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
BAITED = {"A1", "A2", "A3"}
site_positions = {lbl: (x, y) for lbl, (x, y) in SITE_GRID.items()}

PALETTE = {
    "blue": "#3B7DD8", "orange": "#E8823A", "green": "#4AAD5B",
    "red": "#D64550", "gold": "#E8B83D", "grey": "#888888",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FPS = 30
DPI = 120

print("Loading data...")
trials = build_trials()
tc = build_tracks_cache(trials)
test = [t for t in trials
        if t["config"] in {"A", "B", "C", "D"}
        and isinstance(t.get("assay"), int)
        and t["date"] >= "2026-02-17"
        and t["group_num"] not in {9, 14}]

tab10 = plt.colormaps["tab10"]


def generate_combined_video(trial, outpath, duration_min=None):
    """Generate a combined success + leadership video for one trial."""
    tracks = load_trial_tracks(trial, tracks_cache=tc, apply_orient=True)
    if not tracks or len(tracks) < 2:
        print(f"  Skipping {trial['name']}: insufficient tracks")
        return

    visits = detect_site_visits(tracks, trial["field"], 0.2, 5.0)
    bd = {s: min(v[1] for v in vl) for s, vl in visits.items()
          if vl and s in BAITED}
    ct = max(bd.values()) if len(bd) == 3 else None

    sids = sorted(tracks.keys())
    n = len(sids)

    # Interpolate to 1s grid
    t_grid = np.arange(0, 25, 1 / 60)
    interp = {}
    for sid in sids:
        trk = tracks[sid]
        o = np.argsort(trk["t"])
        mask = t_grid <= trk["t"][o].max()
        tg = t_grid[mask]
        interp[sid] = {
            "gx": np.interp(tg, trk["t"][o], trk["gx"][o]),
            "gy": np.interp(tg, trk["t"][o], trk["gy"][o]),
            "t": tg,
        }

    T = min(len(interp[s]["gx"]) for s in sids)
    GX = np.column_stack([interp[s]["gx"][:T] for s in sids])
    GY = np.column_stack([interp[s]["gy"][:T] for s in sids])
    cx_arr, cy_arr = GX.mean(axis=1), GY.mean(axis=1)

    kernel = np.ones(15) / 15
    vx_arr = np.convolve(np.gradient(cx_arr), kernel, mode="same")
    vy_arr = np.convolve(np.gradient(cy_arr), kernel, mode="same")
    speed_arr = np.sqrt(vx_arr ** 2 + vy_arr ** 2)

    # Duration: to completion + 30s, or duration_min, or 10 min
    if duration_min is not None:
        t_end_min = duration_min
    elif ct is not None:
        t_end_min = min(ct + 0.5, 25.0)
    else:
        t_end_min = 10.0

    n_frames = int(t_end_min * 60)  # 1 frame per second of real time
    n_frames = min(n_frames, T)

    # Set up figure with two panels
    fig, (ax_traj, ax_lead) = plt.subplots(1, 2, figsize=(12, 5.5))

    def _draw_arena(ax, show_detection=False):
        ax.set_facecolor("#f5f5f5")
        for lbl, (sx, sy) in site_positions.items():
            c = PALETTE["gold"] if lbl.startswith("A") else "#ddd"
            ax.add_patch(Circle((sx, sy), 0.18, fc=c, ec="k", lw=0.6,
                                alpha=0.7, zorder=5))
            ax.text(sx, sy, lbl, ha="center", va="center", fontsize=6,
                    fontweight="bold", zorder=6)
        if show_detection:
            for s in BAITED:
                sx, sy = site_positions[s]
                ax.add_patch(Circle((sx, sy), 0.2, fill=False, ec="#bbb",
                                    ls="--", lw=0.8, zorder=4))
        ax.set_xlim(-0.1, 5.1)
        ax.set_ylim(-0.1, 5.1)
        ax.set_aspect("equal")
        ax.set_xlabel("Grid X (10 m/unit)")
        ax.set_ylabel("Grid Y")

    # We'll clear and redraw each frame for simplicity with patches
    def update(frame_idx):
        ti = frame_idx
        if ti >= T:
            ti = T - 1
        t_now = t_grid[ti] if ti < len(t_grid) else 0

        # --- LEFT: Trajectory + success ---
        ax_traj.clear()
        _draw_arena(ax_traj, show_detection=True)

        # Update detection circles for discovered sites
        for s in BAITED:
            sx, sy = site_positions[s]
            discovered = s in bd and bd[s] <= t_now
            if discovered:
                ax_traj.add_patch(Circle((sx, sy), 0.2, fill=False,
                                         ec=PALETTE["green"], ls="--", lw=2, zorder=5))
                ax_traj.plot(sx, sy, "*", ms=16, color=PALETTE["green"], zorder=7,
                             markeredgecolor="k", markeredgewidth=0.4)

        for i, sid in enumerate(sids):
            trk = tracks[sid]
            mask = trk["t"] <= t_now
            if mask.sum() == 0:
                continue
            gx, gy = trk["gx"][mask], trk["gy"][mask]
            ax_traj.plot(gx, gy, lw=1.0, alpha=0.5, color=tab10(i % 10), zorder=3)
            ax_traj.plot(gx[-1], gy[-1], "o", ms=7, color=tab10(i % 10),
                         markeredgecolor="k", markeredgewidth=0.4, zorder=8,
                         label=sid if frame_idx == 0 else None)

        n_found = sum(1 for s in BAITED if s in bd and bd[s] <= t_now)
        completed = n_found == 3
        status = "COMPLETE!" if completed else f"{n_found}/3 baited"
        color = PALETTE["green"] if completed else "k"
        ax_traj.set_title(f"Trajectory  |  t = {t_now:.1f} min  |  {status}",
                          color=color)

        # --- RIGHT: Leadership ---
        ax_lead.clear()
        ax_lead.set_facecolor("#fafafa")
        for lbl, (sx, sy) in site_positions.items():
            c = PALETTE["gold"] if lbl.startswith("A") else "#eee"
            ax_lead.add_patch(Circle((sx, sy), 0.15, fc=c, ec="#bbb", lw=0.3,
                                     alpha=0.4, zorder=2))

        # Trail
        trail_start = max(0, ti - 60)
        for i in range(n):
            ax_lead.plot(GX[trail_start:ti + 1, i], GY[trail_start:ti + 1, i],
                         lw=0.8, alpha=0.2, color=tab10(i % 10))

        cx, cy = cx_arr[ti], cy_arr[ti]
        vx, vy = vx_arr[ti], vy_arr[ti]
        spd = speed_arr[ti]
        moving = spd > 0.05

        ax_lead.plot(cx, cy, "+", ms=14, mew=2.5, color="k", zorder=10)

        vnx = vny = 0
        if moving:
            vnx, vny = vx / spd, vy / spd
            arrow_scale = 5.0
            ax_lead.annotate("", xy=(cx + arrow_scale * vx, cy + arrow_scale * vy),
                             xytext=(cx, cy),
                             arrowprops=dict(arrowstyle="-|>", lw=2.5, color="k",
                                             mutation_scale=18), zorder=9)
            ax_lead.plot([cx - 2.0 * vnx, cx + 2.5 * vnx],
                         [cy - 2.0 * vny, cy + 2.5 * vny],
                         "-", color="#ddd", lw=1, zorder=1)

        projections = {}
        for i, sid in enumerate(sids):
            sx, sy = GX[ti, i], GY[ti, i]
            if moving:
                proj = (sx - cx) * vnx + (sy - cy) * vny
                projections[sid] = proj
                px, py = cx + proj * vnx, cy + proj * vny
                ax_lead.plot([sx, px], [sy, py], "--", color=tab10(i % 10),
                             lw=1, alpha=0.4)
            else:
                projections[sid] = 0

            ax_lead.plot(sx, sy, "o", ms=10, color=tab10(i % 10),
                         markeredgecolor="k", markeredgewidth=0.6, zorder=8)
            ax_lead.text(sx + 0.1, sy + 0.12, sid[:5], fontsize=6,
                         color=tab10(i % 10), fontweight="bold")

        if moving and projections:
            leader = max(projections, key=projections.get)
            li = sids.index(leader)
            ax_lead.plot(GX[ti, li], GY[ti, li], "o", ms=15, color="none",
                         markeredgecolor=PALETTE["gold"], markeredgewidth=3, zorder=9)
            lead_status = f"Leader: {leader[:5]}"
        else:
            lead_status = "Stationary"

        ax_lead.set_xlim(-0.1, 5.1)
        ax_lead.set_ylim(-0.1, 5.1)
        ax_lead.set_aspect("equal")
        ax_lead.set_xlabel("Grid X")
        ax_lead.set_ylabel("Grid Y")
        ax_lead.set_title(f"Leadership  |  {lead_status}")

        fig.suptitle(f"Assay {trial['assay']}  |  Group {trial['group_num']}  |  "
                     f"{n} sheep  |  Config {trial['config']}",
                     fontsize=11, fontweight="bold", y=0.98)

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    writer = FFMpegWriter(fps=FPS, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "23"])
    anim.save(str(outpath), writer=writer, dpi=DPI)
    plt.close(fig)

    import os
    size_mb = os.path.getsize(outpath) / 1024 / 1024
    print(f"  -> {outpath.name} ({n_frames} frames, {n_frames/FPS:.1f}s, {size_mb:.1f} MB)")


# ===========================================================================
# SELECT TRIALS AND GENERATE
# ===========================================================================

# Find 3 diverse trials
candidates = []
for trial in test:
    if trial["group_size"] < 3:
        continue
    tracks = load_trial_tracks(trial, tracks_cache=tc, apply_orient=True)
    if not tracks or len(tracks) < 3:
        continue
    visits = detect_site_visits(tracks, trial["field"], 0.2, 5.0)
    bd = {s: min(v[1] for v in vl) for s, vl in visits.items() if vl and s in BAITED}
    ct = max(bd.values()) if len(bd) == 3 else None
    candidates.append((trial, ct, len(bd)))

# Pick: one success (early assay), one success (late assay), one failure
selected = {}
for trial, ct, n_baited in candidates:
    if ct is not None and trial["assay"] in [2, 3] and "early_success" not in selected:
        selected["early_success"] = (trial, min(ct + 1, 25))
    elif ct is not None and trial["assay"] in [5, 6, 7] and "late_success" not in selected:
        selected["late_success"] = (trial, min(ct + 1, 25))
    elif ct is None and trial["assay"] in [0, 1] and "failure" not in selected:
        selected["failure"] = (trial, 8.0)  # show first 8 min of failed trial
    if len(selected) == 3:
        break

print(f"\nGenerating {len(selected)} combined videos...")
for i, (key, (trial, dur)) in enumerate(selected.items(), 1):
    print(f"\n  [{i}] {key}: assay {trial['assay']}, grp {trial['group_num']}, "
          f"duration {dur:.1f} min")
    generate_combined_video(trial, FIGDIR / f"combined_{key}.mp4", duration_min=dur)

print("\nAll videos done!")
