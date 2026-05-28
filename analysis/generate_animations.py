#!/usr/bin/env python3
"""Generate animated GIFs:
  - 3 success animations (early/mid/late assay)
  - 3 leadership animations (distributed/dominant/mid-range)
  All at ~60fps, full arena view.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import io

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

GIF_DPI = 80
GIF_DURATION = 17  # ms per frame ≈ 59 fps
HOLD_FRAMES = 90   # hold last frame ~1.5s at 60fps

print("Loading data...")
trials = build_trials()
tc = build_tracks_cache(trials)
test = [t for t in trials
        if t["config"] in {"A", "B", "C", "D"}
        and isinstance(t.get("assay"), int)
        and t["date"] >= "2026-02-17"
        and t["group_num"] not in {9, 14}]

tab10 = plt.colormaps["tab10"]


# ===========================================================================
# Reusable rendering functions
# ===========================================================================

def _render_success_gif(trial, tracks, visits, bd, outpath):
    """Render a success animation GIF for one trial."""
    ct = max(bd.values()) if len(bd) == 3 else 25.0
    sids = sorted(tracks.keys())
    t_end = min(ct + 0.5, 25.0)

    # 1 frame per real second
    frame_times = np.arange(0, t_end, 1 / 60)
    frames = []

    for t_now in frame_times:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=GIF_DPI)
        ax.set_facecolor("#f5f5f5")

        # Reward sites
        for lbl, (sx, sy) in site_positions.items():
            c = PALETTE["gold"] if lbl.startswith("A") else "#ddd"
            ax.add_patch(Circle((sx, sy), 0.18, fc=c, ec="k", lw=0.6, alpha=0.7, zorder=5))
            ax.text(sx, sy, lbl, ha="center", va="center", fontsize=6, fontweight="bold", zorder=6)

        # 2m detection circles for baited sites
        for s in BAITED:
            sx, sy = site_positions[s]
            discovered = s in bd and bd[s] <= t_now
            ec = PALETTE["green"] if discovered else "#bbb"
            lw = 2.0 if discovered else 0.8
            ax.add_patch(Circle((sx, sy), 0.2, fill=False, ec=ec, ls="--", lw=lw, zorder=5))
            if discovered:
                ax.plot(sx, sy, "*", ms=16, color=PALETTE["green"], zorder=7,
                        markeredgecolor="k", markeredgewidth=0.4)

        # Trajectories up to current time
        for i, sid in enumerate(sids):
            trk = tracks[sid]
            mask = trk["t"] <= t_now
            if mask.sum() == 0:
                continue
            gx, gy = trk["gx"][mask], trk["gy"][mask]
            ax.plot(gx, gy, lw=1.2, alpha=0.5, color=tab10(i % 10), zorder=3)
            ax.plot(gx[-1], gy[-1], "o", ms=7, color=tab10(i % 10),
                    markeredgecolor="k", markeredgewidth=0.4, zorder=8)

        ax.set_xlim(-0.1, 5.1)
        ax.set_ylim(-0.1, 5.1)
        ax.set_aspect("equal")

        n_found = sum(1 for s in BAITED if s in bd and bd[s] <= t_now)
        status = "COMPLETE!" if n_found == 3 else f"{n_found}/3 baited sites found"
        color = PALETTE["green"] if n_found == 3 else "k"
        ax.set_title(f"Assay {trial['assay']}  |  t = {t_now:.1f} min  |  {status}",
                     fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("Grid X (10 m/unit)")
        ax.set_ylabel("Grid Y")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    # Hold last frame
    for _ in range(HOLD_FRAMES):
        frames.append(frames[-1])

    frames[0].save(outpath, save_all=True, append_images=frames[1:],
                   duration=GIF_DURATION, loop=0)
    print(f"  -> {outpath.name} ({len(frames)} frames, {len(frames)*GIF_DURATION/1000:.1f}s)")


def _render_leadership_gif(trial, tracks, outpath, window_min=5.0):
    """Render a leadership animation GIF for one trial. Full arena view."""
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
        }

    T = min(len(interp[s]["gx"]) for s in sids)
    GX = np.column_stack([interp[s]["gx"][:T] for s in sids])
    GY = np.column_stack([interp[s]["gy"][:T] for s in sids])
    cx_arr, cy_arr = GX.mean(axis=1), GY.mean(axis=1)

    kernel = np.ones(15) / 15
    vx_arr = np.convolve(np.gradient(cx_arr), kernel, mode="same")
    vy_arr = np.convolve(np.gradient(cy_arr), kernel, mode="same")
    speed_arr = np.sqrt(vx_arr ** 2 + vy_arr ** 2)

    # Find best window (highest mean speed)
    win = int(window_min * 60)
    best_start, best_speed = 0, 0
    for start in range(0, T - win, 10):
        avg = speed_arr[start:start + win].mean()
        if avg > best_speed:
            best_speed = avg
            best_start = start

    # 1 frame per real second
    frame_indices = range(best_start, min(best_start + win, T))
    frames = []

    for ti in frame_indices:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=GIF_DPI)
        ax.set_facecolor("#fafafa")

        t_now = t_grid[ti] if ti < len(t_grid) else 0

        # Reward sites (background)
        for lbl, (sx, sy) in site_positions.items():
            c = PALETTE["gold"] if lbl.startswith("A") else "#eee"
            ax.add_patch(Circle((sx, sy), 0.15, fc=c, ec="#bbb", lw=0.4, alpha=0.5, zorder=2))
            ax.text(sx, sy, lbl, ha="center", va="center", fontsize=5, color="#999", zorder=3)

        # Trail (last 60 seconds)
        trail_start = max(0, ti - 60)
        for i, sid in enumerate(sids):
            ax.plot(GX[trail_start:ti + 1, i], GY[trail_start:ti + 1, i],
                    lw=0.8, alpha=0.25, color=tab10(i % 10))

        # Centroid and velocity
        cx, cy = cx_arr[ti], cy_arr[ti]
        vx, vy = vx_arr[ti], vy_arr[ti]
        spd = speed_arr[ti]
        moving = spd > 0.05

        ax.plot(cx, cy, "+", ms=14, mew=2.5, color="k", zorder=10)

        if moving:
            vnx, vny = vx / spd, vy / spd
            arrow_scale = 5.0
            ax.annotate("", xy=(cx + arrow_scale * vx, cy + arrow_scale * vy),
                        xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="-|>", lw=2.5, color="k",
                                        mutation_scale=18), zorder=9)

            # Projection axis
            ax.plot([cx - 2.0 * vnx, cx + 2.5 * vnx],
                    [cy - 2.0 * vny, cy + 2.5 * vny],
                    "-", color="#ddd", lw=1, zorder=1)

        # Sheep and projections
        projections = {}
        for i, sid in enumerate(sids):
            sx, sy = GX[ti, i], GY[ti, i]
            if moving:
                proj = (sx - cx) * vnx + (sy - cy) * vny
                projections[sid] = proj
                px, py = cx + proj * vnx, cy + proj * vny
                ax.plot([sx, px], [sy, py], "--", color=tab10(i % 10), lw=1, alpha=0.4)
            else:
                projections[sid] = 0

            ax.plot(sx, sy, "o", ms=10, color=tab10(i % 10),
                    markeredgecolor="k", markeredgewidth=0.6, zorder=8)
            ax.text(sx + 0.1, sy + 0.12, sid[:5], fontsize=6,
                    color=tab10(i % 10), fontweight="bold")

        # Highlight leader
        if moving and projections:
            leader = max(projections, key=projections.get)
            li = sids.index(leader)
            ax.plot(GX[ti, li], GY[ti, li], "o", ms=15, color="none",
                    markeredgecolor=PALETTE["gold"], markeredgewidth=3, zorder=9)
            status = f"Leader: {leader[:5]}"
        else:
            status = "Stationary"

        ax.set_xlim(-0.1, 5.1)
        ax.set_ylim(-0.1, 5.1)
        ax.set_aspect("equal")
        ax.set_title(f"Assay {trial['assay']} Grp{trial['group_num']}  |  "
                     f"t = {t_now:.1f} min  |  {status}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    for _ in range(HOLD_FRAMES):
        frames.append(frames[-1])

    frames[0].save(outpath, save_all=True, append_images=frames[1:],
                   duration=GIF_DURATION, loop=0)
    print(f"  -> {outpath.name} ({len(frames)} frames, {len(frames)*GIF_DURATION/1000:.1f}s)")


# ===========================================================================
# SELECT TRIALS AND GENERATE
# ===========================================================================

# --- Success trials ---
print("\n=== SUCCESS ANIMATIONS ===")
success_candidates = {"early": None, "mid": None, "late": None}

for trial in test:
    tracks = load_trial_tracks(trial, tracks_cache=tc, apply_orient=True)
    if not tracks or len(tracks) < 2:
        continue
    visits = detect_site_visits(tracks, trial["field"], 0.2, 5.0)
    bd = {s: min(v[1] for v in vl) for s, vl in visits.items() if vl and s in BAITED}
    if len(bd) != 3:
        continue
    ct = max(bd.values())

    if trial["assay"] in [2, 3] and ct > 4 and success_candidates["early"] is None:
        success_candidates["early"] = (trial, tracks, visits, bd)
    elif trial["assay"] in [4, 5] and 2.5 < ct < 6 and success_candidates["mid"] is None:
        success_candidates["mid"] = (trial, tracks, visits, bd)
    elif trial["assay"] in [6, 7] and ct < 4 and success_candidates["late"] is None:
        success_candidates["late"] = (trial, tracks, visits, bd)

    if all(v is not None for v in success_candidates.values()):
        break

for i, (key, data) in enumerate(success_candidates.items(), 1):
    if data is None:
        print(f"  Skipping success_{i} ({key}): no suitable trial found")
        continue
    trial, tracks, visits, bd = data
    ct = max(bd.values())
    print(f"  {key}: assay {trial['assay']}, grp {trial['group_num']}, "
          f"completion {ct:.1f} min")
    _render_success_gif(trial, tracks, visits, bd,
                        FIGDIR / f"success_animation_{i}.gif")


# --- Leadership trials ---
print("\n=== LEADERSHIP ANIMATIONS ===")

# Compute entropy for all multi-sheep trials to pick examples
SMOOTH_WIN = 15
SPEED_THRESH = 0.05
leader_candidates = []

for trial in test:
    if trial["group_size"] < 3:
        continue
    tracks = load_trial_tracks(trial, tracks_cache=tc, apply_orient=True)
    if not tracks or len(tracks) < 3:
        continue

    sids = sorted(tracks.keys())
    n_s = len(sids)
    t_grid = np.arange(0, 25, 1 / 60)
    arrs = {}
    for sid in sids:
        trk = tracks[sid]
        o = np.argsort(trk["t"])
        mask = t_grid <= trk["t"][o].max()
        tg = t_grid[mask]
        arrs[sid] = (np.interp(tg, trk["t"][o], trk["gx"][o]),
                     np.interp(tg, trk["t"][o], trk["gy"][o]))
    T = min(len(a[0]) for a in arrs.values())
    GX = np.column_stack([arrs[s][0][:T] for s in sids])
    GY = np.column_stack([arrs[s][1][:T] for s in sids])
    cx_a, cy_a = GX.mean(axis=1), GY.mean(axis=1)
    kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    vx_a = np.convolve(np.gradient(cx_a), kernel, mode="same")
    vy_a = np.convolve(np.gradient(cy_a), kernel, mode="same")
    speed_a = np.sqrt(vx_a ** 2 + vy_a ** 2)
    moving = speed_a > SPEED_THRESH
    if moving.sum() < 30:
        continue

    vnx = np.where(moving, vx_a / speed_a, 0)
    vny = np.where(moving, vy_a / speed_a, 0)
    proj = np.zeros((T, n_s))
    for i in range(n_s):
        proj[:, i] = (GX[:T, i] - cx_a) * vnx + (GY[:T, i] - cy_a) * vny
    leader_idx = np.argmax(proj, axis=1)
    counts = np.array([np.sum((leader_idx == i) & moving) for i in range(n_s)])
    fracs = counts / moving.sum()
    fracs_nz = fracs[fracs > 0]
    H = -np.sum(fracs_nz * np.log(fracs_nz)) / np.log(n_s) if n_s > 1 else 0

    leader_candidates.append((H, trial, tracks))

leader_candidates.sort(key=lambda x: x[0])

# Pick: lowest entropy, highest entropy, and one near median
picked = {}
if leader_candidates:
    picked["dominant"] = leader_candidates[0]
    picked["distributed"] = leader_candidates[-1]
    mid_idx = len(leader_candidates) // 2
    picked["mid"] = leader_candidates[mid_idx]

for i, (key, (H, trial, tracks)) in enumerate(picked.items(), 1):
    print(f"  {key}: assay {trial['assay']}, grp {trial['group_num']}, "
          f"entropy {H:.2f}")
    _render_leadership_gif(trial, tracks,
                           FIGDIR / f"leadership_animation_{i}.gif")


print("\nAll animations done!")
