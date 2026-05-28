#!/usr/bin/env python3
"""Generate publication-quality figures for the Beamer presentation.

Produces Nature-style plots with:
  - Jittered individual data points overlaid on box plots
  - Kruskal-Wallis and Mann-Whitney U statistical tests
  - Sample sizes (n) annotated per group
  - Median + IQR reporting
  - Clean, minimal axis styling
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter

from gps_analysis import (
    build_trials,
    build_tracks_cache,
    build_gps_cache,
    build_arena_transforms,
    load_trial_tracks,
    detect_site_visits,
    cumulative_path_length,
    SITE_GRID,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Nature-style plot defaults
# Nature figures use Arial/Helvetica; we fall back to DejaVu Sans if unavailable.
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
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
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
RADIUS = 0.2          # 0.2 grid units = 2 metres (strict, for completion)
RADIUS_EXPLORE = 0.5  # 0.5 grid units = 5 metres (generous, for site exploration count)
MIN_DWELL_S = 0.0
BAITED_SITES = {"A1", "A2", "A3"}  # success = finding all 3 baited sites
SMOOTH_WIN = 15
SPEED_THRESH = 0.000833  # 5 m/min at 10 Hz (np.gradient units: gu/sample)
TEST_CONFIGS = {"A", "B", "C", "D"}
EARLY_CUTOFF = 5.0    # minutes, for first-5-min analysis
PHASE2_START = "2026-02-17"  # only include second data-acquisition session
MOVE_SPEED_THRESH = 0.5  # m/min for occupancy filtering (individual sheep)
TIME_TO_FIRST_CAP = 15.0  # minutes cap for early-trial time-to-first plot

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
trials = build_trials()
gnss_cache = build_gps_cache(trials)
arena_transforms = build_arena_transforms()
tracks_cache = build_tracks_cache(trials, gnss_cache=gnss_cache,
                                  arena_transforms=arena_transforms)

# Groups 9 (CTRL_BARN) and 14 (CTRL_FAR from assay 1+) are control groups.
# Group 14 has only 1 test trial (assay 0); exclude it for consistency.
_CTRL_GROUPS = {9, 14}
test_trials = [t for t in trials
               if t["config"] in TEST_CONFIGS
               and t["assay"] is not None
               and isinstance(t["assay"], int)
               and t["date"] >= PHASE2_START
               and t["group_num"] not in _CTRL_GROUPS]

print(f"Total trials: {len(trials)}, Phase 2 test trials: {len(test_trials)}")

ref_sites = pd.read_csv(
    Path(__file__).resolve().parent.parent / "data" / "fitted_reward_sites.csv"
)
site_positions = {lbl: (x, y) for lbl, (x, y) in SITE_GRID.items()}


# ===========================================================================
# Helpers
# ===========================================================================
def _interp_to_1s(tracks, dur_min=25):
    # 10 Hz grid (0.1s steps) — matches raw GPS sample rate
    t_grid = np.arange(0, dur_min, 1 / 600)
    result = {}
    for sid, trk in tracks.items():
        order = np.argsort(trk["t"])
        t_s, gx_s, gy_s = trk["t"][order], trk["gx"][order], trk["gy"][order]
        mask = t_grid <= t_s.max()
        tg = t_grid[mask]
        result[sid] = {
            "gx": np.interp(tg, t_s, gx_s),
            "gy": np.interp(tg, t_s, gy_s),
            "t": tg,
        }
    return result


def _smooth_tracks(tracks, window=11, polyorder=3):
    """Savitzky-Golay smoothing on interpolated (regular-grid) tracks."""
    from scipy.signal import savgol_filter
    result = {}
    for sid, trk in tracks.items():
        n = len(trk["gx"])
        w = min(window, n if n % 2 == 1 else n - 1)
        if w < polyorder + 2:
            result[sid] = trk
            continue
        result[sid] = {
            "gx": savgol_filter(trk["gx"], w, polyorder),
            "gy": savgol_filter(trk["gy"], w, polyorder),
            "t": trk["t"],
        }
    return result


def _prepare_tracks(tracks, dur_min=25):
    """Interpolate to 1s grid then Savitzky-Golay smooth."""
    return _smooth_tracks(_interp_to_1s(tracks, dur_min))


def _moving_mask(gx, gy, t, thresh_m_min=0.5):
    """Boolean mask: True where individual sheep speed > threshold (m/min)."""
    if len(gx) < 2:
        return np.zeros(len(gx), dtype=bool)
    dt = np.diff(t)
    dt[dt == 0] = 1e-9
    speed = np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2) / dt * 10.0  # m/min
    return np.concatenate([[False], speed > thresh_m_min])


def _nature_box(ax, data_list, group_labels, colour, ylabel, title,
                show_points=True, show_n=True, ylim=None):
    """Draw a Nature-style box plot with jittered data points and n labels."""
    bp = ax.boxplot(
        data_list,
        tick_labels=group_labels,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="k", lw=1.2),
        whiskerprops=dict(color="#333", lw=0.8),
        capprops=dict(color="#333", lw=0.8),
    )
    for box in bp["boxes"]:
        box.set_facecolor(colour)
        box.set_alpha(0.35)
        box.set_edgecolor(colour)
        box.set_linewidth(0.8)

    if show_points:
        for i, d in enumerate(data_list):
            jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(d))
            ax.scatter(np.full(len(d), i + 1) + jitter, d,
                       s=10, alpha=0.55, color=colour, edgecolors="none", zorder=3)

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    if ylim:
        ax.set_ylim(ylim)

    # Place n-labels at the bottom of the axes using axes-fraction coordinates,
    # so they never collide with data or stat annotations at the top.
    if show_n:
        for i, d in enumerate(data_list):
            x_frac = (i + 1 - 0.5) / (len(data_list) + 0.0)  # approx x in axes coords
            ax.text(i + 1, 0.01, f"n={len(d)}", ha="center", va="bottom",
                    fontsize=5.5, color="#777",
                    transform=ax.get_xaxis_transform())


def _kw_annotation(ax, data_list, y_frac=0.95):
    """Add Kruskal-Wallis H-test result as text annotation."""
    clean = [d for d in data_list if len(d) >= 2]
    if len(clean) >= 2:
        H, p = stats.kruskal(*clean)
        pstr = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.97, y_frac, f"KW H = {H:.1f}, {pstr}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6, color="#444", style="italic")


def _mwu_annotation(ax, d1, d2, y_frac=0.88):
    """Add Mann-Whitney U test between two groups."""
    if len(d1) >= 2 and len(d2) >= 2:
        U, p = stats.mannwhitneyu(d1, d2, alternative="two-sided")
        pstr = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.98, y_frac, f"MWU {pstr}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="#333", style="italic")


# ===========================================================================
# 1. PATH LENGTH & COMPLETION TIME BY ASSAY
# ===========================================================================
print("Computing path length & completion metrics...")
path_records = []
for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks:
        continue
    visits = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)

    # Discovery times per site (earliest visit across all sheep)
    disc_times = {}
    for site, vlist in visits.items():
        if vlist:
            disc_times[site] = min(v[1] for v in vlist)

    # Completion = when the LAST of the 3 baited sites is found (strict 2m)
    baited_disc = {s: disc_times[s] for s in BAITED_SITES if s in disc_times}
    baited_found = len(baited_disc)
    completion_time = max(baited_disc.values()) if baited_found == 3 else None

    # Sites explored (generous 5m radius for general exploration count)
    visits_explore = detect_site_visits(tracks, trial["field"], RADIUS_EXPLORE, MIN_DWELL_S)
    sites_explored = sum(1 for vl in visits_explore.values() if vl)

    sheep_paths = []
    for sid, trk in tracks.items():
        cpl = cumulative_path_length(trk["gx"], trk["gy"]) * 10.0
        path_at_comp = None
        if completion_time is not None and len(trk["t"]) > 0:
            path_at_comp = float(np.interp(completion_time, trk["t"], cpl))
        sheep_paths.append(path_at_comp)

    mean_path_comp = np.nanmean([p for p in sheep_paths if p is not None]) if any(
        p is not None for p in sheep_paths) else None

    path_records.append({
        "assay": trial["assay"],
        "completion_time": completion_time,
        "mean_path_to_completion": mean_path_comp,
        "sites_found": sites_explored,
        "baited_found": baited_found,
    })

path_df = pd.DataFrame(path_records)
path_df_complete = path_df.dropna(subset=["completion_time"])

assays_sorted = sorted(path_df["assay"].unique())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
data_ct = [path_df_complete[path_df_complete["assay"] == a]["completion_time"].values for a in assays_sorted]
data_pl = [path_df_complete[path_df_complete["assay"] == a]["mean_path_to_completion"].values for a in assays_sorted]

_nature_box(ax1, data_ct, [str(a) for a in assays_sorted], PALETTE["blue"],
            "Time to all 3 baited sites (min)", "a  Baited-site completion time")
ax1.set_xlabel("Assay level")
_kw_annotation(ax1, data_ct)

_nature_box(ax2, data_pl, [str(a) for a in assays_sorted], PALETTE["orange"],
            "Mean path to completion (m)", "b  Search path length")
ax2.set_xlabel("Assay level")
_kw_annotation(ax2, data_pl)

# Assay 0 vs 4 Mann-Whitney
a0_ct = path_df_complete[path_df_complete["assay"] == 0]["completion_time"].values
a4_ct = path_df_complete[path_df_complete["assay"] == 4]["completion_time"].values
if len(a0_ct) >= 2 and len(a4_ct) >= 2:
    U, p = stats.mannwhitneyu(a0_ct, a4_ct, alternative="two-sided")
    pstr = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.3f}"
    ax1.text(0.98, 0.90, f"Assay 0 vs 4: {pstr}",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=6.5, color=PALETTE["red"], style="italic")

fig.tight_layout()
fig.savefig(FIGDIR / "path_completion_by_assay.pdf")
fig.savefig(FIGDIR / "path_completion_by_assay.png")
plt.close(fig)
print("  -> path_completion_by_assay")


# ===========================================================================
# 2. SITES FOUND BY ASSAY
# ===========================================================================
sites_df = pd.DataFrame(path_records)
fig, ax = plt.subplots(figsize=(5, 3.5))
data_sf = [sites_df[sites_df["assay"] == a]["sites_found"].values for a in assays_sorted]
_nature_box(ax, data_sf, [str(a) for a in assays_sorted], PALETTE["green"],
            "Unique sites explored (of 12, within 5 m)", "Sites explored per trial")
ax.set_xlabel("Assay level")
_kw_annotation(ax, data_sf)

# Spearman correlation: assay vs sites found
rho, p_sp = stats.spearmanr(sites_df["assay"], sites_df["sites_found"])
pstr = f"p = {p_sp:.2e}" if p_sp < 0.001 else f"p = {p_sp:.3f}"
ax.text(0.97, 0.85, f"Spearman $\\rho$ = {rho:.2f}, {pstr}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=6.5, color="#333", style="italic")

fig.tight_layout()
fig.savefig(FIGDIR / "sites_found_by_assay.pdf")
fig.savefig(FIGDIR / "sites_found_by_assay.png")
plt.close(fig)
print("  -> sites_found_by_assay")


# ===========================================================================
# 3. FLOCKING DYNAMICS
# ===========================================================================
print("Computing flocking dynamics...")
flock_records = []
for trial in test_trials:
    if trial["group_size"] < 2:
        continue
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if len(tracks) < 2:
        continue
    interp = _interp_to_1s(tracks)
    sids = sorted(interp.keys())
    n = len(sids)
    T = min(len(interp[s]["gx"]) for s in sids)

    GX = np.column_stack([interp[s]["gx"][:T] for s in sids])
    GY = np.column_stack([interp[s]["gy"][:T] for s in sids])

    nnd_arr = np.full(T, np.nan)
    for ti in range(T):
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((GX[ti, i] - GX[ti, j]) ** 2 + (GY[ti, i] - GY[ti, j]) ** 2)
                dists.append(d)
        if dists:
            nnd_arr[ti] = min(dists)

    cx = GX.mean(axis=1)
    cy = GY.mean(axis=1)
    spread_arr = np.mean(np.sqrt((GX - cx[:, None]) ** 2 + (GY - cy[:, None]) ** 2), axis=1)

    flock_records.append({
        "assay": trial["assay"],
        "mean_nnd": float(np.nanmean(nnd_arr)) * 10,
        "mean_spread": float(np.nanmean(spread_arr)) * 10,
    })

flock_df = pd.DataFrame(flock_records)
assays_f = sorted(flock_df["assay"].unique())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
data_nnd = [flock_df[flock_df["assay"] == a]["mean_nnd"].values for a in assays_f]
data_spr = [flock_df[flock_df["assay"] == a]["mean_spread"].values for a in assays_f]

_nature_box(ax1, data_nnd, [str(a) for a in assays_f], PALETTE["blue"],
            "Mean NND (m)", "a  Nearest-neighbour distance")
ax1.set_xlabel("Assay level")
_kw_annotation(ax1, data_nnd)

_nature_box(ax2, data_spr, [str(a) for a in assays_f], PALETTE["orange"],
            "Mean spread from centroid (m)", "b  Group spread")
ax2.set_xlabel("Assay level")
_kw_annotation(ax2, data_spr)

fig.tight_layout()
fig.savefig(FIGDIR / "flocking_by_assay.pdf")
fig.savefig(FIGDIR / "flocking_by_assay.png")
plt.close(fig)
print("  -> flocking_by_assay")


# ===========================================================================
# 4. LEADERSHIP ENTROPY & DOMINANT FRACTION
# ===========================================================================
print("Computing leadership metrics...")
leader_records = []
for trial in test_trials:
    if trial["group_size"] < 2:
        continue
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if len(tracks) < 2:
        continue
    interp = _interp_to_1s(tracks)
    sids = sorted(interp.keys())
    n_sheep = len(sids)
    T = min(len(interp[s]["gx"]) for s in sids)

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
        projections[:, i] = (GX[:T, i] - cx) * vnx + (GY[:T, i] - cy) * vny

    leader_idx = np.argmax(projections, axis=1)
    counts = np.array([np.sum((leader_idx == i) & moving) for i in range(n_sheep)])
    total_moving = moving.sum()
    if total_moving == 0:
        continue
    fracs = counts / total_moving

    fracs_nz = fracs[fracs > 0]
    entropy = -np.sum(fracs_nz * np.log(fracs_nz))
    norm_entropy = entropy / np.log(n_sheep) if n_sheep > 1 else 0.0

    leader_records.append({
        "assay": trial["assay"],
        "norm_entropy": norm_entropy,
        "dominant_fraction": float(fracs.max()),
    })

leader_df = pd.DataFrame(leader_records)
assays_l = sorted(leader_df["assay"].unique())

# Convert dominant fraction to percentage
leader_df["dominant_pct"] = leader_df["dominant_fraction"] * 100

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.6))
data_dom = [leader_df[leader_df["assay"] == a]["dominant_pct"].values for a in assays_l]

_nature_box(ax, data_dom, [str(a) for a in assays_l], PALETTE["purple"],
            "Top frontal leader (%)", "Frontal leadership concentration",
            ylim=(-2, 105))
ax.set_xlabel("Assay level")
# Fair share line (median group size is 4 → 25%)
ax.axhline(25, color="#999", ls=":", lw=0.7, zorder=0)
ax.text(0.02, 27, "fair share (4 sheep)", fontsize=5.5, color="#999",
        transform=ax.get_yaxis_transform())
_kw_annotation(ax, data_dom)

fig.tight_layout()
fig.savefig(FIGDIR / "leadership_by_assay.pdf")
fig.savefig(FIGDIR / "leadership_by_assay.png")
plt.close(fig)
print("  -> leadership_by_assay")


# ===========================================================================
# 4b. SITE RECRUITMENT ENTROPY
# ===========================================================================
from gps_analysis import detect_recruitment_episodes

print("Computing site recruitment metrics...")
recruit_records = []
for trial in test_trials:
    if trial["group_size"] < 2:
        continue
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if len(tracks) < 2:
        continue
    sids = sorted(tracks.keys())
    n_sheep = len(sids)
    visits = detect_site_visits(tracks, trial["field"], RADIUS_EXPLORE)
    episodes = detect_recruitment_episodes(visits)

    # Per-sheep: total followers attracted
    foll_counts = {s: 0 for s in sids}
    for ep in episodes:
        if ep["initiator"] in foll_counts:
            foll_counts[ep["initiator"]] += len(ep["followers"])
    total_foll = sum(foll_counts.values())
    if total_foll == 0:
        recruit_records.append({
            "assay": trial["assay"],
            "recruit_entropy": 0.0,
            "top_recruiter_frac": 0.0,
            "episodes": len(episodes),
        })
        continue
    rf = np.array([foll_counts[s] / total_foll for s in sids])
    rf_nz = rf[rf > 0]
    r_ent = -np.sum(rf_nz * np.log(rf_nz))
    max_ent = np.log(n_sheep) if n_sheep > 1 else 1.0
    recruit_records.append({
        "assay": trial["assay"],
        "recruit_entropy": r_ent / max_ent,
        "top_recruiter_frac": float(rf.max()),
        "episodes": len(episodes),
    })

recruit_df = pd.DataFrame(recruit_records)
assays_r = sorted(recruit_df["assay"].unique())

# Convert top fraction to percentage
recruit_df["top_recruiter_pct"] = recruit_df["top_recruiter_frac"] * 100

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.6))
data_rfrac = [recruit_df[recruit_df["assay"] == a]["top_recruiter_pct"].values for a in assays_r]

_nature_box(ax, data_rfrac, [str(a) for a in assays_r], PALETTE["green"],
            "Top recruiter (%)", "Recruitment concentration",
            ylim=(-2, 105))
ax.set_xlabel("Assay level")
ax.axhline(25, color="#999", ls=":", lw=0.7, zorder=0)
ax.text(0.02, 27, "fair share (4 sheep)", fontsize=5.5, color="#999",
        transform=ax.get_yaxis_transform())
_kw_annotation(ax, data_rfrac)

fig.tight_layout()
fig.savefig(FIGDIR / "recruitment_by_assay.pdf")
fig.savefig(FIGDIR / "recruitment_by_assay.png")
plt.close(fig)
print("  -> recruitment_by_assay")


# ===========================================================================
# 5. SPATIAL ENTROPY, REVISIT RATE, COVERAGE
# ===========================================================================
print("Computing spatial information...")
RES = 20
N_CELLS = RES * RES
spatial_records = []
for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks:
        continue

    all_gx = np.concatenate([trk["gx"] for trk in tracks.values()])
    all_gy = np.concatenate([trk["gy"] for trk in tracks.values()])

    ci = np.clip((all_gx / 5.0 * RES).astype(int), 0, RES - 1)
    cj = np.clip((all_gy / 5.0 * RES).astype(int), 0, RES - 1)
    cell_idx = ci * RES + cj
    counts = np.bincount(cell_idx, minlength=N_CELLS)
    p = counts[counts > 0] / counts.sum()
    entropy = -np.sum(p * np.log2(p)) / np.log2(N_CELLS)

    revisit_rates, coverages = [], []
    for sid, trk in tracks.items():
        ci_s = np.clip((trk["gx"] / 5.0 * RES).astype(int), 0, RES - 1)
        cj_s = np.clip((trk["gy"] / 5.0 * RES).astype(int), 0, RES - 1)
        cells_s = ci_s * RES + cj_s
        visited = set()
        revisits = 0
        for c in cells_s:
            if c in visited:
                revisits += 1
            visited.add(c)
        revisit_rates.append(revisits / max(len(cells_s), 1))
        coverages.append(len(visited) / N_CELLS)

    spatial_records.append({
        "assay": trial["assay"],
        "entropy": entropy,
        "revisit_rate": np.mean(revisit_rates),
        "coverage": np.mean(coverages),
    })

spatial_df = pd.DataFrame(spatial_records)
assays_s = sorted(spatial_df["assay"].unique())

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.6))

data_e = [spatial_df[spatial_df["assay"] == a]["entropy"].values for a in assays_s]
_nature_box(ax1, data_e, [str(a) for a in assays_s], PALETTE["blue"],
            "Norm. spatial entropy", "a  Spatial entropy")
ax1.set_xlabel("Assay level")
_kw_annotation(ax1, data_e)

data_r = [spatial_df[spatial_df["assay"] == a]["revisit_rate"].values for a in assays_s]
_nature_box(ax2, data_r, [str(a) for a in assays_s], PALETTE["orange"],
            "Mean revisit rate", "b  Revisit rate")
ax2.set_xlabel("Assay level")
_kw_annotation(ax2, data_r)

data_c = [spatial_df[spatial_df["assay"] == a]["coverage"].values * 100 for a in assays_s]
_nature_box(ax3, data_c, [str(a) for a in assays_s], PALETTE["green"],
            "Arena coverage (%)", "c  Coverage")
ax3.set_xlabel("Assay level")
_kw_annotation(ax3, data_c)

# Spearman on coverage vs assay
rho_c, p_c = stats.spearmanr(spatial_df["assay"], spatial_df["coverage"])
pstr_c = f"p = {p_c:.2e}" if p_c < 0.001 else f"p = {p_c:.3f}"
ax3.text(0.97, 0.85, f"$\\rho$ = {rho_c:.2f}, {pstr_c}",
         transform=ax3.transAxes, ha="right", va="top",
         fontsize=6.5, color="#333", style="italic")

fig.tight_layout()
fig.savefig(FIGDIR / "spatial_info_by_assay.pdf")
fig.savefig(FIGDIR / "spatial_info_by_assay.png")
plt.close(fig)
print("  -> spatial_info_by_assay")


# ===========================================================================
# 6. DISCOVERY EFFECTS
# ===========================================================================
print("Computing discovery effects...")
disc_effect_records = []
for trial in test_trials:
    if trial["group_size"] < 2:
        continue
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if len(tracks) < 2:
        continue
    visits = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)
    interp = _interp_to_1s(tracks)
    sids = sorted(interp.keys())
    T = min(len(interp[s]["gx"]) for s in sids)
    GX = np.column_stack([interp[s]["gx"][:T] for s in sids])
    GY = np.column_stack([interp[s]["gy"][:T] for s in sids])
    cx, cy = GX.mean(axis=1), GY.mean(axis=1)

    dt = 1 / 60
    speed = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2) / dt * 10.0
    kernel = np.ones(60) / 60
    speed_smooth = np.convolve(speed, kernel, mode="same")
    spread = np.mean(np.sqrt((GX - cx[:, None]) ** 2 + (GY - cy[:, None]) ** 2), axis=1) * 10

    win_steps = 60
    for site, vlist in visits.items():
        if not vlist:
            continue
        t_disc_min = min(v[1] for v in vlist)
        t_disc_idx = int(t_disc_min * 60)
        if t_disc_idx - win_steps < 0 or t_disc_idx + win_steps >= len(speed_smooth):
            continue
        speed_before = np.mean(speed_smooth[t_disc_idx - win_steps:t_disc_idx])
        speed_after = np.mean(speed_smooth[t_disc_idx:t_disc_idx + win_steps])
        spread_before = np.mean(spread[t_disc_idx - win_steps:t_disc_idx])
        spread_after = np.mean(spread[t_disc_idx:t_disc_idx + win_steps])

        disc_effect_records.append({
            "assay": trial["assay"],
            "site": site,
            "baited": site.startswith("A"),
            "speed_change_pct": 100 * (speed_after - speed_before) / max(speed_before, 0.01),
            "spread_change_pct": 100 * (spread_after - spread_before) / max(spread_before, 0.01),
        })

disc_df = pd.DataFrame(disc_effect_records)

if len(disc_df) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))
    baited = disc_df[disc_df["baited"]]
    unbaited = disc_df[~disc_df["baited"]]

    categories = ["Baited\n(A-sites)", "Unbaited\n(B/C/D)"]
    speed_data = [baited["speed_change_pct"].values, unbaited["speed_change_pct"].values]
    spread_data = [baited["spread_change_pct"].values, unbaited["spread_change_pct"].values]

    _nature_box(ax1, speed_data, categories, PALETTE["gold"],
                "Speed change (%)", "a  Speed change at discovery")
    ax1.axhline(0, color="k", ls="--", lw=0.7, zorder=0)
    _mwu_annotation(ax1, speed_data[0], speed_data[1])

    # Add median annotations
    for i, d in enumerate(speed_data):
        med = np.median(d)
        ax1.text(i + 1, med, f" {med:+.1f}%", fontsize=6.5, va="center", ha="left",
                 color=PALETTE["red"])

    _nature_box(ax2, spread_data, categories, PALETTE["grey"],
                "Spread change (%)", "b  Spread change at discovery")
    ax2.axhline(0, color="k", ls="--", lw=0.7, zorder=0)
    _mwu_annotation(ax2, spread_data[0], spread_data[1])

    for i, d in enumerate(spread_data):
        med = np.median(d)
        ax2.text(i + 1, med, f" {med:+.1f}%", fontsize=6.5, va="center", ha="left",
                 color=PALETTE["red"])

    fig.tight_layout()
    fig.savefig(FIGDIR / "discovery_effects.pdf")
    fig.savefig(FIGDIR / "discovery_effects.png")
    plt.close(fig)
    print("  -> discovery_effects")


# ===========================================================================
# 7. EXAMPLE TRAJECTORY + HEATMAP
# ===========================================================================
print("Generating example trajectory...")
example_trial = None
for a_pref in [3, 4, 2, 5, 1]:
    cands = [t for t in test_trials if t["assay"] == a_pref and t["group_size"] >= 4]
    if cands:
        example_trial = cands[0]
        break
if example_trial is None:
    example_trial = test_trials[0]

ex_tracks = load_trial_tracks(example_trial, tracks_cache=tracks_cache, apply_orient=True)
tab10 = plt.colormaps["tab10"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
ax1.set_facecolor("#f0f0f0")
for i, (sid, trk) in enumerate(sorted(ex_tracks.items())):
    ax1.scatter(trk["gx"], trk["gy"], s=0.3, alpha=0.4,
                color=tab10(i % 10), label=sid, rasterized=True)
for lbl, (sx, sy) in site_positions.items():
    c = PALETTE["gold"] if lbl.startswith("A") else "#bbb"
    ax1.add_patch(Circle((sx, sy), 0.15, fc=c, ec="k", lw=0.5, alpha=0.7, zorder=5))
    ax1.text(sx, sy, lbl, ha="center", va="center", fontsize=5, zorder=6)
ax1.set_xlim(-0.05, 5.05); ax1.set_ylim(-0.05, 5.05); ax1.set_aspect("equal")
ax1.set_xlabel("Grid X (10 m/unit)"); ax1.set_ylabel("Grid Y")
ax1.set_title(f"a  Trajectory: assay {example_trial['assay']}, "
              f"{example_trial['group_size']} sheep")
ax1.legend(fontsize=6, loc="lower right", framealpha=0.8, markerscale=8)

all_gx = np.concatenate([trk["gx"] for trk in ex_tracks.values()])
all_gy = np.concatenate([trk["gy"] for trk in ex_tracks.values()])
H, xe, ye = np.histogram2d(all_gx, all_gy, bins=100, range=[[0, 5], [0, 5]])
ax2.imshow(np.log1p(H.T), origin="lower", extent=[0, 5, 0, 5],
           cmap="Blues", aspect="equal")
for lbl, (sx, sy) in site_positions.items():
    c = PALETTE["gold"] if lbl.startswith("A") else "#bbb"
    ax2.add_patch(Circle((sx, sy), 0.15, fc=c, ec="k", lw=0.5, alpha=0.7, zorder=5))
    ax2.text(sx, sy, lbl, ha="center", va="center", fontsize=5, zorder=6)
ax2.set_xlabel("Grid X (10 m/unit)"); ax2.set_ylabel("Grid Y")
ax2.set_title("b  Occupancy heatmap (log scale)")

fig.tight_layout()
fig.savefig(FIGDIR / "example_trajectory.pdf")
fig.savefig(FIGDIR / "example_trajectory.png")
plt.close(fig)
print("  -> example_trajectory")


# ===========================================================================
# 8. ARENA LAYOUT
# ===========================================================================
print("Generating arena layout...")
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.set_facecolor("white")
ax.set_xlim(-0.3, 5.3); ax.set_ylim(-0.3, 5.3); ax.set_aspect("equal")
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.grid(True, alpha=0.15, lw=0.5)
ax.set_xlabel("Grid X (10 m/unit)"); ax.set_ylabel("Grid Y (10 m/unit)")
ax.set_title("Arena layout (canonical A-config)")

for lbl, (sx, sy) in site_positions.items():
    is_baited = lbl.startswith("A")
    c = PALETTE["gold"] if is_baited else "#ddd"
    ec = "#B8860B" if is_baited else "#888"
    ax.add_patch(Circle((sx, sy), 0.28, fc=c, ec=ec, lw=1.2, alpha=0.85, zorder=5))
    ax.text(sx, sy, lbl, ha="center", va="center", fontsize=7, fontweight="bold", zorder=6)

ax.add_patch(plt.Rectangle((0, 0), 5, 5, fill=False, ec="k", lw=1.8, zorder=4))
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["gold"],
           markeredgecolor="#B8860B", markersize=10, label="Baited (A1, A2, A3)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#ddd",
           markeredgecolor="#888", markersize=10, label="Unbaited (9 sites)"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "arena_layout.pdf")
fig.savefig(FIGDIR / "arena_layout.png")
plt.close(fig)
print("  -> arena_layout")


# ===========================================================================
# 9. PIONEER VISITS
# ===========================================================================
print("Computing pioneer visits...")
pioneer_records = []
for trial in test_trials:
    if trial["group_size"] < 2:
        continue
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if len(tracks) < 2:
        continue
    visits = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)
    pioneers = {}
    for site, vlist in visits.items():
        if vlist:
            pioneers[site] = min(vlist, key=lambda v: v[1])[0]
    for sid in tracks.keys():
        count = sum(1 for p in pioneers.values() if p == sid)
        pioneer_records.append({
            "assay": trial["assay"],
            "sheep_id": sid,
            "pioneer_count": count,
        })

pioneer_df = pd.DataFrame(pioneer_records)
if len(pioneer_df) > 0:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    assays_p = sorted(pioneer_df["assay"].unique())
    data_p = [pioneer_df[pioneer_df["assay"] == a]["pioneer_count"].values for a in assays_p]
    _nature_box(ax, data_p, [str(a) for a in assays_p], PALETTE["purple"],
                "Pioneer discoveries per sheep", "Pioneer site discoveries")
    ax.set_xlabel("Assay level")
    _kw_annotation(ax, data_p)
    fig.tight_layout()
    fig.savefig(FIGDIR / "pioneer_visits_by_assay.pdf")
    fig.savefig(FIGDIR / "pioneer_visits_by_assay.png")
    plt.close(fig)
    print("  -> pioneer_visits_by_assay")


# ===========================================================================
# 10. SUMMARY STATISTICS
# ===========================================================================
print("Computing summary table...")
summary = []
for a in assays_sorted:
    ct = path_df_complete[path_df_complete["assay"] == a]["completion_time"]
    pl = path_df_complete[path_df_complete["assay"] == a]["mean_path_to_completion"]
    sf = sites_df[sites_df["assay"] == a]["sites_found"]
    fl = flock_df[flock_df["assay"] == a]
    ld = leader_df[leader_df["assay"] == a]
    sp = spatial_df[spatial_df["assay"] == a]
    n_tot = len(sites_df[sites_df["assay"] == a])
    summary.append({
        "Assay": a,
        "N": n_tot,
        "Compl_median": f"{ct.median():.1f}" if len(ct) > 0 else "--",
        "Compl_iqr": f"[{ct.quantile(0.25):.1f}, {ct.quantile(0.75):.1f}]" if len(ct) > 0 else "",
        "Path_median": f"{pl.median():.0f}" if len(pl) > 0 else "--",
        "Path_iqr": f"[{pl.quantile(0.25):.0f}, {pl.quantile(0.75):.0f}]" if len(pl) > 0 else "",
        "Sites_median": f"{sf.median():.0f}" if len(sf) > 0 else "--",
        "NND_median": f"{fl['mean_nnd'].median():.1f}" if len(fl) > 0 else "--",
        "Entropy_median": f"{ld['norm_entropy'].median():.2f}" if len(ld) > 0 else "--",
        "Coverage_median": f"{sp['coverage'].median() * 100:.0f}" if len(sp) > 0 else "--",
    })
summary_df = pd.DataFrame(summary)
summary_df.to_csv(FIGDIR / "summary_statistics.csv", index=False)
print(summary_df.to_string(index=False))


# ===========================================================================
# 11. CONFIGURATION COMPARISON
# ===========================================================================
print("Computing config comparison...")
config_path_records = []
for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks:
        continue
    visits = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)
    disc_times = {s: min(v[1] for v in vl) for s, vl in visits.items() if vl}
    baited_disc = {s: disc_times[s] for s in BAITED_SITES if s in disc_times}
    ct = max(baited_disc.values()) if len(baited_disc) == 3 else None
    sheep_paths = []
    for sid, trk in tracks.items():
        cpl = cumulative_path_length(trk["gx"], trk["gy"]) * 10.0
        if ct is not None and len(trk["t"]) > 0:
            sheep_paths.append(float(np.interp(ct, trk["t"], cpl)))
    mp = np.nanmean(sheep_paths) if sheep_paths else None
    config_path_records.append({
        "config": trial["config"],
        "completion_time": ct,
        "mean_path": mp,
    })

config_df = pd.DataFrame(config_path_records).dropna(subset=["completion_time"])
configs_sorted = sorted(config_df["config"].unique())
colours_cfg = {"A": PALETTE["blue"], "B": PALETTE["orange"],
               "C": PALETTE["green"], "D": PALETTE["red"]}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))
data_ct_cfg = [config_df[config_df["config"] == c]["completion_time"].values for c in configs_sorted]
data_pl_cfg = [config_df[config_df["config"] == c]["mean_path"].values for c in configs_sorted]

for i, (c, d) in enumerate(zip(configs_sorted, data_ct_cfg)):
    bp = ax1.boxplot([d], positions=[i + 1], widths=0.55, patch_artist=True,
                     showfliers=False, tick_labels=[c],
                     medianprops=dict(color="k", lw=1.2),
                     whiskerprops=dict(color="#333", lw=0.8),
                     capprops=dict(color="#333", lw=0.8))
    bp["boxes"][0].set_facecolor(colours_cfg[c])
    bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][0].set_edgecolor(colours_cfg[c])
    jitter = np.random.default_rng(42 + i).uniform(-0.12, 0.12, len(d))
    ax1.scatter(np.full(len(d), i + 1) + jitter, d,
                s=12, alpha=0.5, color=colours_cfg[c], edgecolors="none", zorder=3)
    ax1.text(i + 1, d.max() + 0.5, f"n={len(d)}", ha="center", fontsize=6, color="#555")

ax1.set_ylabel("Completion time (min)")
ax1.set_xlabel("Configuration")
ax1.set_title("a  Completion time by config")
_kw_annotation(ax1, data_ct_cfg)

for i, (c, d) in enumerate(zip(configs_sorted, data_pl_cfg)):
    bp = ax2.boxplot([d], positions=[i + 1], widths=0.55, patch_artist=True,
                     showfliers=False, tick_labels=[c],
                     medianprops=dict(color="k", lw=1.2),
                     whiskerprops=dict(color="#333", lw=0.8),
                     capprops=dict(color="#333", lw=0.8))
    bp["boxes"][0].set_facecolor(colours_cfg[c])
    bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][0].set_edgecolor(colours_cfg[c])
    jitter = np.random.default_rng(42 + i).uniform(-0.12, 0.12, len(d))
    ax2.scatter(np.full(len(d), i + 1) + jitter, d,
                s=12, alpha=0.5, color=colours_cfg[c], edgecolors="none", zorder=3)
    ax2.text(i + 1, d.max() + 5, f"n={len(d)}", ha="center", fontsize=6, color="#555")

ax2.set_ylabel("Mean path to completion (m)")
ax2.set_xlabel("Configuration")
ax2.set_title("b  Path length by config")
_kw_annotation(ax2, data_pl_cfg)

fig.tight_layout()
fig.savefig(FIGDIR / "config_comparison.pdf")
fig.savefig(FIGDIR / "config_comparison.png")
plt.close(fig)
print("  -> config_comparison")

# ===========================================================================
# 12. FIRST-5-MINUTE ANALYSIS
# ===========================================================================
print("Computing first-5-minute metrics...")
early_records = []
for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks:
        continue
    visits_strict = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)
    visits_explore = detect_site_visits(tracks, trial["field"], RADIUS_EXPLORE, MIN_DWELL_S)

    # Baited sites found within first 5 min (2m — must reach reward)
    baited_early = sum(
        1 for s in BAITED_SITES
        if s in visits_strict and visits_strict[s]
        and min(v[1] for v in visits_strict[s]) <= EARLY_CUTOFF
    )

    # All sites explored within first 5 min (5m — awareness)
    all_early = sum(
        1 for s, vl in visits_explore.items()
        if vl and min(v[1] for v in vl) <= EARLY_CUTOFF
    )

    # Mean path length at t=5 min
    paths_5 = []
    for sid, trk in tracks.items():
        cpl = cumulative_path_length(trk["gx"], trk["gy"]) * 10.0
        if trk["t"].max() >= EARLY_CUTOFF:
            paths_5.append(float(np.interp(EARLY_CUTOFF, trk["t"], cpl)))

    # Time to first site visit (5m — exploration awareness)
    all_disc = [min(v[1] for v in vl) for vl in visits_explore.values() if vl]
    t_first = min(all_disc) if all_disc else None

    early_records.append({
        "assay": trial["assay"],
        "baited_in_5min": baited_early,
        "sites_in_5min": all_early,
        "path_5min": np.mean(paths_5) if paths_5 else None,
        "time_to_first": t_first,
    })

early_df = pd.DataFrame(early_records)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.6))
assays_e = sorted(early_df["assay"].unique())

data_b5 = [early_df[early_df["assay"] == a]["baited_in_5min"].values for a in assays_e]
_nature_box(ax1, data_b5, [str(a) for a in assays_e], PALETTE["gold"],
            "Baited sites found (of 3)", "a  Baited sites in first 5 min")
ax1.set_xlabel("Assay level")
ax1.set_ylim(-0.3, 3.5)
_kw_annotation(ax1, data_b5)
rho_b5, p_b5 = stats.spearmanr(early_df["assay"], early_df["baited_in_5min"])
pstr = f"p = {p_b5:.2e}" if p_b5 < 0.001 else f"p = {p_b5:.3f}"
ax1.text(0.97, 0.85, f"$\\rho$ = {rho_b5:.2f}, {pstr}",
         transform=ax1.transAxes, ha="right", va="top",
         fontsize=6.5, color="#333", style="italic")

data_p5 = [early_df[early_df["assay"] == a]["path_5min"].dropna().values for a in assays_e]
_nature_box(ax2, data_p5, [str(a) for a in assays_e], PALETTE["blue"],
            "Path length at 5 min (m)", "b  Path length in first 5 min")
ax2.set_xlabel("Assay level")
_kw_annotation(ax2, data_p5)

data_tf_raw = [early_df[early_df["assay"] == a]["time_to_first"].dropna().values for a in assays_e]
data_tf = [np.clip(d, 0, TIME_TO_FIRST_CAP) for d in data_tf_raw]
_nature_box(ax3, data_tf, [str(a) for a in assays_e], PALETTE["green"],
            "Time to first site visit (min)", "c  Latency to first discovery")
ax3.set_xlabel("Assay level")
ax3.set_ylim(0, TIME_TO_FIRST_CAP + 1)
_kw_annotation(ax3, data_tf_raw)  # stats on unclipped data
n_capped = sum(1 for d in data_tf_raw for v in d if v > TIME_TO_FIRST_CAP)
if n_capped > 0:
    ax3.text(0.97, 0.78, f"{n_capped} trials >{TIME_TO_FIRST_CAP:.0f} min capped",
             transform=ax3.transAxes, ha="right", va="top",
             fontsize=5.5, color="#999", style="italic")

fig.tight_layout()
fig.savefig(FIGDIR / "early_trial_metrics.pdf")
fig.savefig(FIGDIR / "early_trial_metrics.png")
plt.close(fig)
print("  -> early_trial_metrics")


# ===========================================================================
# 13. TRAJECTORY COMPARISON ACROSS ASSAYS
# ===========================================================================
print("Generating trajectory comparison panels...")
TARGET_ASSAYS_TRAJ = [0, 2, 4, 7]
tab10 = plt.colormaps["tab10"]

# For each target assay, pick representative trial (group_size>=3, closest to median completion)
chosen_trials = []
for ta in TARGET_ASSAYS_TRAJ:
    cands = [t for t in test_trials if t["assay"] == ta and t["group_size"] >= 3]
    if not cands:
        cands = [t for t in test_trials if t["assay"] == ta]
    if not cands:
        chosen_trials.append(None)
        continue

    # Compute completion for each candidate
    cand_cts = []
    for c in cands:
        trk = load_trial_tracks(c, tracks_cache=tracks_cache, apply_orient=True)
        if not trk:
            cand_cts.append((c, None, 0))
            continue
        vis = detect_site_visits(trk, c["field"], RADIUS, MIN_DWELL_S)
        bd = {s: min(v[1] for v in vl) for s, vl in vis.items() if vl and s in BAITED_SITES}
        ct_val = max(bd.values()) if len(bd) == 3 else None
        cand_cts.append((c, ct_val, len(bd)))

    # Prefer trials that completed; pick closest to median
    completed = [(c, ct) for c, ct, _ in cand_cts if ct is not None]
    if completed:
        med_ct = np.median([ct for _, ct in completed])
        best = min(completed, key=lambda x: abs(x[1] - med_ct))[0]
    else:
        # Pick trial with most baited sites found
        best = max(cand_cts, key=lambda x: x[2])[0]
    chosen_trials.append(best)

fig, axes = plt.subplots(2, 2, figsize=(9, 9))
panel_labels = ["a", "b", "c", "d"]
assay_labels = {0: "Assay 0 (na\u00efve)", 2: "Assay 2", 4: "Assay 4", 7: "Assay 7 (experienced)"}

for idx, (ax, ta, trial) in enumerate(zip(axes.flat, TARGET_ASSAYS_TRAJ, chosen_trials)):
    ax.set_facecolor("#f0f0f0")
    if trial is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        trk = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
        for i, (sid, t_data) in enumerate(sorted(trk.items())):
            ax.scatter(t_data["gx"], t_data["gy"], s=0.3, alpha=0.4,
                       color=tab10(i % 10), label=sid, rasterized=True)
        ax.legend(fontsize=5, loc="lower right", framealpha=0.7, markerscale=8,
                  handletextpad=0.2, borderpad=0.3)

    for lbl, (sx, sy) in site_positions.items():
        c = PALETTE["gold"] if lbl.startswith("A") else "#ccc"
        ax.add_patch(Circle((sx, sy), 0.18, fc=c, ec="k", lw=0.5, alpha=0.7, zorder=5))
        ax.text(sx, sy, lbl, ha="center", va="center", fontsize=5, zorder=6)

    ax.set_xlim(-0.05, 5.05)
    ax.set_ylim(-0.05, 5.05)
    ax.set_aspect("equal")
    ax.set_xlabel("Grid X (10 m/unit)")
    ax.set_ylabel("Grid Y")
    n_sheep = trial["group_size"] if trial else "?"
    ax.set_title(f"{panel_labels[idx]}  {assay_labels[ta]}  (n={n_sheep} sheep)")

fig.tight_layout()
fig.savefig(FIGDIR / "trajectory_comparison.pdf")
fig.savefig(FIGDIR / "trajectory_comparison.png")
plt.close(fig)
print("  -> trajectory_comparison")


# ===========================================================================
# 14. LEARNING CURVE: SUCCESS RATE + COMPLETION TIME
# ===========================================================================
print("Computing learning curve...")
from scipy.optimize import curve_fit

# Exclude assay 0 from learning curve: no baited sites in assay 0
lc_assays = [a for a in assays_sorted if a != 0]

# Success rate: fraction of trials that found all 3 baited sites
success_rates = []
median_cts = []
iqr_lo, iqr_hi = [], []
n_per_assay = []
for a in lc_assays:
    sub = path_df[path_df["assay"] == a]
    n_tot = len(sub)
    n_success = sub["completion_time"].notna().sum()
    rate = n_success / n_tot if n_tot > 0 else 0
    success_rates.append(rate)
    n_per_assay.append(n_tot)

    ct_vals = sub["completion_time"].dropna()
    if len(ct_vals) > 0:
        median_cts.append(ct_vals.median())
        iqr_lo.append(ct_vals.quantile(0.25))
        iqr_hi.append(ct_vals.quantile(0.75))
    else:
        median_cts.append(np.nan)
        iqr_lo.append(np.nan)
        iqr_hi.append(np.nan)

assays_arr = np.array(lc_assays, dtype=float)
rates_arr = np.array(success_rates)
med_arr = np.array(median_cts)
iqr_lo_arr = np.array(iqr_lo)
iqr_hi_arr = np.array(iqr_hi)

# Exponential decay fit to median completion time
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

fit_mask = ~np.isnan(med_arr)
fit_text = ""
if fit_mask.sum() >= 3:
    try:
        popt, _ = curve_fit(exp_decay, assays_arr[fit_mask], med_arr[fit_mask],
                            p0=[10, 0.5, 1.5], bounds=([0, 0, 0], [50, 5, 20]),
                            maxfev=5000)
        x_fit = np.linspace(0, 7, 100)
        y_fit = exp_decay(x_fit, *popt)
        ss_res = np.sum((med_arr[fit_mask] - exp_decay(assays_arr[fit_mask], *popt)) ** 2)
        ss_tot = np.sum((med_arr[fit_mask] - med_arr[fit_mask].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fit_text = f"Fit: {popt[0]:.1f}·exp(−{popt[1]:.2f}·x) + {popt[2]:.1f},  R² = {r2:.2f}"
    except RuntimeError:
        x_fit, y_fit, r2 = None, None, None
        fit_text = "Exponential fit did not converge"
else:
    x_fit, y_fit, r2 = None, None, None

fig, ax1 = plt.subplots(figsize=(7, 4.2))

# Bar chart: success rate
bars = ax1.bar(assays_arr, rates_arr * 100, width=0.6, color=PALETTE["blue"],
               alpha=0.45, edgecolor=PALETTE["blue"], lw=0.8, zorder=2)
for i, (a, r, n) in enumerate(zip(assays_arr, rates_arr, n_per_assay)):
    ax1.text(a, r * 100 + 1.5, f"{r:.0%}", ha="center", fontsize=7, color=PALETTE["blue"])
    ax1.text(a, -4, f"n={n}", ha="center", fontsize=6, color="#555")

ax1.set_ylabel("Success rate (%)", color=PALETTE["blue"])
ax1.set_xlabel("Assay level")
ax1.set_ylim(-8, 108)
ax1.tick_params(axis="y", colors=PALETTE["blue"])

# Secondary axis: median completion time
ax2 = ax1.twinx()
valid = ~np.isnan(med_arr)
ax2.fill_between(assays_arr[valid], iqr_lo_arr[valid], iqr_hi_arr[valid],
                 alpha=0.15, color=PALETTE["red"], zorder=1)
ax2.plot(assays_arr[valid], med_arr[valid], "o-", color=PALETTE["red"],
         lw=1.5, markersize=5, zorder=3, label="Median completion time")

if x_fit is not None:
    ax2.plot(x_fit, y_fit, "--", color=PALETTE["red"], alpha=0.5, lw=1, zorder=2)

ax2.set_ylabel("Median completion time (min)", color=PALETTE["red"])
ax2.tick_params(axis="y", colors=PALETTE["red"])
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(PALETTE["red"])
ax2.spines["right"].set_linewidth(0.8)

ax1.set_title("Learning curve: baited-site success rate and completion time")
if fit_text:
    ax1.text(0.98, 0.98, fit_text, transform=ax1.transAxes, ha="right", va="top",
             fontsize=6.5, color="#333", style="italic")

# Legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc=PALETTE["blue"], alpha=0.45, ec=PALETTE["blue"],
                  label="Success rate (%)"),
    Line2D([0], [0], color=PALETTE["red"], marker="o", markersize=4, lw=1.5,
           label="Median completion (min)"),
]
ax1.legend(handles=legend_elements, loc="center right", fontsize=7, frameon=False)

fig.tight_layout()
fig.savefig(FIGDIR / "learning_curve.pdf")
fig.savefig(FIGDIR / "learning_curve.png")
plt.close(fig)
print("  -> learning_curve")
if fit_text:
    print(f"     {fit_text}")


# ===========================================================================
# 15. SITE RESIDENCE TIME: BAITED vs UNBAITED (Giving-Up Behaviour)
# ===========================================================================
print("Computing site residence times...")
residence_records = []
for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks:
        continue
    # Use 5m radius for residence time — captures how long sheep linger near sites
    # (the strict 2m radius mostly captures transient pass-throughs)
    visits = detect_site_visits(tracks, trial["field"], RADIUS_EXPLORE, MIN_DWELL_S)

    for site_label, vlist in visits.items():
        is_baited = site_label in BAITED_SITES
        for sheep_id, t_entry, t_exit in vlist:
            dwell_s = (t_exit - t_entry) * 60  # convert min to seconds
            residence_records.append({
                "assay": trial["assay"],
                "site": site_label,
                "baited": is_baited,
                "dwell_s": dwell_s,
            })

res_df = pd.DataFrame(residence_records)

if len(res_df) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    # Panel a: residence time by site type (baited vs unbaited), all assays pooled
    # Cap display at 300s to keep comparison readable; ~1% of visits exceed this
    DWELL_CAP = 300  # 5 min
    baited_dwell = res_df[res_df["baited"]]["dwell_s"].values
    unbaited_dwell = res_df[~res_df["baited"]]["dwell_s"].values
    baited_clipped = np.clip(baited_dwell, 0, DWELL_CAP)
    unbaited_clipped = np.clip(unbaited_dwell, 0, DWELL_CAP)
    _nature_box(ax1, [baited_clipped, unbaited_clipped],
                ["Baited\n(A-sites)", "Unbaited\n(B/C/D)"], PALETTE["gold"],
                "Visit duration (s)", "a  Residence time by site type (5 m radius)")
    ax1.set_ylim(0, DWELL_CAP + 20)
    # Use unclipped data for stats
    _mwu_annotation(ax1, baited_dwell, unbaited_dwell)
    for i, (d, label) in enumerate([(baited_dwell, "Baited"), (unbaited_dwell, "Unbait")]):
        ax1.text(i + 1, np.median(d) + 8, f"Md={np.median(d):.0f}s",
                 fontsize=6.5, va="bottom", ha="center", color="#333")
    n_over = sum(baited_dwell > DWELL_CAP) + sum(unbaited_dwell > DWELL_CAP)
    ax1.text(0.97, 0.78, f"{n_over} visits >{DWELL_CAP}s clipped",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=5.5, color="#999", style="italic")

    # Panel b: unbaited giving-up time by assay (does it decrease with learning?)
    unbaited_res = res_df[~res_df["baited"]]
    assays_r = sorted(unbaited_res["assay"].unique())
    data_gup = [np.clip(unbaited_res[unbaited_res["assay"] == a]["dwell_s"].values, 0, DWELL_CAP)
                for a in assays_r]
    _nature_box(ax2, data_gup, [str(a) for a in assays_r], PALETTE["grey"],
                "Unbaited visit duration (s)", "b  Giving-up time at unbaited sites (5 m)")
    ax2.set_xlabel("Assay level")
    ax2.set_ylim(0, DWELL_CAP + 20)
    # Stats on unclipped data
    data_gup_raw = [unbaited_res[unbaited_res["assay"] == a]["dwell_s"].values for a in assays_r]
    _kw_annotation(ax2, data_gup_raw)
    rho_g, p_g = stats.spearmanr(unbaited_res["assay"], unbaited_res["dwell_s"])
    pstr = f"p = {p_g:.2e}" if p_g < 0.001 else f"p = {p_g:.3f}"
    ax2.text(0.97, 0.85, f"$\\rho$ = {rho_g:.2f}, {pstr}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=6.5, color="#333", style="italic")

    fig.tight_layout()
    fig.savefig(FIGDIR / "residence_time.pdf")
    fig.savefig(FIGDIR / "residence_time.png")
    plt.close(fig)
    print("  -> residence_time")


# ===========================================================================
# 16. STRAIGHTNESS INDEX (Beeline / Path Length)
# ===========================================================================
print("Computing straightness index...")
straight_records = []
for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks:
        continue
    visits = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)

    # Compute straightness from start to first baited site visit
    baited_disc = {}
    for s in BAITED_SITES:
        if s in visits and visits[s]:
            baited_disc[s] = min(v[1] for v in visits[s])

    if not baited_disc:
        # No baited sites found; compute whole-trial straightness
        for sid, trk in tracks.items():
            if len(trk["gx"]) < 2:
                continue
            beeline = np.sqrt((trk["gx"][-1] - trk["gx"][0]) ** 2 +
                              (trk["gy"][-1] - trk["gy"][0]) ** 2) * 10
            path = cumulative_path_length(trk["gx"], trk["gy"])[-1] * 10
            si = beeline / max(path, 0.01)
            straight_records.append({
                "assay": trial["assay"],
                "straightness": si,
                "segment": "whole_trial",
            })
        continue

    # Straightness from start to first baited site found
    first_site = min(baited_disc, key=baited_disc.get)
    t_first = baited_disc[first_site]
    sx_site, sy_site = site_positions[first_site]

    for sid, trk in tracks.items():
        mask = trk["t"] <= t_first
        gx_seg, gy_seg = trk["gx"][mask], trk["gy"][mask]
        if len(gx_seg) < 2:
            continue
        beeline = np.sqrt((sx_site - gx_seg[0]) ** 2 +
                          (sy_site - gy_seg[0]) ** 2) * 10
        path = cumulative_path_length(gx_seg, gy_seg)[-1] * 10
        si = beeline / max(path, 0.01)
        straight_records.append({
            "assay": trial["assay"],
            "straightness": si,
            "segment": "to_first_baited",
        })

straight_df = pd.DataFrame(straight_records)
str_to_site = straight_df[straight_df["segment"] == "to_first_baited"]

if len(str_to_site) > 0:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    assays_st = sorted(str_to_site["assay"].unique())
    data_st = [np.clip(str_to_site[str_to_site["assay"] == a]["straightness"].values, 0, 1)
               for a in assays_st]
    _nature_box(ax, data_st, [str(a) for a in assays_st], PALETTE["blue"],
                "Straightness index (beeline/path)", "Path straightness to first baited site",
                show_n=False, ylim=(-0.05, 1.25))
    ax.set_xlabel("Assay level")
    for i, d in enumerate(data_st):
        ax.text(i + 1, 1.12, f"n={len(d)}", ha="center", fontsize=5.5, color="#777")
    ax.axhline(1.0, color="#999", ls=":", lw=0.7, zorder=0)
    ax.text(0.5, 1.04, "perfectly direct", ha="center", fontsize=5.5, color="#999",
            transform=ax.get_yaxis_transform())
    _kw_annotation(ax, data_st)
    rho_st, p_st = stats.spearmanr(str_to_site["assay"], str_to_site["straightness"])
    pstr = f"p = {p_st:.2e}" if p_st < 0.001 else f"p = {p_st:.3f}"
    ax.text(0.97, 0.85, f"$\\rho$ = {rho_st:.2f}, {pstr}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color="#333", style="italic")

    fig.tight_layout()
    fig.savefig(FIGDIR / "straightness_index.pdf")
    fig.savefig(FIGDIR / "straightness_index.png")
    plt.close(fig)
    print("  -> straightness_index")


# ===========================================================================
# 16b. PER-GROUP LEADERSHIP PROFILES
# ===========================================================================
print("Computing per-group leadership profiles...")
from collections import defaultdict

grp_entropy_by_assay = defaultdict(dict)  # grp -> assay -> entropy
grp_dominant_sheep = defaultdict(list)     # grp -> [(assay, sheep, frac)]

for trial in test_trials:
    if trial["group_size"] < 2:
        continue
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if len(tracks) < 2:
        continue
    interp = _interp_to_1s(tracks)
    sids = sorted(interp.keys())
    n_sheep = len(sids)
    T = min(len(interp[s]["gx"]) for s in sids)

    GX = np.column_stack([interp[s]["gx"][:T] for s in sids])
    GY = np.column_stack([interp[s]["gy"][:T] for s in sids])
    cx, cy = GX.mean(axis=1), GY.mean(axis=1)

    kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    vx = np.convolve(np.gradient(cx), kernel, mode="same")
    vy = np.convolve(np.gradient(cy), kernel, mode="same")
    speed = np.sqrt(vx ** 2 + vy ** 2)
    moving = speed > SPEED_THRESH
    if moving.sum() < 10:
        continue
    vnx = np.where(moving, vx / speed, 0.0)
    vny = np.where(moving, vy / speed, 0.0)

    projections = np.zeros((T, n_sheep))
    for i in range(n_sheep):
        projections[:, i] = (GX[:T, i] - cx) * vnx + (GY[:T, i] - cy) * vny

    leader_idx = np.argmax(projections, axis=1)
    counts = np.array([np.sum((leader_idx == i) & moving) for i in range(n_sheep)])
    fracs = counts / moving.sum()
    fracs_nz = fracs[fracs > 0]
    H = -np.sum(fracs_nz * np.log(fracs_nz)) / np.log(n_sheep) if n_sheep > 1 else 0.0

    g = trial["group_num"]
    a = trial["assay"]
    grp_entropy_by_assay[g][a] = H
    dom_idx = np.argmax(fracs)
    grp_dominant_sheep[g].append((a, sids[dom_idx], fracs[dom_idx]))

# Figure: per-group entropy across assays (heatmap style)
groups_sorted = sorted(grp_entropy_by_assay.keys())
assays_all = sorted(set(a for d in grp_entropy_by_assay.values() for a in d))

# Build matrix
mat = np.full((len(groups_sorted), len(assays_all)), np.nan)
for i, g in enumerate(groups_sorted):
    for j, a in enumerate(assays_all):
        if a in grp_entropy_by_assay[g]:
            mat[i, j] = grp_entropy_by_assay[g][a]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                gridspec_kw={"width_ratios": [3, 1.2]})

# Panel a: heatmap of entropy per group × assay
im = ax1.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                interpolation="nearest")
ax1.set_xticks(range(len(assays_all)))
ax1.set_xticklabels([str(a) for a in assays_all])
ax1.set_yticks(range(len(groups_sorted)))
ax1.set_yticklabels([f"Grp {g}" for g in groups_sorted])
ax1.set_xlabel("Assay level")
ax1.set_title("a  Leadership entropy by group and assay", fontweight="bold", fontsize=10)
cb = fig.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
cb.set_label("Normalised entropy", fontsize=8)

# Annotate cells with values
for i in range(len(groups_sorted)):
    for j in range(len(assays_all)):
        v = mat[i, j]
        if not np.isnan(v):
            color = "white" if v < 0.4 else "black"
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=5.5, color=color)

# Panel b: median entropy per group (bar chart)
med_per_grp = [np.nanmedian(mat[i, :]) for i in range(len(groups_sorted))]
colors_bar = [PALETTE["red"] if m < 0.6 else PALETTE["orange"] if m < 0.75
              else PALETTE["blue"] for m in med_per_grp]
ax2.barh(range(len(groups_sorted)), med_per_grp, color=colors_bar, alpha=0.7,
         edgecolor=[c for c in colors_bar], linewidth=0.8)
ax2.set_yticks(range(len(groups_sorted)))
ax2.set_yticklabels([f"Grp {g}" for g in groups_sorted])
ax2.set_xlabel("Median entropy")
ax2.set_xlim(0, 1.05)
ax2.axvline(0.5, color="#999", ls=":", lw=0.7)
ax2.set_title("b  Median across assays", fontweight="bold", fontsize=10)
ax2.invert_yaxis()
# Match y-axis order with heatmap
ax2.set_ylim(len(groups_sorted) - 0.5, -0.5)

fig.tight_layout()
fig.savefig(FIGDIR / "group_leadership_profiles.pdf")
fig.savefig(FIGDIR / "group_leadership_profiles.png")
plt.close(fig)
print("  -> group_leadership_profiles")

# Print summary of notable groups
for g in groups_sorted:
    med = np.nanmedian([grp_entropy_by_assay[g].get(a, np.nan) for a in assays_all])
    if med < 0.6:
        dominant_counts = defaultdict(int)
        for a, sheep, frac in grp_dominant_sheep[g]:
            dominant_counts[sheep] += 1
        most_common = max(dominant_counts, key=dominant_counts.get)
        print(f"  Group {g}: median H={med:.3f}, dominant sheep={most_common} "
              f"(leads {dominant_counts[most_common]}/{len(grp_dominant_sheep[g])} assays)")


# ===========================================================================
# 17. AGGREGATE OCCUPANCY HEATMAPS BY ASSAY (first 5 minutes)
# ===========================================================================
print("Generating aggregate occupancy heatmaps...")

HEAT_BINS = 120
HEAT_SIGMA = 1.5
HEAT_CUTOFF = 5.0  # fixed 5-min window (covers median completion for assay 3+)
TARGET_ASSAYS_HEAT = [1, 3, 5, 7]

fig, axes = plt.subplots(1, len(TARGET_ASSAYS_HEAT),
                         figsize=(3.2 * len(TARGET_ASSAYS_HEAT), 3.4))
panel_labels_h = "abcd"

for idx, (ax, ta) in enumerate(zip(axes, TARGET_ASSAYS_HEAT)):
    assay_trials = [t for t in test_trials if t["assay"] == ta]
    all_gx, all_gy = [], []
    for trial in assay_trials:
        tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
        for trk in tracks.values():
            mask = trk["t"] <= HEAT_CUTOFF
            all_gx.append(trk["gx"][mask])
            all_gy.append(trk["gy"][mask])
    if all_gx:
        gx_cat = np.concatenate(all_gx)
        gy_cat = np.concatenate(all_gy)
        H, _, _ = np.histogram2d(gx_cat, gy_cat, bins=HEAT_BINS, range=[[0, 5], [0, 5]])
        H = gaussian_filter(H, sigma=HEAT_SIGMA)
        ax.imshow(np.log1p(H.T), origin="lower", extent=[0, 5, 0, 5],
                  cmap="Blues", aspect="equal", vmin=0)

    for lbl, (sx, sy) in site_positions.items():
        c = PALETTE["gold"] if lbl.startswith("A") else "#ccc"
        ax.add_patch(Circle((sx, sy), 0.15, fc=c, ec="k", lw=0.4, alpha=0.6, zorder=5))
        ax.text(sx, sy, lbl, ha="center", va="center", fontsize=4, zorder=6)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_title(f"{panel_labels_h[idx]}  Assay {ta}  (n={len(assay_trials)})",
                 fontsize=8, fontweight="bold")
    if idx == 0:
        ax.set_ylabel("Grid Y")
    ax.set_xlabel("Grid X")

fig.suptitle(f"Occupancy in first {HEAT_CUTOFF:.0f} min (log scale, all sheep pooled)",
             fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig(FIGDIR / "occupancy_by_assay.pdf")
fig.savefig(FIGDIR / "occupancy_by_assay.png")
plt.close(fig)
print("  -> occupancy_by_assay")


# ===========================================================================
# 18. FLOCKING DYNAMICS TIME SERIES (NND & SPREAD OVER TRIAL DURATION)
# ===========================================================================
print("Computing flocking time-series by assay...")
# Re-use flock computation but keep temporal curves
T_COMMON = np.arange(0, 25, 1 / 60)  # 1-second grid, 25 min
FLOCK_ASSAYS = sorted(flock_df["assay"].unique())
nnd_curves_by_assay = {a: [] for a in FLOCK_ASSAYS}
spread_curves_by_assay = {a: [] for a in FLOCK_ASSAYS}

for trial in test_trials:
    if trial["group_size"] < 2:
        continue
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if len(tracks) < 2:
        continue
    interp = _interp_to_1s(tracks)
    sids = sorted(interp.keys())
    n = len(sids)
    T = min(len(interp[s]["gx"]) for s in sids)
    T = min(T, len(T_COMMON))

    GX = np.column_stack([interp[s]["gx"][:T] for s in sids])
    GY = np.column_stack([interp[s]["gy"][:T] for s in sids])

    # NND per timestep
    nnd_ts = np.full(T, np.nan)
    for ti in range(0, T, 10):  # subsample every 10s for speed
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((GX[ti, i] - GX[ti, j]) ** 2 + (GY[ti, i] - GY[ti, j]) ** 2)
                dists.append(d)
        if dists:
            nnd_ts[ti] = min(dists)
    # Interpolate gaps
    valid = ~np.isnan(nnd_ts)
    if valid.sum() > 1:
        nnd_ts = np.interp(np.arange(T), np.where(valid)[0], nnd_ts[valid])

    # Spread per timestep
    cx = GX.mean(axis=1)
    cy = GY.mean(axis=1)
    spread_ts = np.mean(np.sqrt((GX - cx[:, None]) ** 2 + (GY - cy[:, None]) ** 2), axis=1)

    a = trial["assay"]
    # Pad/truncate to common length
    pad_len = len(T_COMMON)
    nnd_padded = np.full(pad_len, np.nan)
    spread_padded = np.full(pad_len, np.nan)
    nnd_padded[:T] = nnd_ts[:T] * 10  # metres
    spread_padded[:T] = spread_ts[:T] * 10
    nnd_curves_by_assay[a].append(nnd_padded)
    spread_curves_by_assay[a].append(spread_padded)

# Plot: NND and spread time series by assay group (exclude assay 0: unbaited)
ASSAY_GROUPS = [(1, "Assay 1-2"), (3, "Assay 3-4"), (5, "Assay 5-7")]
assay_group_map = {1: [1, 2], 3: [3, 4], 5: [5, 6, 7]}
colors_ag = [PALETTE["orange"], PALETTE["blue"], PALETTE["green"]]
t_min = T_COMMON  # in minutes

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

for (ag_key, ag_label), col in zip(ASSAY_GROUPS, colors_ag):
    # Combine curves from assays in this group
    curves_nnd = []
    curves_spr = []
    for a in assay_group_map[ag_key]:
        curves_nnd.extend(nnd_curves_by_assay.get(a, []))
        curves_spr.extend(spread_curves_by_assay.get(a, []))
    if not curves_nnd:
        continue
    mat_nnd = np.array(curves_nnd)
    mat_spr = np.array(curves_spr)
    mean_nnd = np.nanmean(mat_nnd, axis=0)
    sem_nnd = np.nanstd(mat_nnd, axis=0) / np.sqrt(np.sum(~np.isnan(mat_nnd), axis=0).clip(1))
    mean_spr = np.nanmean(mat_spr, axis=0)
    sem_spr = np.nanstd(mat_spr, axis=0) / np.sqrt(np.sum(~np.isnan(mat_spr), axis=0).clip(1))

    n_trials = len(curves_nnd)
    ax1.plot(t_min, mean_nnd, color=col, lw=1.2, label=f"{ag_label} (n={n_trials})")
    ax1.fill_between(t_min, mean_nnd - sem_nnd, mean_nnd + sem_nnd, color=col, alpha=0.15)
    ax2.plot(t_min, mean_spr, color=col, lw=1.2, label=f"{ag_label} (n={n_trials})")
    ax2.fill_between(t_min, mean_spr - sem_spr, mean_spr + sem_spr, color=col, alpha=0.15)

ax1.set_ylabel("Mean NND (m)")
ax1.set_title("a  Nearest-neighbour distance over trial duration")
ax1.set_xlim(0, 25)
ax1.legend(fontsize=6.5, ncol=2)
ax2.set_ylabel("Mean spread (m)")
ax2.set_xlabel("Time (min)")
ax2.set_xlim(0, 25)
ax2.set_title("b  Group spread over trial duration")
ax2.legend(fontsize=6.5, ncol=2)

fig.tight_layout()
fig.savefig(FIGDIR / "flocking_timeseries.pdf")
fig.savefig(FIGDIR / "flocking_timeseries.png")
plt.close(fig)
print("  -> flocking_timeseries")


# ===========================================================================
# 19. SUCCESS vs FAILURE EXAMPLE TRAJECTORIES
# ===========================================================================
print("Generating success/failure examples...")
success_trial = failure_trial = None
for trial in test_trials:
    tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
    if not tracks or len(tracks) < 2:
        continue
    visits = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)
    baited_disc = {s: min(v[1] for v in vl) for s, vl in visits.items()
                   if vl and s in BAITED_SITES}
    if len(baited_disc) == 3 and success_trial is None and trial["assay"] in [3, 4, 5]:
        success_trial = (trial, tracks, visits, baited_disc)
    elif len(baited_disc) <= 1 and failure_trial is None and trial["assay"] in [0, 1]:
        failure_trial = (trial, tracks, visits, baited_disc)
    if success_trial and failure_trial:
        break

# Fallback
if success_trial is None:
    for trial in test_trials:
        tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
        if not tracks:
            continue
        visits = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)
        bd = {s: min(v[1] for v in vl) for s, vl in visits.items() if vl and s in BAITED_SITES}
        if len(bd) == 3:
            success_trial = (trial, tracks, visits, bd)
            break
if failure_trial is None:
    for trial in test_trials:
        if trial["assay"] != 0:
            continue
        tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
        if not tracks:
            continue
        visits = detect_site_visits(tracks, trial["field"], RADIUS, MIN_DWELL_S)
        bd = {s: min(v[1] for v in vl) for s, vl in visits.items() if vl and s in BAITED_SITES}
        failure_trial = (trial, tracks, visits, bd)
        break

if success_trial and failure_trial:
    tab10 = plt.colormaps["tab10"]
    fig, (ax_s, ax_f) = plt.subplots(1, 2, figsize=(10, 4.8))

    for ax, (trial, tracks, visits, bd), label in [
        (ax_s, success_trial, "a  Success"),
        (ax_f, failure_trial, "b  Failure"),
    ]:
        ax.set_facecolor("#f0f0f0")
        # Cut tracks at completion time for success panel
        ct = max(bd.values()) if len(bd) == 3 else 25.0
        for i, (sid, trk) in enumerate(sorted(tracks.items())):
            mask = trk["t"] <= ct
            ax.plot(trk["gx"][mask], trk["gy"][mask], lw=0.6, alpha=0.5,
                    color=tab10(i % 10), label=sid, rasterized=True)

        # Reward sites
        for lbl, (sx, sy) in site_positions.items():
            c = PALETTE["gold"] if lbl.startswith("A") else "#ddd"
            ax.add_patch(Circle((sx, sy), 0.15, fc=c, ec="k", lw=0.5, alpha=0.7, zorder=5))
            ax.text(sx, sy, lbl, ha="center", va="center", fontsize=5, zorder=6)

        # 2m detection circles around baited sites
        for s in BAITED_SITES:
            sx, sy = site_positions[s]
            found = s in bd
            ec = PALETTE["green"] if found else PALETTE["red"]
            ax.add_patch(Circle((sx, sy), RADIUS, fill=False, ec=ec, ls="--", lw=1.2, zorder=5))
            if found:
                ax.plot(sx, sy, "*", ms=12, color=PALETTE["green"], zorder=7,
                        markeredgecolor="k", markeredgewidth=0.3)

        n_found = len(bd)
        ax.set_xlim(-0.1, 5.1)
        ax.set_ylim(-0.1, 5.1)
        ax.set_aspect("equal")
        ax.set_xlabel("Grid X (10 m/unit)")
        ax.set_ylabel("Grid Y")
        t_label = f"t<{ct:.0f} min" if len(bd) == 3 else "full trial"
        ax.set_title(f"{label}: assay {trial['assay']}, {n_found}/3 baited ({t_label})",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=5.5, loc="lower right", framealpha=0.7, markerscale=3)

    fig.tight_layout()
    fig.savefig(FIGDIR / "success_failure_example.pdf")
    fig.savefig(FIGDIR / "success_failure_example.png")
    plt.close(fig)
    print("  -> success_failure_example")


# ===========================================================================
# 20. LEADERSHIP METRIC SCHEMATIC
# ===========================================================================
print("Generating leadership schematic...")
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.set_xlim(1.0, 4.0)
ax.set_ylim(1.0, 4.0)
ax.set_aspect("equal")
ax.set_facecolor("#fafafa")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Synthetic sheep positions
sheep = {"A": (1.8, 3.0), "B": (2.4, 2.5), "C": (3.2, 3.3), "D": (2.1, 2.0)}
colors_s = {"A": tab10(0), "B": tab10(1), "C": tab10(2), "D": tab10(3)}

cx = np.mean([p[0] for p in sheep.values()])
cy = np.mean([p[1] for p in sheep.values()])

# Velocity direction
v_angle = np.radians(55)
vx, vy = np.cos(v_angle), np.sin(v_angle)
arrow_len = 0.8

# Draw velocity arrow
ax.annotate("", xy=(cx + arrow_len * vx, cy + arrow_len * vy), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", lw=2.5, color="k"))
ax.text(cx + arrow_len * vx + 0.08, cy + arrow_len * vy + 0.08,
        "Group velocity", fontsize=8, fontweight="bold", color="k")

# Centroid
ax.plot(cx, cy, "+", ms=15, mew=2.5, color="k", zorder=10)
ax.text(cx - 0.02, cy - 0.15, "Centroid", ha="center", fontsize=7, color="k")

# For each sheep: position, projection, line
projections = {}
for sid, (sx, sy) in sheep.items():
    dx, dy = sx - cx, sy - cy
    proj = dx * vx + dy * vy
    projections[sid] = proj

    # Projection point on velocity line
    px, py = cx + proj * vx, cy + proj * vy

    # Dashed projection line
    ax.plot([sx, px], [sy, py], "--", color=colors_s[sid], lw=1, alpha=0.6)

    # Sheep marker
    is_leader = (sid == max(projections, key=projections.get))
    ms = 14 if is_leader else 10
    ax.plot(sx, sy, "o", ms=ms, color=colors_s[sid], zorder=8,
            markeredgecolor="k", markeredgewidth=0.8)
    ax.text(sx + 0.12, sy + 0.1, f"Sheep {sid}\nproj={proj:.2f}",
            fontsize=6.5, color=colors_s[sid], fontweight="bold")

# Highlight leader
leader_sid = max(projections, key=projections.get)
lx, ly = sheep[leader_sid]
ax.plot(lx, ly, "o", ms=18, color="none", markeredgecolor=PALETTE["gold"],
        markeredgewidth=2.5, zorder=9)
ax.text(lx + 0.15, ly - 0.18, "LEADER", fontsize=8, color=PALETTE["gold"],
        fontweight="bold")

# Projection axis (faint line through centroid along velocity)
ax.plot([cx - 1.2 * vx, cx + 1.5 * vx], [cy - 1.2 * vy, cy + 1.5 * vy],
        "-", color="#bbb", lw=1, zorder=0)
ax.text(cx + 1.5 * vx, cy + 1.5 * vy - 0.15, "+", fontsize=10, color="#999", ha="center")
ax.text(cx - 1.2 * vx, cy - 1.2 * vy - 0.15, "−", fontsize=10, color="#999", ha="center")

ax.set_title("Frontal-position leadership metric", fontsize=10, fontweight="bold", pad=12)

fig.tight_layout()
fig.savefig(FIGDIR / "leadership_schematic.pdf")
fig.savefig(FIGDIR / "leadership_schematic.png")
plt.close(fig)
print("  -> leadership_schematic")


# ===========================================================================
# 21. OCCUPANCY HEATMAP — MOVING PERIODS ONLY
# ===========================================================================
print("Generating moving-only occupancy heatmaps...")
TARGET_ASSAYS_MOV = [1, 3, 5, 7]

fig, axes = plt.subplots(1, len(TARGET_ASSAYS_MOV),
                         figsize=(3.2 * len(TARGET_ASSAYS_MOV), 3.4))
panel_labels_m = "abcd"

for idx, (ax, ta) in enumerate(zip(axes, TARGET_ASSAYS_MOV)):
    assay_trials_m = [t for t in test_trials if t["assay"] == ta]
    all_gx, all_gy = [], []
    for trial in assay_trials_m:
        tracks = load_trial_tracks(trial, tracks_cache=tracks_cache, apply_orient=True)
        if not tracks:
            continue
        interp = _prepare_tracks(tracks)
        for sid, trk in interp.items():
            mask_time = trk["t"] <= HEAT_CUTOFF
            mask_move = _moving_mask(trk["gx"], trk["gy"], trk["t"], MOVE_SPEED_THRESH)
            combined = mask_time & mask_move
            all_gx.append(trk["gx"][combined])
            all_gy.append(trk["gy"][combined])
    if all_gx:
        gx_cat = np.concatenate(all_gx)
        gy_cat = np.concatenate(all_gy)
        H, _, _ = np.histogram2d(gx_cat, gy_cat, bins=HEAT_BINS, range=[[0, 5], [0, 5]])
        H = gaussian_filter(H, sigma=HEAT_SIGMA)
        ax.imshow(np.log1p(H.T), origin="lower", extent=[0, 5, 0, 5],
                  cmap="Oranges", aspect="equal", vmin=0)

    for lbl, (sx, sy) in site_positions.items():
        c = PALETTE["gold"] if lbl.startswith("A") else "#ccc"
        ax.add_patch(Circle((sx, sy), 0.15, fc=c, ec="k", lw=0.4, alpha=0.6, zorder=5))
        ax.text(sx, sy, lbl, ha="center", va="center", fontsize=4, zorder=6)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_title(f"{panel_labels_m[idx]}  Assay {ta}  (n={len(assay_trials_m)})",
                 fontsize=8, fontweight="bold")
    if idx == 0:
        ax.set_ylabel("Grid Y")
    ax.set_xlabel("Grid X")

fig.suptitle(f"Search occupancy: moving only (>{MOVE_SPEED_THRESH} m/min, first {HEAT_CUTOFF:.0f} min)",
             fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig(FIGDIR / "occupancy_moving_only.pdf")
fig.savefig(FIGDIR / "occupancy_moving_only.png")
plt.close(fig)
print("  -> occupancy_moving_only")


print("\nAll figures generated!")
