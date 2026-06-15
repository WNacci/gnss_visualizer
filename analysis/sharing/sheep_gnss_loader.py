"""Self-contained loader for the Phase 2 sheep GNSS dataset.

Public functions
----------------
load_trials(csv_path) -> list[dict]
    Parse trial_metadata.csv into a list of trial dicts.

load_trial_tracks(trial, gnss_root, reward_sites_csv, apply_orient=False)
    Return per-sheep tracks in arena grid coordinates for one trial.

Dependencies: numpy, pandas. (Standard library: re, pathlib.)

Coordinate system
-----------------
Arena grid is 0-5 in each axis; 1 unit = 10 m.
Origin is bottom-left; y increases north, x increases east.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd


_R = 6_371_000.0  # Earth radius (m) for the local equirectangular projection.

# Test-config orientation transforms (rotation_deg, reflection).
# After apply_orient=True, each test config's baited triplet maps to the
# canonical positions {A1, A2, A3}. Control configs have no meaningful
# canonical orientation; CTRL_FAR is included for completeness, CTRL_BARN
# falls through to (0, "none").
CONFIG_TRANSFORMS = {
    "A":        (0,   "none"),
    "B":        (90,  "none"),
    "C":        (0,   "mirror y"),
    "D":        (270, "mirror x"),
    "CTRL_FAR": (0,   "mirror y"),
}

# Site label -> arena grid (x, y) in 10-m units. 12 reward sites in a 4x3 layout.
SITE_GRID = {
    "B1": (2, 4), "A3": (3, 4),
    "C1": (1, 3), "D2": (2, 3), "C2": (3, 3), "D3": (4, 3),
    "A1": (1, 2), "B2": (2, 2), "A2": (3, 2), "B3": (4, 2),
    "D1": (2, 1), "C3": (3, 1),
}


# ---------------------------------------------------------------------------
# Raw GNSS log parsing
# ---------------------------------------------------------------------------

def load_gnss_dir(device_dir):
    """Parse all LOGS*.TXT files in one GNSS device directory.

    Log lines are colon-delimited:
      T{dev}:{sats}:{dop}:{lat}:{lon}:{unix_time}:{ax}:{ay}:{az}:

    Returns (lats, lons, times) as float64 arrays sorted by file name (which
    corresponds to chronological order within a session).
    """
    device_dir = Path(device_dir)
    lats, lons, times = [], [], []
    for f in sorted(device_dir.iterdir(), key=lambda p: p.name):
        if re.match(r"LOGS\d+\.TXT", f.name, re.IGNORECASE) and f.stat().st_size > 0:
            for line in open(f, errors="ignore"):
                parts = line.split(":")
                if len(parts) >= 6:
                    try:
                        lats.append(float(parts[3]))
                        lons.append(float(parts[4]))
                        times.append(float(parts[5]))
                    except ValueError:
                        pass
    return np.array(lats), np.array(lons), np.array(times)


def load_gnss_date(gnss_root, date_str):
    """Load all GNSS device data for one session date.

    `date_str` is YYYY-MM-DD. Looks under `gnss_root/{day}-02-26/GNSS*/`.

    Returns dict: device_num (int) -> (lats, lons, times).
    """
    day = int(date_str.split("-")[2])
    gnss_dir = Path(gnss_root) / f"{day}-02-26"
    if not gnss_dir.exists():
        return {}
    out = {}
    for device_dir in sorted(gnss_dir.iterdir()):
        if not (device_dir.is_dir() and device_dir.name.startswith("GNSS")):
            continue
        try:
            dev_num = int(device_dir.name.replace("GNSS-", "").replace("GNSS_", ""))
        except ValueError:
            continue
        lats, lons, times = load_gnss_dir(device_dir)
        if len(lats) > 0:
            out[dev_num] = (lats, lons, times)
    return out


# ---------------------------------------------------------------------------
# GPS -> arena grid projection
# ---------------------------------------------------------------------------

def _latlon_to_meters(lats, lons, lat0, lon0):
    lat0_r = np.radians(lat0)
    x = (lons - lon0) * np.cos(lat0_r) * (np.pi / 180) * _R
    y = (lats - lat0) * (np.pi / 180) * _R
    return x, y


def build_arena_transforms(reward_sites_csv):
    """Fit per-field affine transforms GPS -> arena grid.

    Each field uses all 16 control points (12 reward sites + 4 corners E1-E4)
    for an overdetermined least-squares fit. Reward site positions come from
    a dedicated calibration session (2026-01-26) with a GNSS device placed
    at each known site.

    Returns dict: field ("A" or "B") -> {"lat0", "lon0", "M"} where M is the
    3x2 affine matrix used by `latlon_to_grid`.
    """
    df = pd.read_csv(reward_sites_csv)
    transforms = {}
    for field in ("A", "B"):
        fdf = df[df["field"] == field]
        if len(fdf) == 0:
            continue
        lat0 = fdf["latitude"].mean()
        lon0 = fdf["longitude"].mean()
        xm, ym = _latlon_to_meters(
            fdf["latitude"].values, fdf["longitude"].values, lat0, lon0
        )
        A_mat = np.column_stack([xm, ym, np.ones(len(xm))])
        dst = fdf[["grid_x", "grid_y"]].values.astype(float)
        M, _, _, _ = np.linalg.lstsq(A_mat, dst, rcond=None)
        transforms[field] = {"lat0": lat0, "lon0": lon0, "M": M}
    return transforms


def latlon_to_grid(lats, lons, transform):
    """Project lat/lon arrays to arena grid coords (typically 0-5 range)."""
    lat0, lon0, M = transform["lat0"], transform["lon0"], transform["M"]
    x, y = _latlon_to_meters(lats, lons, lat0, lon0)
    pts = np.column_stack([x, y, np.ones(len(x))])
    res = pts @ M
    return res[:, 0], res[:, 1]


def apply_orientation(gx, gy, rotation_deg, reflection):
    """Rotate / reflect grid coords around arena centre (2.5, 2.5).

    Reflection is applied before rotation. Use only with test configs (A-D)
    when comparing across configurations in a shared canonical frame.
    """
    cx, cy = 2.5, 2.5
    gx = np.asarray(gx, dtype=float) - cx
    gy = np.asarray(gy, dtype=float) - cy
    if reflection == "mirror x":
        gx = -gx
    elif reflection == "mirror y":
        gy = -gy
    if rotation_deg != 0:
        theta = np.radians(float(rotation_deg))
        c, s = np.cos(theta), np.sin(theta)
        gx, gy = c * gx - s * gy, s * gx + c * gy
    return gx + cx, gy + cy


def _best_field(lats, lons, csv_field, arena_transforms):
    """Choose the field transform that keeps the most GPS points in-arena.

    The `field` column in the trial CSV is occasionally wrong; this picks the
    field whose affine transform keeps a larger fraction of points inside
    [0, 5]^2 grid units.
    """
    def _in_arena_count(f):
        gx, gy = latlon_to_grid(lats, lons, arena_transforms[f])
        return int(np.sum((gx >= 0) & (gx <= 5) & (gy >= 0) & (gy <= 5)))

    if csv_field not in arena_transforms:
        avail = list(arena_transforms.keys())
        return avail[0] if avail else csv_field

    best, best_n = csv_field, _in_arena_count(csv_field)
    for f in arena_transforms:
        if f == csv_field:
            continue
        n = _in_arena_count(f)
        if n > best_n:
            best_n, best = n, f
    return best


# ---------------------------------------------------------------------------
# Trial metadata
# ---------------------------------------------------------------------------

def load_trials(csv_path):
    """Parse trial_metadata.csv into a list of trial dicts.

    Each dict contains:
      name, date, field, config, start_time (HH:MM:SS Paris local),
      duration_min, devices (list[int]), device_to_sheep (dict int -> str),
      assay, notes, group_num, group_size.

    Rows without a `start_time` are skipped (data-cleaning artifacts).
    Sheep IDs are preserved as strings to avoid leading-zero loss.
    """
    df = pd.read_csv(csv_path, dtype={"Sheep ID": str})
    trials = []
    with_start = df[df["start_time"].notna()].copy()
    for (date, start_time, gnum, gsize), group in with_start.groupby(
        ["date", "start_time", "Group #", "Group Size"], dropna=False
    ):
        devices = sorted(set(
            [int(v) for v in group["GNSS_SN1"].dropna()]
            + [int(v) for v in group["GNSS_SN2"].dropna()]
        ))
        d2s = {}
        for _, row in group.iterrows():
            sid = row["Sheep ID"] if pd.notna(row["Sheep ID"]) else "Unknown"
            if pd.notna(row["GNSS_SN1"]):
                d2s[int(row["GNSS_SN1"])] = sid
            if pd.notna(row["GNSS_SN2"]):
                d2s[int(row["GNSS_SN2"])] = sid

        field = str(group["field"].iloc[0]) if pd.notna(group["field"].iloc[0]) else "Unknown"
        config = str(group["configuration"].iloc[0]) if pd.notna(group["configuration"].iloc[0]) else "Unknown"
        gnum_i = int(gnum) if pd.notna(gnum) else 0
        gsize_i = int(gsize) if pd.notna(gsize) else 0
        av = group["assay"].iloc[0]
        assay = None
        if pd.notna(av):
            try:
                assay = int(float(av))
            except (ValueError, TypeError):
                assay = str(av)
        notes_list = group["note"].dropna().unique()
        notes = f"Group {gnum_i}, Size {gsize_i}"
        if len(notes_list) > 0:
            notes += " - " + "; ".join(notes_list)

        trials.append({
            "name": f"{date} - Grp{gnum_i:02d} {str(start_time)[:5]} Field {field}, {config}, {gsize_i} sheep",
            "date": str(date),
            "field": field,
            "config": config,
            "start_time": str(start_time),
            "duration_min": 35,
            "devices": devices,
            "device_to_sheep": d2s,
            "assay": assay,
            "notes": notes,
            "group_num": gnum_i,
            "group_size": gsize_i,
        })
    return trials


# ---------------------------------------------------------------------------
# Per-trial tracks
# ---------------------------------------------------------------------------

def load_trial_tracks(trial, gnss_root, reward_sites_csv, apply_orient=False):
    """Load and project GPS data for one trial.

    Parameters
    ----------
    trial : dict
        One trial dict from `load_trials()`.
    gnss_root : str | Path
        Path to the `gnss/` directory in the package.
    reward_sites_csv : str | Path
        Path to `fitted_reward_sites.csv`.
    apply_orient : bool
        If True, rotates/mirrors test-config tracks so the baited triplet
        always lands at the canonical positions {A1, A2, A3}. Meaningless
        for CTRL trials and skipped silently for them.

    Returns
    -------
    dict : sheep_id (str) -> {"gx", "gy", "t"}
        gx, gy are arena grid coordinates (1 unit = 10 m).
        t is minutes from trial start (Paris local time).
        Sheep that carried two GNSS devices are interpolated to a common
        10 Hz grid and averaged.
    """
    raw = load_gnss_date(gnss_root, trial["date"])
    if not raw:
        return {}

    arena_transforms = build_arena_transforms(reward_sites_csv)
    if not arena_transforms:
        return {}

    rot, ref = (0, "none")
    if apply_orient:
        rot, ref = CONFIG_TRANSFORMS.get(trial["config"], (0, "none"))

    # Trial window in unix seconds (Paris local -> UTC).
    dt_str = f"{trial['date']} {trial['start_time']}"
    start_unix = (
        pd.to_datetime(dt_str)
        .tz_localize("Europe/Paris")
        .tz_convert("UTC")
        .timestamp()
    )
    end_unix = start_unix + trial["duration_min"] * 60

    # Auto-correct the field label using the first device with enough data
    # in the trial window. The CSV `field` is occasionally wrong.
    csv_field = trial["field"]
    field = csv_field
    for dev_num in trial["devices"]:
        if dev_num not in raw:
            continue
        lats_d, lons_d, times_d = raw[dev_num]
        mask_d = (times_d >= start_unix) & (times_d <= end_unix)
        if mask_d.sum() < 10:
            continue
        field = _best_field(lats_d[mask_d], lons_d[mask_d], csv_field, arena_transforms)
        break

    if field not in arena_transforms:
        return {}

    transform = arena_transforms[field]
    d2s = trial.get("device_to_sheep", {})

    # Collect per-device tracks, grouped by sheep_id (some sheep carried two
    # devices and are averaged below).
    sheep_arrays = {}
    for dev_num in trial["devices"]:
        if dev_num not in raw:
            continue
        lats, lons, times = raw[dev_num]
        mask = (times >= start_unix) & (times <= end_unix)
        if not mask.any():
            continue
        lf, lnf, tf = lats[mask], lons[mask], times[mask]
        t_rel = (tf - start_unix) / 60.0
        gx, gy = latlon_to_grid(lf, lnf, transform)
        if apply_orient:
            gx, gy = apply_orientation(gx, gy, rot, ref)
        sheep_id = d2s.get(dev_num, f"Dev{dev_num}")
        sheep_arrays.setdefault(sheep_id, []).append((gx, gy, t_rel))

    result = {}
    for sheep_id, dev_tracks in sheep_arrays.items():
        if len(dev_tracks) == 1:
            gx_f, gy_f, t_f = dev_tracks[0]
        else:
            # Average two devices on a common 10 Hz grid.
            t_min = min(t.min() for _, _, t in dev_tracks)
            t_max = max(t.max() for _, _, t in dev_tracks)
            t_f = np.arange(t_min, t_max + 1 / 600, 1 / 600)
            stacks_gx, stacks_gy = [], []
            for gx_d, gy_d, t_d in dev_tracks:
                order = np.argsort(t_d)
                stacks_gx.append(np.interp(t_f, t_d[order], gx_d[order]))
                stacks_gy.append(np.interp(t_f, t_d[order], gy_d[order]))
            gx_f = np.mean(stacks_gx, axis=0)
            gy_f = np.mean(stacks_gy, axis=0)
        result[sheep_id] = {"gx": gx_f, "gy": gy_f, "t": t_f}

    return result
