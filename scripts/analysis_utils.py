"""Shared utilities for sheep GPS analysis.

Provides data loading, coordinate projection, and trial metadata helpers
used by the analysis marimo scripts.
"""
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

_R = 6_371_000.0  # Earth radius (m)

DATA_DIR = Path(__file__).parent.parent / "data"

CONFIG_TRANSFORMS = {
    "A":        (0,   "none"),
    "B":        (90,  "none"),
    "C":        (0,   "mirror y"),
    "D":        (270, "mirror x"),
    "CTRL_FAR": (0,   "mirror y"),
}

# Reward site grid positions (grid_x, grid_y) for each label.
# Origin is bottom-left; y increases north, x increases east.
# Units: 10 m per grid unit, arena spans ~0–5.
SITE_GRID = {
    'B1': (2, 4), 'A3': (3, 4),
    'C1': (1, 3), 'D2': (2, 3), 'C2': (3, 3), 'D3': (4, 3),
    'A1': (1, 2), 'B2': (2, 2), 'A2': (3, 2), 'B3': (4, 2),
    'D1': (2, 1), 'C3': (3, 1),
}
SITE_LABELS = sorted(SITE_GRID.keys())


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_gnss_dir(device_dir: Path):
    """Load GPS data from a single GNSS device directory.

    Returns (lats, lons, times) as float64 arrays.
    """
    lats, lons, times = [], [], []
    for f in sorted(device_dir.iterdir(), key=lambda x: x.name):
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


def build_gps_cache(trials: list | None = None, max_workers: int = 8) -> dict:
    """Pre-load all GNSS data for every trial date in parallel.

    Returns dict: date_str → {device_num (int) → (lats, lons, times)}.
    Call once at notebook startup; pass the result as ``gnss_cache`` to
    :func:`load_trial_tracks`.
    """
    if trials is None:
        trials = build_trials()
    dates = sorted({t["date"] for t in trials})

    def _load(date):
        return date, load_gnss_date(date)

    cache = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(dates))) as pool:
        for date, data in pool.map(_load, dates):
            cache[date] = data

    total_pts = sum(
        len(lats) for day_data in cache.values() for (lats, _, _) in day_data.values()
    )
    print(f"GPS cache: {len(cache)} dates, {total_pts:,} total points")
    return cache


def load_gnss_date(date_str: str):
    """Load all GNSS device data for a session date (YYYY-MM-DD).

    Returns dict: device_num (int) → (lats, lons, times).
    """
    day = int(date_str.split("-")[2])
    gnss_dir = DATA_DIR / "gnss" / f"{day}-02-26"
    if not gnss_dir.exists():
        return {}
    result = {}
    for device_dir in sorted(gnss_dir.iterdir()):
        if not (device_dir.is_dir() and device_dir.name.startswith("GNSS")):
            continue
        try:
            dev_num = int(device_dir.name.replace("GNSS-", "").replace("GNSS_", ""))
        except ValueError:
            continue
        lats, lons, times = load_gnss_dir(device_dir)
        if len(lats) > 0:
            result[dev_num] = (lats, lons, times)
    return result


# ---------------------------------------------------------------------------
# Arena coordinate transforms
# ---------------------------------------------------------------------------

def _latlon_to_meters(lats, lons, lat0, lon0):
    lat0_r = np.radians(lat0)
    x = (lons - lon0) * np.cos(lat0_r) * (np.pi / 180) * _R
    y = (lats - lat0) * (np.pi / 180) * _R
    return x, y


def build_arena_transforms(reward_sites_df=None):
    """Fit affine transforms from GPS → arena grid coords for each field.

    Returns dict: field → {'lat0', 'lon0', 'M'}.
    """
    if reward_sites_df is None:
        reward_sites_df = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
    transforms = {}
    for field in ["A", "B"]:
        fdf = reward_sites_df[reward_sites_df["field"] == field]
        corners = fdf[fdf["label"].str.startswith("E")]
        lat0 = corners["latitude"].mean()
        lon0 = corners["longitude"].mean()
        xm, ym = _latlon_to_meters(
            corners["latitude"].values, corners["longitude"].values, lat0, lon0
        )
        A_mat = np.column_stack([xm, ym, np.ones(len(xm))])
        dst = corners[["grid_x", "grid_y"]].values.astype(float)
        M, _, _, _ = np.linalg.lstsq(A_mat, dst, rcond=None)
        transforms[field] = {"lat0": lat0, "lon0": lon0, "M": M}
    return transforms


def latlon_to_grid(lats, lons, transform):
    """Project lat/lon arrays to arena grid coordinates (≈0–5 range)."""
    lat0, lon0, M = transform["lat0"], transform["lon0"], transform["M"]
    x, y = _latlon_to_meters(lats, lons, lat0, lon0)
    pts = np.column_stack([x, y, np.ones(len(x))])
    res = pts @ M
    return res[:, 0], res[:, 1]


def apply_orientation(gx, gy, rotation_deg, reflection):
    """Rotate and/or reflect grid coords around arena centre (2.5, 2.5).

    reflection: 'mirror x' | 'mirror y' | 'none'
    Reflection is applied before rotation.
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


# ---------------------------------------------------------------------------
# Trial metadata
# ---------------------------------------------------------------------------

def build_trials(df=None):
    """Build list of trial dicts from the experimental CSV.

    Each dict contains:
      name, date, field, config, start_time (str HH:MM:SS),
      duration_min, devices (list[int]), device_to_sheep (dict int→str),
      assay, notes, group_num, group_size.
    """
    if df is None:
        df = pd.read_csv(
            DATA_DIR / "experimental" / "Sheep_Trial_Data.csv",
            dtype={"Sheep ID": str},
        )
    trials = []
    with_start = df[df["start_time"].notna()].copy()
    for (date, start_time, group_num, group_size), group in with_start.groupby(
        ["date", "start_time", "Group #", "Group Size"], dropna=False
    ):
        devices = sorted(
            set(
                [int(v) for v in group["GNSS_SN1"].dropna()]
                + [int(v) for v in group["GNSS_SN2"].dropna()]
            )
        )
        d2s = {}
        for _, row in group.iterrows():
            sid = row["Sheep ID"] if pd.notna(row["Sheep ID"]) else "Unknown"
            if pd.notna(row["GNSS_SN1"]):
                d2s[int(row["GNSS_SN1"])] = sid
            if pd.notna(row["GNSS_SN2"]):
                d2s[int(row["GNSS_SN2"])] = sid

        field = str(group["field"].iloc[0]) if pd.notna(group["field"].iloc[0]) else "Unknown"
        config = (
            str(group["configuration"].iloc[0])
            if pd.notna(group["configuration"].iloc[0])
            else "Unknown"
        )
        gnum = int(group_num) if pd.notna(group_num) else 0
        gsize = int(group_size) if pd.notna(group_size) else 0
        av = group["assay"].iloc[0]
        assay = None
        if pd.notna(av):
            try:
                assay = int(float(av))
            except (ValueError, TypeError):
                assay = str(av)
        notes_list = group["note"].dropna().unique()
        notes = f"Group {gnum}, Size {gsize}"
        if len(notes_list) > 0:
            notes += f" - {'; '.join(notes_list)}"

        trials.append(
            {
                "name": f"{date} - Field {field}, {config}, {gsize} sheep",
                "date": str(date),
                "field": field,
                "config": config,
                "start_time": str(start_time),
                "duration_min": 35,
                "devices": devices,
                "device_to_sheep": d2s,
                "assay": assay,
                "notes": notes,
                "group_num": gnum,
                "group_size": gsize,
            }
        )
    return trials


# ---------------------------------------------------------------------------
# Loading per-trial GPS data
# ---------------------------------------------------------------------------

def _best_field(lats, lons, csv_field, arena_transforms):
    """Return the field transform that keeps the most GPS points inside the arena.

    Falls back to csv_field if no better transform is found.  Mirrors the
    auto-correction logic in occupancy_heatmap.py.
    """
    def _in_arena_count(f):
        gx, gy = latlon_to_grid(lats, lons, arena_transforms[f])
        return int(np.sum((gx >= 0) & (gx <= 5) & (gy >= 0) & (gy <= 5)))

    if csv_field not in arena_transforms:
        # Fall back to whichever field is available
        available = list(arena_transforms.keys())
        return available[0] if available else csv_field

    best, best_n = csv_field, _in_arena_count(csv_field)
    for f in arena_transforms:
        if f == csv_field:
            continue
        n = _in_arena_count(f)
        if n > best_n:
            best_n, best = n, f
    return best


def load_trial_tracks(
    trial: dict,
    gnss_cache: dict | None = None,
    apply_orient: bool = False,
    arena_transforms: dict | None = None,
):
    """Load and project GPS data for a single trial.

    Parameters
    ----------
    trial : dict
        A trial dict as returned by build_trials().
    gnss_cache : dict, optional
        Pre-loaded GNSS data: date_str → {dev_num → (lats, lons, times)}.
        If None, data is loaded from disk.
    apply_orient : bool
        Whether to apply per-configuration orientation transform.
    arena_transforms : dict, optional
        Pre-built arena transforms. Built on demand if None.

    Returns
    -------
    dict : sheep_id → {'gx': array, 'gy': array, 't': array}
        where t is minutes from trial start.
    """
    date = trial["date"]

    if gnss_cache is not None:
        raw = gnss_cache.get(date, {})
    else:
        raw = load_gnss_date(date)

    if not raw:
        return {}

    if arena_transforms is None:
        arena_transforms = build_arena_transforms()

    csv_field = trial["field"]
    if csv_field not in arena_transforms and not arena_transforms:
        return {}

    rot, ref = (0, "none")
    if apply_orient:
        rot, ref = CONFIG_TRANSFORMS.get(trial["config"], (0, "none"))

    # Convert start time (Paris local → UTC → unix)
    dt_str = f"{date} {trial['start_time']}"
    start_unix = (
        pd.to_datetime(dt_str)
        .tz_localize("Europe/Paris")
        .tz_convert("UTC")
        .timestamp()
    )
    end_unix = start_unix + trial["duration_min"] * 60

    # Auto-detect best field using first device with sufficient time-filtered data
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
    # Group arrays by sheep_id (sheep may carry 2 devices → average)
    sheep_arrays: dict[str, list[tuple]] = {}

    for dev_num in trial["devices"]:
        if dev_num not in raw:
            continue
        lats, lons, times = raw[dev_num]
        mask = (times >= start_unix) & (times <= end_unix)
        lf, lnf, tf = lats[mask], lons[mask], times[mask]
        if len(lf) == 0:
            continue
        t_rel = (tf - start_unix) / 60.0
        gx, gy = latlon_to_grid(lf, lnf, transform)
        if apply_orient:
            gx, gy = apply_orientation(gx, gy, rot, ref)

        sheep_id = d2s.get(dev_num, f"Dev{dev_num}")
        sheep_arrays.setdefault(sheep_id, []).append((gx, gy, t_rel))

    result = {}
    for sheep_id, tracks in sheep_arrays.items():
        if len(tracks) == 1:
            gx_f, gy_f, t_f = tracks[0]
        else:
            # Average two devices on a common time grid
            t_min = min(t.min() for _, _, t in tracks)
            t_max = max(t.max() for _, _, t in tracks)
            t_grid = np.arange(t_min, t_max + 1 / 600, 1 / 600)
            stacks_gx, stacks_gy = [], []
            for gx_d, gy_d, t_d in tracks:
                order = np.argsort(t_d)
                stacks_gx.append(np.interp(t_grid, t_d[order], gx_d[order]))
                stacks_gy.append(np.interp(t_grid, t_d[order], gy_d[order]))
            gx_f = np.mean(stacks_gx, axis=0)
            gy_f = np.mean(stacks_gy, axis=0)
            t_f = t_grid

        result[sheep_id] = {"gx": gx_f, "gy": gy_f, "t": t_f}

    return result


# ---------------------------------------------------------------------------
# Site visit detection
# ---------------------------------------------------------------------------

def detect_site_visits(
    sheep_tracks: dict,
    field: str = "A",
    radius: float = 0.5,
    min_dwell_s: float = 5.0,
    reward_sites_df=None,
):
    """Detect first and all visits to each reward site.

    Both fields use identical grid positions, so the field parameter only
    selects which row to load the canonical site list from (defaults to 'A').
    Passing either 'A' or 'B' gives the same detection result.

    Parameters
    ----------
    sheep_tracks : dict
        Output of load_trial_tracks().
    field : str
        'A' or 'B' (both have identical grid positions; for filtering only).
    radius : float
        Detection radius in grid units (1 unit ≈ 10 m).
    min_dwell_s : float
        Minimum continuous dwell time (seconds) to count as a visit.
    reward_sites_df : DataFrame, optional
        Reward site positions. Loaded if None.

    Returns
    -------
    dict : site_label → list of (sheep_id, first_entry_min, last_exit_min)
    """
    if reward_sites_df is None:
        reward_sites_df = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")

    min_dwell_min = min_dwell_s / 60.0
    # Use 'A' as canonical field - both A and B have identical grid_x/grid_y
    _canon_field = field if field in reward_sites_df["field"].values else "A"
    field_sites = reward_sites_df[
        (reward_sites_df["field"] == _canon_field)
        & (~reward_sites_df["label"].str.startswith("E"))
    ]

    visits: dict[str, list] = {row["label"]: [] for _, row in field_sites.iterrows()}

    for sheep_id, track in sheep_tracks.items():
        gx, gy, t = track["gx"], track["gy"], track["t"]
        for _, row in field_sites.iterrows():
            sx, sy = row["grid_x"], row["grid_y"]
            dist = np.sqrt((gx - sx) ** 2 + (gy - sy) ** 2)
            inside = dist <= radius

            # Find contiguous runs of inside=True
            padded = np.concatenate([[False], inside, [False]])
            diffs = np.diff(padded.astype(int))
            enters = np.where(diffs == 1)[0]
            exits = np.where(diffs == -1)[0]

            for en, ex in zip(enters, exits):
                dwell = t[ex - 1] - t[en] if ex > en else 0.0
                if dwell >= min_dwell_min:
                    visits[row["label"]].append((sheep_id, float(t[en]), float(t[ex - 1])))

    return visits


# ---------------------------------------------------------------------------
# Path length utilities
# ---------------------------------------------------------------------------

def cumulative_path_length(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Compute cumulative path length in grid units (1 unit ≈ 10 m)."""
    dx = np.diff(gx)
    dy = np.diff(gy)
    step = np.sqrt(dx**2 + dy**2)
    return np.concatenate([[0.0], np.cumsum(step)])
