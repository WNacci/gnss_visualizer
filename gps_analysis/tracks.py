"""GPS-to-track processing pipeline."""
import numpy as np
import pandas as pd

from gps_analysis.config import CONFIG_TRANSFORMS, GPS_OFFSET
from gps_analysis.io import load_gnss_date
from gps_analysis.coords import latlon_to_grid, apply_orientation, build_arena_transforms, _best_field
from gps_analysis._signal import kalman_smooth_track


def load_trial_tracks(
    trial: dict,
    gnss_cache: dict | None = None,
    apply_orient: bool = False,
    arena_transforms: dict | None = None,
    tracks_cache: dict | None = None,
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
    tracks_cache : dict, optional
        Pre-computed tracks from build_tracks_cache().
        When provided, the result is looked up directly without any GPS
        processing — all scripts that pass the same cache are guaranteed to
        use identical coordinate transforms.

    Returns
    -------
    dict : sheep_id → {'gx': array, 'gy': array, 't': array}
        where t is minutes from trial start.
    """
    if tracks_cache is not None:
        variant = "oriented" if apply_orient else "raw"
        entry = tracks_cache.get(trial["name"])
        if entry is not None:
            return entry[variant]

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
            # Apply per field+config GPS offset correction (oriented coords)
            _off = GPS_OFFSET.get((field, trial["config"]), (0, 0))
            gx = gx - _off[0] / 10.0
            gy = gy - _off[1] / 10.0

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

        # Apply Kalman + RTS smoothing for optimal position estimates
        gx_s, gy_s = kalman_smooth_track(gx_f, gy_f, t_f)
        result[sheep_id] = {"gx": gx_s, "gy": gy_s, "t": t_f}

    return result


def load_trial_device_tracks(
    trial: dict,
    gnss_cache: dict | None = None,
    apply_orient: bool = False,
    arena_transforms: dict | None = None,
) -> dict:
    """Like load_trial_tracks but keeps each GPS device as a separate entry.

    Sheep that carried two devices are returned as two independent tracks that
    share the same ``sheep_id`` — callers should use ``sheep_id`` to assign
    colour and ``dev_num`` as a unique key.

    Parameters
    ----------
    trial, gnss_cache, apply_orient, arena_transforms
        Same as load_trial_tracks(). tracks_cache is intentionally not
        supported because the pre-built cache stores averaged tracks.

    Returns
    -------
    dict : dev_num (int) → {'gx', 'gy', 't', 'sheep_id'}
        ``sheep_id`` is the ID of the sheep that wore the device, or
        ``f"Dev{dev_num}"`` when the mapping is unknown.
    """
    date = trial["date"]
    raw = gnss_cache.get(date, {}) if gnss_cache is not None else load_gnss_date(date)
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

    dt_str = f"{date} {trial['start_time']}"
    start_unix = (
        pd.to_datetime(dt_str)
        .tz_localize("Europe/Paris")
        .tz_convert("UTC")
        .timestamp()
    )
    end_unix = start_unix + trial["duration_min"] * 60

    # Auto-detect best field (same logic as load_trial_tracks)
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
    result = {}

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
            _off = GPS_OFFSET.get((field, trial["config"]), (0, 0))
            gx = gx - _off[0] / 10.0
            gy = gy - _off[1] / 10.0
        # Kalman + RTS smooth
        gx, gy = kalman_smooth_track(gx, gy, t_rel)
        sheep_id = d2s.get(dev_num, f"Dev{dev_num}")
        result[dev_num] = {"gx": gx, "gy": gy, "t": t_rel, "sheep_id": sheep_id}

    return result
