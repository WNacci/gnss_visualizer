"""Arena coordinate transforms: GPS → grid space."""
import numpy as np
import pandas as pd

from gps_analysis.config import DATA_DIR, GPS_OFFSET, _R


def _latlon_to_meters(lats, lons, lat0, lon0):
    lat0_r = np.radians(lat0)
    x = (lons - lon0) * np.cos(lat0_r) * (np.pi / 180) * _R
    y = (lats - lat0) * (np.pi / 180) * _R
    return x, y


def build_arena_transforms(reward_sites_df=None):
    """Fit affine transforms from GPS → arena grid coords for each field.

    Uses all 16 known site positions (12 reward + 4 corners) as control
    points for a robust overdetermined fit.

    Returns dict: field → {'lat0', 'lon0', 'M'}.
    """
    if reward_sites_df is None:
        reward_sites_df = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
    transforms = {}
    for field in ["A", "B"]:
        fdf = reward_sites_df[reward_sites_df["field"] == field]
        # Use ALL known positions (corners + interior sites) for a robust fit
        lat0 = fdf["latitude"].mean()
        lon0 = fdf["longitude"].mean()
        xm, ym = _latlon_to_meters(
            fdf["latitude"].values, fdf["longitude"].values, lat0, lon0
        )
        A_mat = np.column_stack([xm, ym, np.ones(len(xm))])
        dst = fdf[["grid_x", "grid_y"]].values.astype(float)
        M, _, _, _ = np.linalg.lstsq(A_mat, dst, rcond=None)

        # Diagnostic: check residuals
        pred = A_mat @ M
        err_m = np.linalg.norm(pred - dst, axis=1) * 10
        if err_m.max() > 0.5:
            print(f"  Arena transform field {field}: mean err {err_m.mean():.2f}m, "
                  f"max {err_m.max():.2f}m (16 GCPs)")

        transforms[field] = {"lat0": lat0, "lon0": lon0, "M": M}
    return transforms


def latlon_to_grid(lats, lons, transform):
    """Project lat/lon arrays to arena grid coordinates (≈0–5 range).

    Includes an empirical offset correction for systematic GPS bias,
    measured from sheep clustering at known reward site positions.
    """
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


def _best_field(lats, lons, csv_field, arena_transforms):
    """Return the field transform that keeps the most GPS points inside the arena.

    Falls back to csv_field if no better transform is found.
    """
    def _in_arena_count(f):
        gx, gy = latlon_to_grid(lats, lons, arena_transforms[f])
        return int(np.sum((gx >= 0) & (gx <= 5) & (gy >= 0) & (gy <= 5)))

    if csv_field not in arena_transforms:
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
