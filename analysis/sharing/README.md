# Sheep GNSS dataset — Phase 2 (Feb 2026)

Raw 10 Hz GNSS tracks from sheep wearing GPS collars during foraging trials
in a 50 m × 50 m arena, plus the metadata and loader code needed to produce
per-trial per-sheep tracks in arena grid coordinates.

## What's in the package

| File | Description |
|---|---|
| `README.md` | This file. |
| `sheep_gnss_loader.py` | Self-contained loader. Depends on `numpy` and `pandas` only. |
| `example_usage.py` | Worked example: load trials, plot one trial. |
| `trial_metadata.csv` | One row per (sheep, trial); maps GNSS device IDs to sheep. |
| `fitted_reward_sites.csv` | Fitted lat/lon for each reward site and arena corner per field. |
| `gnss/` | Raw GNSS log files, organised as `gnss/{day}-02-26/GNSS-{N}/LOGS*.TXT`. |

## Quick start

```bash
tar --zstd -xf sheep_gnss_2026.tar.zst
cd sheep_gnss_2026
pip install numpy pandas matplotlib   # the only dependencies
python example_usage.py
```

Then in your own code:

```python
from sheep_gnss_loader import load_trials, load_trial_tracks

trials = load_trials("trial_metadata.csv")
tracks = load_trial_tracks(
    trials[0],
    gnss_root="gnss/",
    reward_sites_csv="fitted_reward_sites.csv",
)
# tracks = {sheep_id: {"gx": np.array, "gy": np.array, "t": np.array}}
# gx, gy in arena grid units (0–5; 1 unit = 10 m)
# t in minutes from trial start
```

## Coordinate system

Arena grid is 0–5 in each axis. **1 grid unit = 10 m.** Origin is the
bottom-left corner; `y` increases north, `x` increases east. The 12 reward
sites occupy integer positions in a 4×3 arrangement:

```
y=4    B1  A3
y=3  C1 D2 C2 D3
y=2  A1 B2 A2 B3
y=1    D1 C3
       1  2  3  4
```

The mapping from raw GPS lat/lon to arena grid is an **affine transform**
fit per field (A or B) by overdetermined least-squares using all 16 known
control points (12 reward sites + 4 corner markers E1–E4). The transform
parameters come from `fitted_reward_sites.csv`.

The reward site positions in that CSV were measured in a dedicated
calibration session on **2026-01-26**: a GNSS device was placed at each
known site for several minutes, and the per-site position is the median of
the later portion of the recording (when satellite lock is best).

## Data schema

### `trial_metadata.csv`

| Column | Description |
|---|---|
| `date` | Trial date `YYYY-MM-DD`. |
| `Sheep ID` | 5-digit internal lab ID for the sheep (string, preserve leading zeros). |
| `Group #` | Group identifier (integer). Same group = same animals across trials. |
| `Group Size` | Number of sheep in the group (Phase 2: always 4). |
| `Sheep #` | Within-group sheep index. |
| `GNSS_SN1`, `GNSS_SN2` | GNSS device numbers worn by this sheep (1–2 devices). |
| `field` | Arena field (`A` or `B`). |
| `start_time` | Trial start, `HH:MM:SS` in **Paris local time** (Europe/Paris). |
| `configuration` | Trial configuration: `A`/`B`/`C`/`D` (test) or `CTRL_FAR`/`CTRL_BARN` (control). |
| `assay` | Within-group trial index (0–7); tracks learning across days. |
| `note` | Free-text notes. |

Test configurations bait a **fixed** triplet of sites every trial (different
triplet per config A/B/C/D), allowing spatial learning across assays. Control
configurations bait a **random** triplet each trial — sheep do encounter
reward, but cannot accumulate spatial memory for fixed locations.

Trial duration is **35 minutes** (hardcoded in the loader). Rows without
`start_time` are data-cleaning artifacts and are skipped.

### `fitted_reward_sites.csv`

| Column | Description |
|---|---|
| `field` | `A` or `B`. |
| `label` | Site label: 12 reward sites (`A1`–`D3`) + 4 corner markers (`E1`–`E4`). |
| `grid_x`, `grid_y` | Integer arena grid coordinates. |
| `latitude`, `longitude` | GPS coordinates of the site (decimal degrees, WGS84). |

### `gnss/{day}-02-26/GNSS-{N}/LOGS*.TXT`

Plain text, one reading per line. Colon-delimited fields:

```
T{dev}:{sats}:{dop}:{lat}:{lon}:{unix_time}:{ax}:{ay}:{az}:
```

| Field | Meaning |
|---|---|
| `T{dev}` | Device tag (parseable but not used downstream). |
| `sats` | Visible satellite count. |
| `dop` | Dilution of precision. |
| `lat`, `lon` | Decimal degrees, WGS84. |
| `unix_time` | Unix seconds (UTC, with millisecond fraction). |
| `ax`, `ay`, `az` | Accelerometer reading, units approximately g. |

The loader extracts only `(lat, lon, unix_time)`. The accelerometer fields
are present if you want to use them.

Per-device per-day there are typically 10–15 files (`LOGS5.TXT` through
`LOGS18.TXT`, etc.). Files are processed in lexicographic order, which
matches chronological order.

## Trial design summary

- **Arena**: 50 m × 50 m, 12 reward sites on a 4×3 grid in a roughly
  octagonal layout.
- **Group**: 4 sheep per trial in Phase 2 (this dataset).
- **Assays**: each group runs trials on successive days, numbered by `assay`
  (0 = first/naive, 7 = experienced).
- **Test configurations** (`A`, `B`, `C`, `D`): each baits a fixed triplet
  of 3 sites every trial. After `apply_orient=True`, the canonical baited
  triplet is always `{A1, A2, A3}`.
- **Control configurations** (`CTRL_FAR`, `CTRL_BARN`): reward is placed at
  random sites each trial. No stable spatial memory is possible.

**Important limitation:** the per-trial CTRL baiting positions are NOT in
the metadata pipeline. You can see which trials are CTRL via the
`configuration` column, but you cannot reconstruct *which 3 sites were
baited* in any given CTRL trial from this package alone.

## Known systematic

There is a residual ~0.5–1 m systematic GPS bias per `(field, configuration)`
combination, measured by the offset between where experienced sheep cluster
and the known reward site coordinates. This package does **not** apply that
correction — you get the raw projected coordinates.

If you want to apply your own correction, the recommended approach is: fit
a 2D offset per `(field, config)` by minimising the distance between
high-density sheep dwell positions (in experienced trials, assay ≥ 2) and
the nominal reward site grid positions.

## Two GNSS devices per sheep

In Phase 2, every sheep wears two GNSS devices. The loader interpolates
both devices to a common 10 Hz time grid and averages them per timestep,
yielding one track per sheep. If you want device-level data instead, copy
the per-device loop from `load_trial_tracks` and skip the averaging step.

## Time zone

`start_time` in the metadata is **Paris local time** (Europe/Paris). The
loader converts it to UTC internally to align with GNSS Unix timestamps.

## Optional enhancements

The loader is deliberately minimal. If you want to add:

- **Kalman + RTS smoothing** for cleaner tracks: install `filterpy` and
  apply a 4-state constant-velocity Kalman filter (lat, lon, vx, vy) with
  measurement noise scaled by your expected GPS accuracy (~1–2 m), then run
  the RTS smoother backward.
- **Trial caching** for repeated analyses: pickle the output of
  `load_trial_tracks` per trial.

## Citation and license

- **Citation**: TODO — fill in lab citation and contact email.
- **License**: TODO — fill in (e.g. CC-BY-4.0).
- **Contact**: TODO — fill in.

## Provenance

This dataset and loader were prepared from the lab's internal analysis
repository. The loader is a simplified extraction of the larger
`gps_analysis` package used internally; behaviour matches the internal
pipeline except that GPS_OFFSET corrections and Kalman smoothing are
omitted (see "Known systematic" and "Optional enhancements" above).
