"""Raw GNSS file loaders."""
import re
from pathlib import Path

import numpy as np

from gps_analysis.config import DATA_DIR


def load_gnss_dir(device_dir: Path):
    """Load GPS + IMU data from a single GNSS device directory.

    Log format per line:
      T{dev}:{sats}:{dop}:{lat}:{lon}:{unix_time}:{ax}:{ay}:{az}:

    Returns (lats, lons, times) as float64 arrays.
    Additional fields (sats, dop, accel) are available via load_gnss_dir_full().
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


def load_gnss_dir_full(device_dir: Path):
    """Load GPS + IMU data including satellite count, DOP, and accelerometer.

    Returns dict with keys: lats, lons, times, sats, dop, ax, ay, az
    (all float64 arrays of same length).
    """
    lats, lons, times, sats, dop = [], [], [], [], []
    ax, ay, az = [], [], []
    for f in sorted(device_dir.iterdir(), key=lambda x: x.name):
        if re.match(r"LOGS\d+\.TXT", f.name, re.IGNORECASE) and f.stat().st_size > 0:
            for line in open(f, errors="ignore"):
                parts = line.split(":")
                if len(parts) >= 9:
                    try:
                        lats.append(float(parts[3]))
                        lons.append(float(parts[4]))
                        times.append(float(parts[5]))
                        sats.append(float(parts[1]))
                        dop.append(float(parts[2]))
                        ax.append(float(parts[6]))
                        ay.append(float(parts[7]))
                        az.append(float(parts[8]))
                    except ValueError:
                        pass
    return {
        "lats": np.array(lats), "lons": np.array(lons), "times": np.array(times),
        "sats": np.array(sats), "dop": np.array(dop),
        "ax": np.array(ax), "ay": np.array(ay), "az": np.array(az),
    }


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
