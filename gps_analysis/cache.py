"""Disk caching for GPS data and pre-computed tracks."""
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gps_analysis.config import DATA_DIR
from gps_analysis.io import load_gnss_date
from gps_analysis.coords import build_arena_transforms
from gps_analysis.trials import build_trials
from gps_analysis.tracks import load_trial_tracks

_GPS_CACHE_PATH = DATA_DIR / "gps_cache.pkl"
_TRACKS_CACHE_PATH = DATA_DIR / "tracks_cache.pkl"


def build_gps_cache(
    trials: list | None = None,
    max_workers: int = 8,
    cache_path: Path | None = _GPS_CACHE_PATH,
    force_rebuild: bool = False,
) -> dict:
    """Pre-load all GNSS data for every trial date in parallel.

    Returns dict: date_str → {device_num (int) → (lats, lons, times)}.
    Call once at notebook startup; pass the result as ``gnss_cache`` to
    load_trial_tracks().

    On first run the result is serialised to ``cache_path`` (default:
    ``data/gps_cache.pkl``).  Subsequent calls load from that file
    instantly instead of re-reading the raw GNSS logs.

    Pass ``force_rebuild=True`` or call rebuild_caches() to
    discard the cached file and re-read from the raw GNSS logs.
    Pass ``cache_path=None`` to skip disk caching entirely.
    """
    if not force_rebuild and cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
        total_pts = sum(
            len(lats) for day_data in cache.values() for (lats, _, _) in day_data.values()
        )
        print(f"GPS cache loaded from disk: {len(cache)} dates, {total_pts:,} total points")
        return cache

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
    print(f"GPS cache built: {len(cache)} dates, {total_pts:,} total points")

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"GPS cache saved to {cache_path}")

    return cache


def build_tracks_cache(
    trials: list | None = None,
    gnss_cache: dict | None = None,
    arena_transforms: dict | None = None,
    cache_path: Path | None = _TRACKS_CACHE_PATH,
    force_rebuild: bool = False,
) -> dict:
    """Pre-compute and cache projected tracks (raw and orientation-normalised).

    Returns dict: trial_name → {"raw": tracks, "oriented": tracks}
    where each ``tracks`` value is the output of load_trial_tracks().

    Both orientation variants are computed here from the same GPS data and the
    same ``CONFIG_TRANSFORMS`` table, so every script that reads from this cache
    is guaranteed to use identical coordinate transforms.

    On first run the result is serialised to ``cache_path`` (default:
    ``data/tracks_cache.pkl``).  Subsequent calls return instantly.

    Pass ``force_rebuild=True`` or call rebuild_caches() to regenerate.
    Pass ``cache_path=None`` to skip disk caching.
    """
    if trials is None:
        trials = build_trials()

    if not force_rebuild and cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as fh:
            tc = pickle.load(fh)
        # Rebuild automatically if any current trial name is absent from the
        # cache — this catches stale pickles from before the name was fixed to
        # include group_num, or any future schema change.
        _missing = [t["name"] for t in trials if t["name"] not in tc]
        if _missing:
            print(
                f"Tracks cache stale ({len(_missing)} trial name(s) missing, "
                f'e.g. "{_missing[0]}") — rebuilding…'
            )
            # fall through to rebuild
        else:
            print(f"Tracks cache loaded from disk: {len(tc)} trials")
            return tc
    if gnss_cache is None:
        gnss_cache = build_gps_cache(trials)
    if arena_transforms is None:
        arena_transforms = build_arena_transforms()

    def _process_trial(trial):
        raw = load_trial_tracks(
            trial, gnss_cache=gnss_cache, apply_orient=False,
            arena_transforms=arena_transforms,
        )
        oriented = load_trial_tracks(
            trial, gnss_cache=gnss_cache, apply_orient=True,
            arena_transforms=arena_transforms,
        )
        return trial["name"], {"raw": raw, "oriented": oriented}

    tc: dict = {}
    # Parallelise: Kalman smoothing is numpy-heavy and releases the GIL
    with ThreadPoolExecutor(max_workers=min(8, len(trials))) as pool:
        for name, entry in pool.map(_process_trial, trials):
            tc[name] = entry

    print(f"Tracks cache built: {len(tc)} trials")

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(tc, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tracks cache saved to {cache_path}")

    return tc


def rebuild_caches(
    gps_cache_path: Path | None = _GPS_CACHE_PATH,
    tracks_cache_path: Path | None = _TRACKS_CACHE_PATH,
) -> tuple[dict, dict]:
    """Delete all on-disk caches and rebuild from raw GNSS data.

    Returns (gnss_cache, tracks_cache) — the same objects that would be
    returned by calling build_gps_cache() and build_tracks_cache() independently.

    Call this from any notebook after adding new trial data::

        from gps_analysis import rebuild_caches
        GPS_CACHE, TRACKS_CACHE = rebuild_caches()
    """
    for p in [gps_cache_path, tracks_cache_path]:
        if p is not None and Path(p).exists():
            Path(p).unlink()
            print(f"Deleted {p}")

    trials = build_trials()
    arena_transforms = build_arena_transforms()
    gnss_cache = build_gps_cache(trials, cache_path=gps_cache_path)
    tracks_cache = build_tracks_cache(
        trials, gnss_cache=gnss_cache, arena_transforms=arena_transforms,
        cache_path=tracks_cache_path,
    )
    return gnss_cache, tracks_cache
