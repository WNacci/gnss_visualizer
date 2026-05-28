"""Analysis helper functions: site visit detection and path length."""
import numpy as np
import pandas as pd

from gps_analysis.config import DATA_DIR


def detect_site_visits(
    sheep_tracks: dict,
    field: str = "A",
    radius: float = 0.5,
    min_dwell_s: float = 0.0,
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


def detect_recruitment_episodes(
    visits: dict,
    episode_gap: float = 1.0,
) -> list[dict]:
    """Group site visits into recruitment episodes.

    An episode begins when a sheep enters a site and ends when no new sheep
    has entered for *episode_gap* minutes.  The first sheep to enter is the
    **initiator**; all other sheep entering during the episode are
    **followers**.

    Parameters
    ----------
    visits : dict
        Output of detect_site_visits():
        site_label → list of (sheep_id, entry_min, exit_min).
    episode_gap : float
        Maximum gap (minutes) between consecutive entries within one episode.

    Returns
    -------
    list of dict, each with keys:
        'site': str — site label,
        'time': float — initiator entry time (minutes),
        'initiator': str — sheep_id of the first sheep to enter,
        'followers': list of {'id': str, 'time': float} — subsequent sheep.
    """
    episodes = []
    for site, vlist in sorted(visits.items()):
        if not vlist:
            continue
        sorted_v = sorted(vlist, key=lambda x: x[1])
        ep_initiator = sorted_v[0][0]
        ep_start = sorted_v[0][1]
        ep_followers = []
        prev_entry = sorted_v[0][1]

        for sheep_id, entry_min, _exit_min in sorted_v[1:]:
            if entry_min - prev_entry > episode_gap:
                episodes.append({
                    "site": site,
                    "time": ep_start,
                    "initiator": ep_initiator,
                    "followers": ep_followers,
                })
                ep_initiator = sheep_id
                ep_start = entry_min
                ep_followers = []
            else:
                if sheep_id != ep_initiator:
                    ep_followers.append({"id": sheep_id, "time": entry_min})
            prev_entry = entry_min

        episodes.append({
            "site": site,
            "time": ep_start,
            "initiator": ep_initiator,
            "followers": ep_followers,
        })
    return episodes


def cumulative_path_length(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Compute cumulative path length in grid units (1 unit ≈ 10 m)."""
    dx = np.diff(gx)
    dy = np.diff(gy)
    step = np.sqrt(dx**2 + dy**2)
    return np.concatenate([[0.0], np.cumsum(step)])
