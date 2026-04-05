"""Path Length & Trial Completion Analysis

Measures cumulative path length for each sheep and detects when reward sites
are first visited.  The "trial completion time" is defined as the moment the
last un-visited reward site first receives a sustained visit.  Path length to
that event is the primary foraging efficiency metric.

Todos addressed: #4 (path length, trial end timing)
"""
import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import marimo as mo
    from analysis_utils import (
        build_trials, load_gnss_date, build_arena_transforms,
        load_trial_tracks, detect_site_visits, cumulative_path_length,
        DATA_DIR,
    )

    TRIALS = build_trials()
    ARENA_TRANSFORMS = build_arena_transforms()

    print(f"Loaded {len(TRIALS)} trials")
    return (
        np, pd, plt, mo, matplotlib,
        build_trials, load_gnss_date, build_arena_transforms,
        load_trial_tracks, detect_site_visits, cumulative_path_length,
        DATA_DIR, TRIALS, ARENA_TRANSFORMS,
    )


@app.cell(hide_code=True)
def _(TRIALS, mo):
    _options = {}
    for _i, _t in enumerate(TRIALS):
        _assay_str = f" [Assay {_t['assay']}]" if _t['assay'] is not None else ""
        _label = f"[{_i:3d}] {_t['notes'].split(' - ')[0]:20s} {_t['name']}{_assay_str}"
        _options[_label] = _i

    trial_selector = mo.ui.dropdown(options=_options, label="Select trial")
    radius_slider = mo.ui.slider(
        start=0.1, stop=2.0, step=0.1, value=0.5,
        label="Visit detection radius (grid units)",
    )
    dwell_slider = mo.ui.slider(
        start=1, stop=60, step=1, value=5,
        label="Min dwell time (s)",
    )
    n_sites_to_find = mo.ui.slider(
        start=1, stop=12, step=1, value=3,
        label="# sites needed to end trial",
    )

    mo.md(f"""
    # Path Length & Trial Completion

    Computes cumulative path length per sheep and marks reward site visit events.
    The **trial completion point** is when the N-th unique site is first visited
    (across all sheep in the group).

    {trial_selector}
    {mo.hstack([radius_slider, dwell_slider, n_sites_to_find])}
    """)
    return trial_selector, radius_slider, dwell_slider, n_sites_to_find


@app.cell(hide_code=True)
def _(
    trial_selector, TRIALS, mo,
    load_gnss_date, load_trial_tracks, detect_site_visits,
    cumulative_path_length, ARENA_TRANSFORMS, DATA_DIR,
    np, pd, plt,
    radius_slider, dwell_slider, n_sites_to_find,
):
    if trial_selector.value is None:
        mo.stop(True, mo.md("*Select a trial above.*"))

    _tidx = trial_selector.value
    _trial = TRIALS[_tidx]
    _radius = radius_slider.value
    _min_dwell = dwell_slider.value
    _n_needed = n_sites_to_find.value

    _gnss = load_gnss_date(_trial['date'])
    _tracks = load_trial_tracks(
        _trial, gnss_cache={_trial['date']: _gnss},
        apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
    )
    if not _tracks:
        mo.stop(True, mo.md("*No GPS data found for this trial.*"))

    _rdf = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
    _visits = detect_site_visits(
        _tracks, _trial['field'],
        radius=_radius, min_dwell_s=float(_min_dwell),
        reward_sites_df=_rdf,
    )

    # Collect all first-visit events (site, time) sorted chronologically
    _first_visits = {}
    for _lbl, _vlist in _visits.items():
        if _vlist:
            _first_visits[_lbl] = min(v[1] for v in _vlist)
    _sorted_first = sorted(_first_visits.items(), key=lambda x: x[1])

    # Trial completion = when N-th site is first found
    _completion_time = None
    if len(_sorted_first) >= _n_needed:
        _completion_time = _sorted_first[_n_needed - 1][1]

    # Compute per-sheep path length curves
    COLORS = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#a65628', '#f781bf', '#999999',
    ]

    _fig, (_ax_path, _ax_speed) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]},
    )

    _dur = _trial['duration_min']
    _sheep_ids = sorted(_tracks.keys())

    _summary_rows = []
    _all_path_at_completion = []

    for _ci, _sheep_id in enumerate(_sheep_ids):
        _trk = _tracks[_sheep_id]
        _gx, _gy, _t = _trk['gx'], _trk['gy'], _trk['t']
        _color = COLORS[_ci % len(COLORS)]

        # Cumulative path length
        _pl = cumulative_path_length(_gx, _gy)  # in grid units
        _pl_m = _pl * 10.0  # convert to metres

        # Speed (m/min, smoothed with 10-sample rolling mean)
        _dt = np.diff(_t)
        _dt_safe = np.where(_dt > 0, _dt, 1e-6)
        _speed = np.diff(_pl_m) / _dt_safe
        _speed_smooth = np.convolve(
            _speed, np.ones(20) / 20, mode='same'
        )
        _t_mid = 0.5 * (_t[:-1] + _t[1:])

        _ax_path.plot(_t, _pl_m, color=_color, lw=1.5, label=_sheep_id)
        _ax_speed.plot(_t_mid, _speed_smooth, color=_color, lw=1.0, alpha=0.8)

        # Path at completion
        _path_at_c = None
        if _completion_time is not None:
            _idx_c = np.searchsorted(_t, _completion_time)
            if _idx_c < len(_pl_m):
                _path_at_c = float(_pl_m[_idx_c])
                _all_path_at_completion.append(_path_at_c)

        _summary_rows.append({
            'Sheep': _sheep_id,
            'Total path (m)': round(float(_pl_m[-1]), 1),
            'Duration (min)': round(float(_t[-1] - _t[0]), 2),
            'Mean speed (m/min)': round(float(_pl_m[-1] / max(_t[-1] - _t[0], 0.001)), 1),
            'Path to completion (m)': round(_path_at_c, 1) if _path_at_c is not None else 'N/A',
        })

    # Mark site discovery events on path plot
    for _si, (_lbl, _t_first) in enumerate(_sorted_first):
        _ax_path.axvline(_t_first, color='gray', lw=0.8, ls='--', alpha=0.6)
        _ax_path.text(_t_first + 0.1, _ax_path.get_ylim()[1] * 0.98,
                      _lbl, fontsize=6, va='top', rotation=90, color='gray')

    # Mark completion time
    if _completion_time is not None:
        _ax_path.axvline(_completion_time, color='#d63b3b', lw=2, ls='-',
                         label=f'Trial complete (site #{_n_needed} found)')
        _ax_speed.axvline(_completion_time, color='#d63b3b', lw=2, ls='-')

    _ax_path.set_ylabel("Cumulative path length (m)")
    _ax_path.legend(loc='upper left', fontsize=9)
    _ax_path.set_xlim(0, _dur)

    _ax_speed.set_xlabel("Time (min)")
    _ax_speed.set_ylabel("Speed (m/min, smoothed)")
    _ax_speed.set_ylim(bottom=0)

    _n_found = len(_sorted_first)
    _c_str = f"{_completion_time:.1f} min" if _completion_time is not None else "not reached"
    _fig.suptitle(
        f"Path length — {_trial['name']}\n"
        f"Sites found: {_n_found}/12  |  "
        f"Completion time (site #{_n_needed}): {_c_str}",
        fontsize=11,
    )
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack([
        mo.mpl.interactive(_fig),
        mo.md("### Per-sheep summary"),
        mo.ui.table(pd.DataFrame(_summary_rows)),
    ])
    return


@app.cell(hide_code=True)
def _(
    TRIALS, mo,
    load_gnss_date, load_trial_tracks, detect_site_visits,
    cumulative_path_length, ARENA_TRANSFORMS, DATA_DIR,
    np, pd, plt,
):
    """Aggregate path length statistics across all trials.

    Shows distribution of path-to-completion across assay levels.
    """
    mo.md("---\n## Aggregate: path length to 3rd site found, by assay")
    return


@app.cell(hide_code=True)
def _(
    TRIALS, mo,
    load_gnss_date, load_trial_tracks, detect_site_visits,
    cumulative_path_length, ARENA_TRANSFORMS, DATA_DIR,
    np, pd, plt,
):
    """Aggregate path length across test-configuration trials."""
    _TEST_CONFIGS = {'A', 'B', 'C', 'D'}
    _records = []

    for _tidx, _trial in enumerate(TRIALS):
        if _trial['config'] not in _TEST_CONFIGS:
            continue
        _gnss = load_gnss_date(_trial['date'])
        _tracks = load_trial_tracks(
            _trial, gnss_cache={_trial['date']: _gnss},
            apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
        )
        if not _tracks:
            continue
        _rdf = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
        _visits = detect_site_visits(
            _tracks, _trial['field'], radius=0.5, min_dwell_s=5.0,
            reward_sites_df=_rdf,
        )
        _first_visits = {}
        for _lbl, _vlist in _visits.items():
            if _vlist:
                _first_visits[_lbl] = min(v[1] for v in _vlist)
        _sorted_first = sorted(_first_visits.items(), key=lambda x: x[1])

        # Trial completion = 3rd site found
        _completion_time = _sorted_first[2][1] if len(_sorted_first) >= 3 else None

        # Mean path length across all sheep at completion
        _path_vals = []
        for _sheep_id, _trk in _tracks.items():
            _gx, _gy, _t = _trk['gx'], _trk['gy'], _trk['t']
            _pl = cumulative_path_length(_gx, _gy) * 10.0
            if _completion_time is not None:
                _idx_c = np.searchsorted(_t, _completion_time)
                if _idx_c < len(_pl):
                    _path_vals.append(_pl[_idx_c])

        _records.append({
            'Trial': _tidx,
            'Date': _trial['date'],
            'Field': _trial['field'],
            'Config': _trial['config'],
            'Group size': _trial['group_size'],
            'Assay': str(_trial['assay']),
            'Sites found': len(_sorted_first),
            'Completion time (min)': round(_completion_time, 2) if _completion_time is not None else None,
            'Mean path to completion (m)': round(float(np.mean(_path_vals)), 1) if _path_vals else None,
        })

    if not _records:
        mo.stop(True, mo.md("*No test-configuration trials with GPS data found.*"))

    _agg_df = pd.DataFrame(_records)
    _agg_df = _agg_df.dropna(subset=['Completion time (min)'])

    _fig2, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Box plots by assay
    _assays = sorted(_agg_df['Assay'].unique(), key=lambda x: (not x.isdigit(), x))
    _ct_by_assay = [_agg_df[_agg_df['Assay'] == a]['Completion time (min)'].dropna().values
                    for a in _assays]
    _pl_by_assay = [_agg_df[_agg_df['Assay'] == a]['Mean path to completion (m)'].dropna().values
                    for a in _assays]

    _ax1.boxplot(_ct_by_assay, labels=_assays)
    _ax1.set_xlabel("Assay")
    _ax1.set_ylabel("Completion time (min)")
    _ax1.set_title("Time to find 3rd site")

    _ax2.boxplot(_pl_by_assay, labels=_assays)
    _ax2.set_xlabel("Assay")
    _ax2.set_ylabel("Mean path length (m)")
    _ax2.set_title("Path length to find 3rd site")

    _fig2.suptitle("Path length / completion time across all test trials  (radius=0.5, dwell≥5 s)")
    _fig2.tight_layout()
    plt.close(_fig2)

    mo.vstack([
        mo.mpl.interactive(_fig2),
        mo.md("### Aggregate table"),
        mo.ui.table(_agg_df),
    ])
    return


if __name__ == "__main__":
    app.run()
