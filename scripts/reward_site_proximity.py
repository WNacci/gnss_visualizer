"""Reward Site Proximity Analysis

For each trial, tracks how many sheep are within a radius of each of the
12 reward sites throughout the trial. Collapses the 2-D trajectory data
down to a ~12-dimensional time series that clearly shows discovery events.

Todos addressed: #2 (radii around reward sites), #9 (temporal progression)
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
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba
    import marimo as mo
    from analysis_utils import (
        build_trials, build_gps_cache, load_gnss_date, build_arena_transforms,
        load_trial_tracks, detect_site_visits, SITE_LABELS, DATA_DIR,
    )

    TRIALS = build_trials()
    ARENA_TRANSFORMS = build_arena_transforms()

    print(f"Loaded {len(TRIALS)} trials — building GPS cache (runs once)…")
    GPS_CACHE = build_gps_cache(TRIALS)
    return (
        np, pd, plt, mo, mpatches, to_rgba, matplotlib,
        build_trials, load_gnss_date, build_arena_transforms,
        load_trial_tracks, detect_site_visits, SITE_LABELS, DATA_DIR,
        TRIALS, ARENA_TRANSFORMS, GPS_CACHE,
    )


@app.cell(hide_code=True)
def _(TRIALS, mo):
    _options = {}
    for _i, _t in enumerate(TRIALS):
        _assay_str = f" [Assay {_t['assay']}]" if _t['assay'] is not None else ""
        _label = f"[{_i:3d}] {_t['notes'].split(' - ')[0]:20s} {_t['name']}{_assay_str}"
        _options[_label] = _i

    trial_selector = mo.ui.dropdown(
        options=_options,
        label="Select trial",
    )
    radius_slider = mo.ui.slider(
        start=0.1, stop=2.0, step=0.1, value=0.5,
        label="Detection radius (grid units, 1 unit ≈ 10 m)",
    )
    bin_size_slider = mo.ui.slider(
        start=5, stop=120, step=5, value=30,
        label="Time bin size (seconds)",
    )
    smooth_slider = mo.ui.slider(
        start=0, stop=10, step=1, value=2,
        label="Smoothing (bins)",
    )

    mo.md(f"""
    # Reward Site Proximity Analysis

    Track how many sheep are within a given radius of each of the 12 reward sites
    over the course of a trial.  A spike from 0 → N indicates a discovery event.

    {trial_selector}

    {mo.hstack([radius_slider, bin_size_slider, smooth_slider])}
    """)
    return trial_selector, radius_slider, bin_size_slider, smooth_slider


@app.cell(hide_code=True)
def _(
    trial_selector, TRIALS, mo,
    GPS_CACHE, load_trial_tracks, detect_site_visits,
    ARENA_TRANSFORMS, DATA_DIR, np, pd,
    radius_slider, bin_size_slider, smooth_slider,
    plt, mpatches,
):
    if trial_selector.value is None:
        mo.stop(True, mo.md("*Select a trial above.*"))

    _tidx = trial_selector.value
    _trial = TRIALS[_tidx]
    _radius = radius_slider.value
    _bin_s = bin_size_slider.value
    _smooth = smooth_slider.value

    # Load GPS data
    _tracks = load_trial_tracks(
        _trial, gnss_cache=GPS_CACHE,
        apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
    )

    if not _tracks:
        mo.stop(True, mo.md("*No GPS data found for this trial.*"))

    _dur_min = _trial['duration_min']
    _n_sheep = len(_tracks)

    # Load reward site positions for this field
    _rdf = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
    _field_sites = _rdf[
        (_rdf['field'] == _trial['field']) &
        (~_rdf['label'].str.startswith('E'))
    ].sort_values('label').reset_index(drop=True)

    # Build time bins (in minutes)
    _bin_min = _bin_s / 60.0
    _t_edges = np.arange(0, _dur_min + _bin_min, _bin_min)
    _t_centres = 0.5 * (_t_edges[:-1] + _t_edges[1:])
    _n_bins = len(_t_centres)

    # For each site, for each bin: count sheep within radius
    _site_counts = {}
    for _, _row in _field_sites.iterrows():
        _sx, _sy = _row['grid_x'], _row['grid_y']
        _site_tc = np.zeros(_n_bins, dtype=float)
        for _sheep_id, _trk in _tracks.items():
            _gx, _gy, _t = _trk['gx'], _trk['gy'], _trk['t']
            _dist = np.sqrt((_gx - _sx)**2 + (_gy - _sy)**2)
            _near = (_dist <= _radius).astype(float)
            # Bin: fraction of time inside radius for this bin
            for _bi in range(_n_bins):
                _mask_b = (_t >= _t_edges[_bi]) & (_t < _t_edges[_bi + 1])
                if _mask_b.sum() > 0:
                    _site_tc[_bi] += _near[_mask_b].mean()
        _site_counts[_row['label']] = _site_tc

    # Optional smoothing (simple moving average)
    def _smooth_arr(arr, w):
        if w <= 1:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='same')

    # Detect first visit times (entry into radius, sustained ≥5 s)
    _visits = detect_site_visits(
        _tracks, _trial['field'],
        radius=_radius, min_dwell_s=5.0, reward_sites_df=_rdf,
    )
    _first_visits = {}
    for _lbl, _vlist in _visits.items():
        if _vlist:
            _first_visits[_lbl] = min(v[1] for v in _vlist)

    # --- Plot ---
    _n_sites = len(_field_sites)
    _ncols = 3
    _nrows = (_n_sites + _ncols - 1) // _ncols

    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(14, 3.5 * _nrows), sharey=True)
    _axes_flat = _axes.flatten() if hasattr(_axes, 'flatten') else [_axes]

    _site_order = [r['label'] for _, r in _field_sites.iterrows()]

    for _ai, _lbl in enumerate(_site_order):
        _ax = _axes_flat[_ai]
        _tc = _site_counts.get(_lbl, np.zeros(_n_bins))
        _tc_s = _smooth_arr(_tc, _smooth)
        _ax.bar(_t_centres, _tc, width=_bin_min * 0.9, color='#aac4de', alpha=0.7, label='raw')
        _ax.plot(_t_centres, _tc_s, color='#1a5fa8', lw=1.5, label='smoothed')
        if _lbl in _first_visits:
            _ax.axvline(_first_visits[_lbl], color='#d63b3b', lw=1.5, ls='--', label='1st visit')
        _ax.set_title(f"{_lbl}  (x={SITE_GRID_REF[_lbl][0]}, y={SITE_GRID_REF[_lbl][1]})")
        _ax.set_xlim(0, _dur_min)
        _ax.set_ylim(0, _n_sheep + 0.1)
        _ax.set_xlabel("Time (min)")
        _ax.set_ylabel("Sheep near site")
        _ax.axhline(0, color='k', lw=0.5)

    # Hide unused axes
    for _ai in range(_n_sites, len(_axes_flat)):
        _axes_flat[_ai].set_visible(False)

    _handles = [
        mpatches.Patch(color='#aac4de', alpha=0.7, label='raw'),
        plt.Line2D([0], [0], color='#1a5fa8', lw=1.5, label='smoothed'),
        plt.Line2D([0], [0], color='#d63b3b', lw=1.5, ls='--', label='1st visit'),
    ]
    _fig.legend(handles=_handles, loc='upper right', ncol=3)

    _n_first = len(_first_visits)
    _total_visits = sum(len(v) for v in _visits.values())
    _title = (
        f"Reward site proximity — {_trial['name']}\n"
        f"{_n_sheep} sheep, radius={_radius:.1f} grid units ({_radius*10:.0f} m), "
        f"bin={_bin_s}s   |   Sites with ≥1 visit: {_n_first}/12, "
        f"total visit events: {_total_visits}"
    )
    _fig.suptitle(_title, fontsize=11, y=1.01)
    _fig.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _():
    # Local reference so the plot cell can access SITE_GRID
    SITE_GRID_REF = {
        'B1': (2, 4), 'A3': (3, 4),
        'C1': (1, 3), 'D2': (2, 3), 'C2': (3, 3), 'D3': (4, 3),
        'A1': (1, 2), 'B2': (2, 2), 'A2': (3, 2), 'B3': (4, 2),
        'D1': (2, 1), 'C3': (3, 1),
    }
    return (SITE_GRID_REF,)


@app.cell(hide_code=True)
def _(
    trial_selector, TRIALS, mo,
    GPS_CACHE, load_trial_tracks, detect_site_visits,
    ARENA_TRANSFORMS, DATA_DIR, np, pd, plt,
):
    """Summary table: which sites were visited, when, and by which sheep."""
    if trial_selector.value is None:
        mo.stop(True, mo.md(""))

    _tidx = trial_selector.value
    _trial = TRIALS[_tidx]

    _tracks = load_trial_tracks(
        _trial, gnss_cache=GPS_CACHE,
        apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
    )
    if not _tracks:
        mo.stop(True, mo.md(""))

    _rdf = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
    _visits = detect_site_visits(_tracks, _trial['field'], radius=0.5, min_dwell_s=5.0, reward_sites_df=_rdf)

    _rows = []
    for _lbl, _vlist in sorted(_visits.items()):
        for _sheep_id, _t_enter, _t_exit in sorted(_vlist, key=lambda x: x[1]):
            _rows.append({
                'Site': _lbl,
                'Sheep': _sheep_id,
                'Entry (min)': round(_t_enter, 2),
                'Exit (min)': round(_t_exit, 2),
                'Dwell (s)': round((_t_exit - _t_enter) * 60, 1),
            })

    if _rows:
        _summary_df = pd.DataFrame(_rows)
        mo.vstack([
            mo.md("### Visit events (radius=0.5, dwell≥5 s)"),
            mo.ui.table(_summary_df),
        ])
    else:
        mo.md("*No visit events detected with current parameters.*")
    return


if __name__ == "__main__":
    app.run()
