"""Site Discovery Effects & Probability of Site Presence

Two analyses:

1. **Probability of reward-site presence over time** (Todo #3):
   A running estimate of the fraction of the group near each site at each
   timestep — a smooth "how likely are sheep at each site" view.

2. **Behaviour change around discovery events** (Todo #8):
   Compares sheep speed and group spread in windows *before* and *after*
   the first discovery of each site.  If sheep slow down or cluster after
   finding a site, that is visible here.
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
        DATA_DIR, SITE_GRID,
    )

    TRIALS = build_trials()
    ARENA_TRANSFORMS = build_arena_transforms()
    print(f"Loaded {len(TRIALS)} trials")
    return (
        np, pd, plt, mo, matplotlib,
        build_trials, load_gnss_date, build_arena_transforms,
        load_trial_tracks, detect_site_visits, cumulative_path_length,
        DATA_DIR, SITE_GRID, TRIALS, ARENA_TRANSFORMS,
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
        label="Site detection radius (grid units)",
    )
    smooth_slider = mo.ui.slider(
        start=1, stop=120, step=1, value=30,
        label="Probability smoothing (seconds)",
    )
    window_slider = mo.ui.slider(
        start=10, stop=300, step=10, value=60,
        label="Before/after window (seconds)",
    )

    mo.md(f"""
    # Site Discovery Effects

    {trial_selector}
    {mo.hstack([radius_slider, smooth_slider, window_slider])}
    """)
    return trial_selector, radius_slider, smooth_slider, window_slider


@app.cell(hide_code=True)
def _(
    trial_selector, TRIALS, mo,
    load_gnss_date, load_trial_tracks, detect_site_visits,
    ARENA_TRANSFORMS, DATA_DIR, SITE_GRID,
    np, pd, plt,
    radius_slider, smooth_slider,
):
    """Part 1: probability of site presence over time (smooth occupancy per site)."""
    if trial_selector.value is None:
        mo.stop(True, mo.md("*Select a trial above.*"))

    _tidx = trial_selector.value
    _trial = TRIALS[_tidx]
    _radius = radius_slider.value
    _smooth_s = smooth_slider.value
    _smooth_min = _smooth_s / 60.0

    _gnss = load_gnss_date(_trial['date'])
    _tracks = load_trial_tracks(
        _trial, gnss_cache={_trial['date']: _gnss},
        apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
    )
    if not _tracks:
        mo.stop(True, mo.md("*No GPS data.*"))

    _rdf = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
    _field_sites = _rdf[
        (_rdf['field'] == _trial['field']) &
        (~_rdf['label'].str.startswith('E'))
    ].sort_values('label').reset_index(drop=True)

    # Detect first visits
    _visits = detect_site_visits(_tracks, _trial['field'], radius=_radius, min_dwell_s=5.0, reward_sites_df=_rdf)
    _first_visit_t = {lbl: min(v[1] for v in vlist) for lbl, vlist in _visits.items() if vlist}

    _dur = _trial['duration_min']
    _n_sheep = len(_tracks)

    # Common 1-s time grid
    _t_common = np.arange(0, _dur + 1 / 60, 1 / 60)

    # Interpolate all sheep
    _gx_all, _gy_all = [], []
    for _sid, _trk in _tracks.items():
        _order = np.argsort(_trk['t'])
        _gx_all.append(np.interp(_t_common, _trk['t'][_order], _trk['gx'][_order]))
        _gy_all.append(np.interp(_t_common, _trk['t'][_order], _trk['gy'][_order]))
    _gx_all = np.array(_gx_all)
    _gy_all = np.array(_gy_all)

    # For each site: fraction of sheep within radius at each timestep (smoothed)
    _site_prob = {}
    for _, _row in _field_sites.iterrows():
        _sx, _sy = _row['grid_x'], _row['grid_y']
        _dist = np.sqrt((_gx_all - _sx)**2 + (_gy_all - _sy)**2)  # (n_sheep, n_t)
        _frac = (_dist <= _radius).mean(axis=0)  # (n_t,)
        # Gaussian smooth
        _win = max(1, int(_smooth_min * 60))
        _kernel = np.exp(-0.5 * (np.arange(-_win, _win + 1) / (_win / 3)) ** 2)
        _kernel /= _kernel.sum()
        _smooth = np.convolve(_frac, _kernel, mode='same')
        _site_prob[_row['label']] = _smooth

    # --- Plot probability landscape ---
    _n_sites = len(_field_sites)
    _ncols = 3
    _nrows = (_n_sites + _ncols - 1) // _ncols + 1  # +1 for summary row

    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(14, 3.0 * _nrows), sharey=False)
    _axes_flat = _axes.flatten()

    # Summary: all sites overlaid
    _ax_sum = _axes_flat[0]
    _cmap_sites = plt.cm.get_cmap('tab20', _n_sites)
    for _si, (_, _row) in enumerate(_field_sites.iterrows()):
        _lbl = _row['label']
        _ax_sum.plot(_t_common, _site_prob[_lbl], color=_cmap_sites(_si), lw=1.0,
                     label=_lbl, alpha=0.8)
        if _lbl in _first_visit_t:
            _ax_sum.axvline(_first_visit_t[_lbl], color=_cmap_sites(_si), lw=0.8, ls='--', alpha=0.5)
    _ax_sum.set_title("All sites overlay")
    _ax_sum.set_xlabel("Time (min)")
    _ax_sum.set_ylabel("P(sheep at site)")
    _ax_sum.legend(fontsize=6, ncol=3, loc='upper right')
    _ax_sum.set_xlim(0, _dur)

    # Per-site subplots
    for _ai, (_, _row) in enumerate(_field_sites.iterrows()):
        _ax = _axes_flat[_ai + 1]
        _lbl = _row['label']
        _prob = _site_prob[_lbl]
        _ax.fill_between(_t_common, 0, _prob, alpha=0.3, color='#2171b5')
        _ax.plot(_t_common, _prob, color='#2171b5', lw=1.2)
        if _lbl in _first_visit_t:
            _ax.axvline(_first_visit_t[_lbl], color='#d63b3b', lw=1.5, ls='--', label='1st visit')
            _ax.legend(fontsize=7)
        _ax.set_title(f"{_lbl}  ({SITE_GRID[_lbl][0]},{SITE_GRID[_lbl][1]})")
        _ax.set_xlim(0, _dur)
        _ax.set_ylim(0, 1.05)
        _ax.set_ylabel("P(sheep near site)")
        _ax.set_xlabel("Time (min)")

    # Hide unused
    for _ai in range(_n_sites + 1, len(_axes_flat)):
        _axes_flat[_ai].set_visible(False)

    _fig.suptitle(
        f"Site occupancy probability — {_trial['name']}\n"
        f"radius={_radius} grid units, smooth={_smooth_s}s",
        fontsize=11, y=1.01,
    )
    _fig.tight_layout()
    plt.close(_fig)
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## Behaviour change around discovery events")
    return


@app.cell(hide_code=True)
def _(
    trial_selector, TRIALS, mo,
    load_gnss_date, load_trial_tracks, detect_site_visits,
    cumulative_path_length, ARENA_TRANSFORMS, DATA_DIR,
    np, pd, plt,
    radius_slider, window_slider,
):
    """Part 2: compare speed & spread before vs after each discovery event."""
    if trial_selector.value is None:
        mo.stop(True, mo.md(""))

    _tidx = trial_selector.value
    _trial = TRIALS[_tidx]
    _radius = radius_slider.value
    _win_s = window_slider.value
    _win_min = _win_s / 60.0

    _gnss = load_gnss_date(_trial['date'])
    _tracks = load_trial_tracks(
        _trial, gnss_cache={_trial['date']: _gnss},
        apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
    )
    if not _tracks:
        mo.stop(True, mo.md(""))

    _rdf = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
    _visits = detect_site_visits(_tracks, _trial['field'], radius=_radius, min_dwell_s=5.0, reward_sites_df=_rdf)
    _first_visit_t = {lbl: min(v[1] for v in vlist) for lbl, vlist in _visits.items() if vlist}

    if not _first_visit_t:
        mo.stop(True, mo.md("*No visit events detected; try a larger radius.*"))

    _dur = _trial['duration_min']
    _n_sheep = len(_tracks)
    _t_common = np.arange(0, _dur + 1 / 60, 1 / 60)

    # Interpolate sheep
    _gx_all, _gy_all = [], []
    for _sid, _trk in sorted(_tracks.items()):
        _order = np.argsort(_trk['t'])
        _gx_all.append(np.interp(_t_common, _trk['t'][_order], _trk['gx'][_order]))
        _gy_all.append(np.interp(_t_common, _trk['t'][_order], _trk['gy'][_order]))
    _gx_all = np.array(_gx_all)
    _gy_all = np.array(_gy_all)

    # Speed per sheep (m/min)
    _dt = 1 / 60.0
    _speed_all = np.sqrt(np.diff(_gx_all, axis=1)**2 + np.diff(_gy_all, axis=1)**2) / _dt * 10.0
    _speed_all = np.concatenate([_speed_all, _speed_all[:, -1:]], axis=1)
    _mean_speed = _speed_all.mean(axis=0)

    # Group spread (mean dist to centroid)
    _cx = _gx_all.mean(axis=0)
    _cy = _gy_all.mean(axis=0)
    _spread = np.sqrt((_gx_all - _cx)**2 + (_gy_all - _cy)**2).mean(axis=0)

    # For each discovery event: extract before/after windows
    _rows = []
    _win_steps = max(1, int(_win_min * 60))
    for _lbl, _t_disc in sorted(_first_visit_t.items(), key=lambda x: x[1]):
        _idx = int(_t_disc * 60)
        _lo_before = max(0, _idx - _win_steps)
        _hi_after = min(len(_t_common), _idx + _win_steps)

        _speed_before = float(_mean_speed[_lo_before:_idx].mean()) if _idx > _lo_before else np.nan
        _speed_after = float(_mean_speed[_idx:_hi_after].mean()) if _hi_after > _idx else np.nan
        _spread_before = float(_spread[_lo_before:_idx].mean()) if _idx > _lo_before else np.nan
        _spread_after = float(_spread[_idx:_hi_after].mean()) if _hi_after > _idx else np.nan

        _rows.append({
            'Site': _lbl,
            'Discovery (min)': round(_t_disc, 2),
            'Speed before (m/min)': round(_speed_before, 2) if not np.isnan(_speed_before) else None,
            'Speed after (m/min)': round(_speed_after, 2) if not np.isnan(_speed_after) else None,
            'Spread before (m)': round(_spread_before * 10, 2) if not np.isnan(_spread_before) else None,
            'Spread after (m)': round(_spread_after * 10, 2) if not np.isnan(_spread_after) else None,
            'Speed change (%)': round(100 * (_speed_after - _speed_before) / max(_speed_before, 0.01), 1)
                if not np.isnan(_speed_before) and not np.isnan(_speed_after) else None,
            'Spread change (%)': round(100 * (_spread_after - _spread_before) / max(_spread_before, 0.01), 1)
                if not np.isnan(_spread_before) and not np.isnan(_spread_after) else None,
        })

    _event_df = pd.DataFrame(_rows) if _rows else pd.DataFrame()

    # --- Plot: speed + spread + discovery events ---
    _fig2, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    _ax1.plot(_t_common, _mean_speed, color='#2171b5', lw=0.8, alpha=0.7)
    _kernel = np.ones(60) / 60
    _ax1.plot(_t_common, np.convolve(_mean_speed, _kernel, mode='same'),
              color='#08306b', lw=1.5, label='smoothed (1-min)')
    _ax1.set_ylabel("Mean group speed (m/min)")
    _ax1.set_title("Group speed with site discovery events")

    _ax2.plot(_t_common, _spread * 10, color='#d7191c', lw=0.8, alpha=0.7)
    _ax2.plot(_t_common, np.convolve(_spread * 10, _kernel, mode='same'),
              color='#7f0000', lw=1.5, label='smoothed')
    _ax2.set_ylabel("Mean group spread (m)")
    _ax2.set_xlabel("Time (min)")
    _ax2.set_title("Group spread over time")

    _cmap_ev = plt.cm.get_cmap('tab20', len(_first_visit_t))
    for _ei, (_lbl, _t_disc) in enumerate(sorted(_first_visit_t.items(), key=lambda x: x[1])):
        for _ax in [_ax1, _ax2]:
            _ax.axvline(_t_disc, color=_cmap_ev(_ei), lw=1.2, ls='--', alpha=0.8)
            _ax.text(_t_disc + 0.1, _ax.get_ylim()[1] * 0.95,
                     _lbl, fontsize=7, va='top', rotation=90, color=_cmap_ev(_ei))

    for _ax in [_ax1, _ax2]:
        _ax.set_xlim(0, _dur)

    _fig2.suptitle(
        f"Behaviour change around discoveries — {_trial['name']}\n"
        f"window={_win_s}s  |  {len(_first_visit_t)} sites found",
        fontsize=11,
    )
    _fig2.tight_layout()
    plt.close(_fig2)

    mo.vstack([
        mo.mpl.interactive(_fig2),
        mo.md(f"### Before/after comparison (window = {_win_s} s each side)"),
        mo.ui.table(_event_df) if not _event_df.empty else mo.md("*No data.*"),
    ])
    return


if __name__ == "__main__":
    app.run()
