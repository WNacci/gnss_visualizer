"""Leader-Follower Dynamics

Identifies leadership behaviour within a group across time by measuring:

  1. **Frontal position**: which sheep is farthest ahead (in the direction of
     group motion) at each timestep.  The "leader" leads from the front.
  2. **Pioneer visits**: which sheep is the *first* to enter the vicinity of
     each reward site.  Consistent pioneers are candidates for leaders.
  3. **Leadership consistency**: entropy of the leadership distribution.
     Low entropy → one sheep leads most of the time.

Todos addressed: #6 (leader/follower dynamics)
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
        build_trials, build_gps_cache, load_gnss_date, build_arena_transforms,
        load_trial_tracks, detect_site_visits, DATA_DIR,
    )

    TRIALS = build_trials()
    ARENA_TRANSFORMS = build_arena_transforms()
    print(f"Loaded {len(TRIALS)} trials — building GPS cache (runs once)…")
    GPS_CACHE = build_gps_cache(TRIALS)
    return (
        np, pd, plt, mo, matplotlib,
        build_trials, load_gnss_date, build_arena_transforms,
        load_trial_tracks, detect_site_visits, DATA_DIR,
        TRIALS, ARENA_TRANSFORMS, GPS_CACHE,
    )


@app.cell(hide_code=True)
def _(TRIALS, mo):
    _options = {}
    for _i, _t in enumerate(TRIALS):
        if _t['group_size'] < 2:
            continue
        _assay_str = f" [Assay {_t['assay']}]" if _t['assay'] is not None else ""
        _label = f"[{_i:3d}] {_t['notes'].split(' - ')[0]:20s} {_t['name']}{_assay_str}"
        _options[_label] = _i

    trial_selector = mo.ui.dropdown(options=_options, label="Select trial (≥2 sheep)")
    radius_slider = mo.ui.slider(
        start=0.1, stop=2.0, step=0.1, value=0.5,
        label="Pioneer detection radius (grid units)",
    )
    smooth_slider = mo.ui.slider(
        start=1, stop=60, step=1, value=15,
        label="Smoothing window (s) for frontal position",
    )

    mo.md(f"""
    # Leader-Follower Dynamics

    Two complementary metrics:
    - **Frontal position leadership**: at each timestep, which sheep is farthest
      ahead of the group centroid in the direction of travel?
    - **Pioneer leadership**: which sheep *first* enters each reward site?

    {trial_selector}
    {mo.hstack([radius_slider, smooth_slider])}
    """)
    return trial_selector, radius_slider, smooth_slider


@app.cell(hide_code=True)
def _(
    trial_selector, TRIALS, mo,
    GPS_CACHE, load_trial_tracks, detect_site_visits,
    ARENA_TRANSFORMS, DATA_DIR,
    np, pd, plt,
    radius_slider, smooth_slider,
):
    if trial_selector.value is None:
        mo.stop(True, mo.md("*Select a trial above.*"))

    _tidx = trial_selector.value
    _trial = TRIALS[_tidx]
    _radius = radius_slider.value
    _smooth_s = smooth_slider.value

    _tracks = load_trial_tracks(
        _trial, gnss_cache=GPS_CACHE,
        apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
    )

    if len(_tracks) < 2:
        mo.stop(True, mo.md("*Need ≥2 sheep with GPS data.*"))

    _sheep_ids = sorted(_tracks.keys())
    _n_sheep = len(_sheep_ids)
    _dur = _trial['duration_min']

    # Common time grid (1/min resolution = 1 s)
    _t_common = np.arange(0, _dur + 1 / 60, 1 / 60)
    _n_t = len(_t_common)

    _gx_all = np.zeros((_n_sheep, _n_t))
    _gy_all = np.zeros((_n_sheep, _n_t))
    for _ci, _sid in enumerate(_sheep_ids):
        _trk = _tracks[_sid]
        _order = np.argsort(_trk['t'])
        _gx_all[_ci] = np.interp(_t_common, _trk['t'][_order], _trk['gx'][_order])
        _gy_all[_ci] = np.interp(_t_common, _trk['t'][_order], _trk['gy'][_order])

    # -----------------------------------------------------------------------
    # Metric 1: Frontal position leadership
    # Leader = sheep with largest projection onto group velocity vector
    # -----------------------------------------------------------------------
    _cx = _gx_all.mean(axis=0)
    _cy = _gy_all.mean(axis=0)

    # Group velocity (smoothed)
    _smooth_w = max(1, int(_smooth_s))
    _kernel = np.ones(_smooth_w) / _smooth_w
    _vcx = np.convolve(np.gradient(_cx, _t_common), _kernel, mode='same')
    _vcy = np.convolve(np.gradient(_cy, _t_common), _kernel, mode='same')
    _v_speed = np.sqrt(_vcx**2 + _vcy**2)
    # Normalise velocity vector (avoid div-by-zero)
    _v_speed_safe = np.where(_v_speed > 1e-6, _v_speed, 1.0)
    _vnx = _vcx / _v_speed_safe
    _vny = _vcy / _v_speed_safe

    # Displacement from centroid
    _dx = _gx_all - _cx  # (n_sheep, n_time)
    _dy = _gy_all - _cy

    # Projection of each sheep's displacement onto velocity direction
    _proj = _dx * _vnx + _dy * _vny  # (n_sheep, n_time)

    # Leader at each timestep = sheep with max forward projection
    # Only count when group is actually moving (speed > threshold)
    _moving = _v_speed > 0.02  # grid units / min ≈ 0.2 m/min
    _leader_idx = np.argmax(_proj, axis=0)  # (n_time,)
    _leader_idx_moving = _leader_idx.copy()
    _leader_idx_moving[~_moving] = -1

    # Leadership fraction per sheep
    _leadership_counts = np.bincount(
        _leader_idx_moving[_leader_idx_moving >= 0], minlength=_n_sheep
    )
    _moving_total = (_leader_idx_moving >= 0).sum()
    _leadership_frac = _leadership_counts / max(_moving_total, 1)

    # Shannon entropy of leadership distribution
    _p = _leadership_frac[_leadership_frac > 0]
    _entropy = float(-np.sum(_p * np.log(_p))) if len(_p) > 0 else 0.0
    _max_entropy = np.log(_n_sheep) if _n_sheep > 1 else 1.0
    _norm_entropy = _entropy / _max_entropy  # 0=one leader, 1=fully distributed

    # -----------------------------------------------------------------------
    # Metric 2: Pioneer visits
    # -----------------------------------------------------------------------
    _rdf = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
    _visits = detect_site_visits(
        _tracks, _trial['field'], radius=_radius, min_dwell_s=5.0,
        reward_sites_df=_rdf,
    )

    _pioneer_counts = {_sid: 0 for _sid in _sheep_ids}
    _pioneer_rows = []
    for _lbl, _vlist in sorted(_visits.items()):
        if not _vlist:
            continue
        _first = min(_vlist, key=lambda x: x[1])
        _pioneer_counts[_first[0]] = _pioneer_counts.get(_first[0], 0) + 1
        _pioneer_rows.append({
            'Site': _lbl,
            'Pioneer sheep': _first[0],
            'Entry time (min)': round(_first[1], 2),
        })

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    COLORS = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#a65628', '#f781bf', '#999999',
    ]

    _fig, _axes = plt.subplots(2, 2, figsize=(14, 8))
    _ax_traj, _ax_lead_ts, _ax_frac, _ax_pioneer = _axes.flatten()

    # --- Trajectories coloured by animal ---
    for _ci, _sid in enumerate(_sheep_ids):
        _ax_traj.plot(_gx_all[_ci], _gy_all[_ci], color=COLORS[_ci % len(COLORS)],
                      alpha=0.4, lw=0.8, label=_sid)
        _ax_traj.scatter(
            _gx_all[_ci, 0], _gy_all[_ci, 0],
            color=COLORS[_ci % len(COLORS)], s=60, zorder=5, marker='o',
        )
    _ax_traj.plot(_cx, _cy, color='k', lw=1, ls='--', label='centroid')
    _ax_traj.set_xlabel("Grid x")
    _ax_traj.set_ylabel("Grid y")
    _ax_traj.set_title("Trajectories (circle=start)")
    _ax_traj.legend(fontsize=8)
    _ax_traj.set_aspect('equal')

    # --- Leader time series (smoothed with 60-s window for readability) ---
    _window = max(1, int(60))
    _lead_smooth = np.convolve(
        _leader_idx_moving.astype(float), np.ones(_window) / _window, mode='same'
    )
    # Plot fractional leader ID per time window
    _bin_w = 60  # 1 min bins
    _t_bin_edges = np.arange(0, _dur + 1, 1)
    _t_bin_c = 0.5 * (_t_bin_edges[:-1] + _t_bin_edges[1:])
    _lead_by_bin = np.zeros((_n_sheep, len(_t_bin_c)))
    for _bi, _tb in enumerate(_t_bin_c):
        _mask = (_t_common >= _t_bin_edges[_bi]) & (_t_common < _t_bin_edges[_bi + 1])
        _valid = _leader_idx_moving[_mask]
        _valid = _valid[_valid >= 0]
        if len(_valid) > 0:
            _bc = np.bincount(_valid, minlength=_n_sheep) / len(_valid)
            _lead_by_bin[:, _bi] = _bc

    _bottom = np.zeros(len(_t_bin_c))
    for _ci, _sid in enumerate(_sheep_ids):
        _ax_lead_ts.bar(
            _t_bin_c, _lead_by_bin[_ci], width=0.95, bottom=_bottom,
            color=COLORS[_ci % len(COLORS)], label=_sid, alpha=0.85,
        )
        _bottom += _lead_by_bin[_ci]
    _ax_lead_ts.set_xlabel("Time (min)")
    _ax_lead_ts.set_ylabel("Fraction of time as frontal leader")
    _ax_lead_ts.set_title("Frontal leadership over time (1-min bins)")
    _ax_lead_ts.set_xlim(0, _dur)
    _ax_lead_ts.legend(fontsize=8, loc='upper right')

    # --- Overall leadership fraction bar chart ---
    _ax_frac.bar(
        range(_n_sheep), _leadership_frac,
        color=[COLORS[_i % len(COLORS)] for _i in range(_n_sheep)],
        tick_label=_sheep_ids,
    )
    _ax_frac.set_ylabel("Fraction of moving time as frontal leader")
    _ax_frac.set_title(
        f"Leadership fraction\nnorm. entropy={_norm_entropy:.2f} "
        f"(0=1 leader, 1=equal)"
    )
    _ax_frac.set_ylim(0, 1)

    # --- Pioneer counts ---
    _pioneer_labels = list(_pioneer_counts.keys())
    _pioneer_vals = [_pioneer_counts[k] for k in _pioneer_labels]
    _ax_pioneer.bar(
        range(_n_sheep), _pioneer_vals,
        color=[COLORS[_sheep_ids.index(k) % len(COLORS)] for k in _pioneer_labels],
        tick_label=_pioneer_labels,
    )
    _ax_pioneer.set_ylabel("# sites where sheep was first to enter")
    _ax_pioneer.set_title("Pioneer visits per sheep")

    _fig.suptitle(f"Leader-follower dynamics — {_trial['name']}", fontsize=11)
    _fig.tight_layout()

    _pioneer_df = pd.DataFrame(_pioneer_rows) if _pioneer_rows else pd.DataFrame(
        columns=['Site', 'Pioneer sheep', 'Entry time (min)']
    )

    mo.vstack([
        _fig,
        mo.md(
            f"**Normalised leadership entropy:** {_norm_entropy:.3f}  "
            f"(0 = one sheep always leads; 1 = leadership equally shared)  \n"
            f"**Pioneer visits:** {sum(_pioneer_vals)} sites visited"
        ),
        mo.md("### Pioneer visit events"),
        mo.ui.table(_pioneer_df),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## Aggregate: leadership consistency across trials by assay")
    return


@app.cell(hide_code=True)
def _(
    TRIALS, mo,
    GPS_CACHE, load_trial_tracks, ARENA_TRANSFORMS,
    np, pd, plt,
):
    """Compute leadership entropy for every multi-sheep test trial."""
    _TEST_CONFIGS = {'A', 'B', 'C', 'D'}
    _records = []

    for _tidx, _trial in enumerate(TRIALS):
        if _trial['group_size'] < 2:
            continue
        if _trial['config'] not in _TEST_CONFIGS:
            continue

        _tracks = load_trial_tracks(
            _trial, gnss_cache=GPS_CACHE,
            apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
        )
        if len(_tracks) < 2:
            continue

        _sheep_ids = sorted(_tracks.keys())
        _n_sheep = len(_sheep_ids)
        _dur = _trial['duration_min']
        _t_common = np.arange(0, _dur + 1 / 60, 1 / 60)
        _n_t = len(_t_common)

        _gx_all = np.zeros((_n_sheep, _n_t))
        _gy_all = np.zeros((_n_sheep, _n_t))
        for _ci, _sid in enumerate(_sheep_ids):
            _trk = _tracks[_sid]
            _order = np.argsort(_trk['t'])
            _gx_all[_ci] = np.interp(_t_common, _trk['t'][_order], _trk['gx'][_order])
            _gy_all[_ci] = np.interp(_t_common, _trk['t'][_order], _trk['gy'][_order])

        _cx = _gx_all.mean(axis=0)
        _cy = _gy_all.mean(axis=0)
        _vcx = np.gradient(_cx, _t_common)
        _vcy = np.gradient(_cy, _t_common)
        _v_speed = np.sqrt(_vcx**2 + _vcy**2)
        _v_speed_safe = np.where(_v_speed > 1e-6, _v_speed, 1.0)
        _vnx = _vcx / _v_speed_safe
        _vny = _vcy / _v_speed_safe
        _dx = _gx_all - _cx
        _dy = _gy_all - _cy
        _proj = _dx * _vnx + _dy * _vny
        _moving = _v_speed > 0.02
        _leader_idx = np.argmax(_proj, axis=0)
        _leader_idx[~_moving] = -1
        _lc = np.bincount(_leader_idx[_leader_idx >= 0], minlength=_n_sheep)
        _mt = (_leader_idx >= 0).sum()
        _lf = _lc / max(_mt, 1)
        _p = _lf[_lf > 0]
        _entropy = float(-np.sum(_p * np.log(_p))) if len(_p) > 0 else 0.0
        _norm = _entropy / np.log(_n_sheep) if _n_sheep > 1 else 0.0

        _records.append({
            'Trial': _tidx,
            'Date': _trial['date'],
            'Config': _trial['config'],
            'Group size': _trial['group_size'],
            'Assay': str(_trial['assay']),
            'Norm. leadership entropy': round(_norm, 3),
            'Dominant leader fraction': round(float(_lf.max()), 3),
        })

    if not _records:
        mo.stop(True, mo.md("*No multi-sheep test trials with GPS data found.*"))

    _agg_df = pd.DataFrame(_records)

    _fig2, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _assays = sorted(_agg_df['Assay'].unique(), key=lambda x: (not x.isdigit(), x))

    _ent_by_assay = [_agg_df[_agg_df['Assay'] == a]['Norm. leadership entropy'].dropna().values
                     for a in _assays]
    _dom_by_assay = [_agg_df[_agg_df['Assay'] == a]['Dominant leader fraction'].dropna().values
                     for a in _assays]

    _ax1.boxplot(_ent_by_assay, labels=_assays)
    _ax1.set_xlabel("Assay")
    _ax1.set_ylabel("Normalised leadership entropy")
    _ax1.set_title("Leadership consistency by assay\n(low=one sheep leads)")
    _ax1.set_ylim(0, 1.05)

    _ax2.boxplot(_dom_by_assay, labels=_assays)
    _ax2.set_xlabel("Assay")
    _ax2.set_ylabel("Dominant leader fraction")
    _ax2.set_title("Fraction of time dominant sheep leads")
    _ax2.set_ylim(0, 1.05)

    _fig2.suptitle("Leadership consistency across test trials")
    _fig2.tight_layout()

    mo.vstack([
        _fig2,
        mo.md("### Per-trial leadership table"),
        mo.ui.table(_agg_df),
    ])
    return


if __name__ == "__main__":
    app.run()
