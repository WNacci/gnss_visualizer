"""Flocking & Group Cohesion Analysis

Computes pairwise inter-animal distances over time to quantify how tightly
the group moves together throughout a trial.  Reports:
  - Mean and max inter-animal distance over time
  - Nearest-neighbour distance (NND) over time
  - Group spread (distance of each sheep from the group centroid)
  - Aggregate view: mean NND across trials binned by assay / group size

Todos addressed: #10 (grouping across time, consistency of flocking)
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
        load_trial_tracks,
    )

    TRIALS = build_trials()
    ARENA_TRANSFORMS = build_arena_transforms()
    print(f"Loaded {len(TRIALS)} trials")
    return (
        np, pd, plt, mo, matplotlib,
        build_trials, load_gnss_date, build_arena_transforms,
        load_trial_tracks, TRIALS, ARENA_TRANSFORMS,
    )


@app.cell(hide_code=True)
def _(TRIALS, mo):
    _options = {}
    for _i, _t in enumerate(TRIALS):
        if _t['group_size'] < 2:
            continue  # skip solitary trials for flocking
        _assay_str = f" [Assay {_t['assay']}]" if _t['assay'] is not None else ""
        _label = f"[{_i:3d}] {_t['notes'].split(' - ')[0]:20s} {_t['name']}{_assay_str}"
        _options[_label] = _i

    trial_selector = mo.ui.dropdown(options=_options, label="Select trial (≥2 sheep)")
    bin_slider = mo.ui.slider(start=5, stop=120, step=5, value=30, label="Time bin (s)")

    mo.md(f"""
    # Flocking & Group Cohesion

    Requires a trial with ≥2 sheep.  Distances are in grid units (1 unit ≈ 10 m).

    {trial_selector}

    {bin_slider}
    """)
    return trial_selector, bin_slider


@app.cell(hide_code=True)
def _(
    trial_selector, TRIALS, mo,
    load_gnss_date, load_trial_tracks, ARENA_TRANSFORMS,
    np, pd, plt, bin_slider,
):
    if trial_selector.value is None:
        mo.stop(True, mo.md("*Select a trial above.*"))

    _tidx = trial_selector.value
    _trial = TRIALS[_tidx]
    _bin_s = bin_slider.value
    _bin_min = _bin_s / 60.0

    _gnss = load_gnss_date(_trial['date'])
    _tracks = load_trial_tracks(
        _trial, gnss_cache={_trial['date']: _gnss},
        apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
    )

    if len(_tracks) < 2:
        mo.stop(True, mo.md("*Need ≥2 sheep with GPS data.*"))

    _sheep_ids = sorted(_tracks.keys())
    _dur = _trial['duration_min']

    # Interpolate all sheep onto a common 1-second grid
    _t_common = np.arange(0, _dur + 1 / 60, 1 / 60)
    _gx_all, _gy_all = [], []
    for _sid in _sheep_ids:
        _trk = _tracks[_sid]
        _order = np.argsort(_trk['t'])
        _t_s = _trk['t'][_order]
        _gx_i = np.interp(_t_common, _t_s, _trk['gx'][_order])
        _gy_i = np.interp(_t_common, _t_s, _trk['gy'][_order])
        _gx_all.append(_gx_i)
        _gy_all.append(_gy_i)

    _gx_all = np.array(_gx_all)   # shape: (n_sheep, n_time)
    _gy_all = np.array(_gy_all)

    _n_sheep = len(_sheep_ids)
    _n_time = len(_t_common)

    # Group centroid
    _cx = _gx_all.mean(axis=0)
    _cy = _gy_all.mean(axis=0)

    # Distance to centroid per sheep
    _dist_to_centroid = np.sqrt((_gx_all - _cx)**2 + (_gy_all - _cy)**2)  # (n_sheep, n_time)
    _spread = _dist_to_centroid.mean(axis=0)  # mean spread

    # Pairwise distances
    _pw = []
    for _i in range(_n_sheep):
        for _j in range(_i + 1, _n_sheep):
            _d = np.sqrt(
                (_gx_all[_i] - _gx_all[_j])**2 +
                (_gy_all[_i] - _gy_all[_j])**2
            )
            _pw.append(_d)
    _pw = np.array(_pw)  # shape: (n_pairs, n_time)

    _mean_pw = _pw.mean(axis=0)
    _max_pw = _pw.max(axis=0)
    _nnd = _pw.min(axis=0)  # nearest-neighbour distance

    # Bin into time windows
    _t_edges = np.arange(0, _dur + _bin_min, _bin_min)
    _t_centres = 0.5 * (_t_edges[:-1] + _t_edges[1:])

    def _bin(arr):
        return np.array([
            arr[(_t_common >= _t_edges[_bi]) & (_t_common < _t_edges[_bi + 1])].mean()
            if ((_t_common >= _t_edges[_bi]) & (_t_common < _t_edges[_bi + 1])).sum() > 0
            else np.nan
            for _bi in range(len(_t_centres))
        ])

    _mean_pw_b = _bin(_mean_pw)
    _max_pw_b = _bin(_max_pw)
    _nnd_b = _bin(_nnd)
    _spread_b = _bin(_spread)

    # --- Plot ---
    COLORS_SHEEP = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#a65628', '#f781bf', '#999999',
    ]

    _fig, _axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    _ax_mean, _ax_max, _ax_nnd, _ax_spread = _axes.flatten()

    _ax_mean.plot(_t_centres, _mean_pw_b * 10, color='#2171b5', lw=1.5)
    _ax_mean.fill_between(_t_centres, 0, _mean_pw_b * 10, alpha=0.2, color='#2171b5')
    _ax_mean.set_ylabel("Mean pairwise dist. (m)")
    _ax_mean.set_title("Mean inter-animal distance")

    _ax_max.plot(_t_centres, _max_pw_b * 10, color='#d7191c', lw=1.5)
    _ax_max.fill_between(_t_centres, 0, _max_pw_b * 10, alpha=0.2, color='#d7191c')
    _ax_max.set_ylabel("Max pairwise dist. (m)")
    _ax_max.set_title("Maximum inter-animal distance")

    _ax_nnd.plot(_t_centres, _nnd_b * 10, color='#1a9641', lw=1.5)
    _ax_nnd.fill_between(_t_centres, 0, _nnd_b * 10, alpha=0.2, color='#1a9641')
    _ax_nnd.set_ylabel("Nearest-neighbour dist. (m)")
    _ax_nnd.set_title("Nearest-neighbour distance")
    _ax_nnd.set_xlabel("Time (min)")

    for _ci, _sid in enumerate(_sheep_ids):
        _d_to_c = _dist_to_centroid[_ci]
        _d_b = _bin(_d_to_c)
        _ax_spread.plot(_t_centres, _d_b * 10, color=COLORS_SHEEP[_ci % len(COLORS_SHEEP)],
                        lw=1.2, label=_sid)
    _ax_spread.set_ylabel("Dist. to centroid (m)")
    _ax_spread.set_title("Spread: each sheep from centroid")
    _ax_spread.set_xlabel("Time (min)")
    _ax_spread.legend(loc='upper right', fontsize=8)

    for _ax in _axes.flatten():
        _ax.set_xlim(0, _dur)
        _ax.set_ylim(bottom=0)

    _fig.suptitle(
        f"Flocking dynamics — {_trial['name']}\n"
        f"{_n_sheep} sheep  |  "
        f"mean NND overall: {float(_nnd.mean()*10):.1f} m, "
        f"mean group spread: {float(_spread.mean()*10):.1f} m",
        fontsize=11,
    )
    _fig.tight_layout()
    plt.close(_fig)

    # Summary table
    _summary = pd.DataFrame({
        'Metric': [
            'Mean inter-animal distance (m)',
            'Max inter-animal distance (m)',
            'Mean nearest-neighbour distance (m)',
            'Mean group spread (dist to centroid, m)',
        ],
        'Value': [
            round(float(_mean_pw.mean() * 10), 2),
            round(float(_max_pw.mean() * 10), 2),
            round(float(_nnd.mean() * 10), 2),
            round(float(_spread.mean() * 10), 2),
        ],
    })

    mo.vstack([
        mo.mpl.interactive(_fig),
        mo.md("### Summary statistics"),
        mo.ui.table(_summary),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## Aggregate: group cohesion across all multi-sheep trials")
    return


@app.cell(hide_code=True)
def _(
    TRIALS, mo,
    load_gnss_date, load_trial_tracks, ARENA_TRANSFORMS,
    np, pd, plt,
):
    """Compute mean NND for each multi-sheep trial; plot by assay and group size."""
    _TEST_CONFIGS = {'A', 'B', 'C', 'D'}
    _records = []

    for _tidx, _trial in enumerate(TRIALS):
        if _trial['group_size'] < 2:
            continue
        if _trial['config'] not in _TEST_CONFIGS:
            continue

        _gnss = load_gnss_date(_trial['date'])
        _tracks = load_trial_tracks(
            _trial, gnss_cache={_trial['date']: _gnss},
            apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
        )
        if len(_tracks) < 2:
            continue

        _sheep_ids = sorted(_tracks.keys())
        _dur = _trial['duration_min']
        _t_common = np.arange(0, _dur + 1 / 60, 1 / 60)

        _gx_all, _gy_all = [], []
        for _sid in _sheep_ids:
            _trk = _tracks[_sid]
            _order = np.argsort(_trk['t'])
            _gx_all.append(np.interp(_t_common, _trk['t'][_order], _trk['gx'][_order]))
            _gy_all.append(np.interp(_t_common, _trk['t'][_order], _trk['gy'][_order]))
        _gx_all = np.array(_gx_all)
        _gy_all = np.array(_gy_all)
        _n_sheep = len(_sheep_ids)

        # NND
        _pw = []
        for _i in range(_n_sheep):
            for _j in range(_i + 1, _n_sheep):
                _d = np.sqrt(
                    (_gx_all[_i] - _gx_all[_j])**2 +
                    (_gy_all[_i] - _gy_all[_j])**2
                )
                _pw.append(_d)
        _pw = np.array(_pw)
        _nnd = _pw.min(axis=0) if _pw.ndim == 2 else _pw
        _cx = _gx_all.mean(axis=0)
        _cy = _gy_all.mean(axis=0)
        _spread = np.sqrt((_gx_all - _cx)**2 + (_gy_all - _cy)**2).mean(axis=0)

        _records.append({
            'Trial': _tidx,
            'Date': _trial['date'],
            'Config': _trial['config'],
            'Group size': _trial['group_size'],
            'Assay': str(_trial['assay']),
            'Mean NND (m)': round(float(_nnd.mean() * 10), 2),
            'Mean spread (m)': round(float(_spread.mean() * 10), 2),
        })

    if not _records:
        mo.stop(True, mo.md("*No multi-sheep test trials with GPS data found.*"))

    _agg_df = pd.DataFrame(_records)

    _fig2, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))

    _assays = sorted(_agg_df['Assay'].unique(), key=lambda x: (not x.isdigit(), x))
    _nnd_by_assay = [_agg_df[_agg_df['Assay'] == a]['Mean NND (m)'].dropna().values for a in _assays]
    _sp_by_assay = [_agg_df[_agg_df['Assay'] == a]['Mean spread (m)'].dropna().values for a in _assays]

    _ax1.boxplot(_nnd_by_assay, labels=_assays)
    _ax1.set_xlabel("Assay")
    _ax1.set_ylabel("Mean NND (m)")
    _ax1.set_title("Nearest-neighbour distance by assay")

    _ax2.boxplot(_sp_by_assay, labels=_assays)
    _ax2.set_xlabel("Assay")
    _ax2.set_ylabel("Mean spread (m)")
    _ax2.set_title("Group spread by assay")

    _fig2.suptitle("Group cohesion across test trials (multi-sheep)")
    _fig2.tight_layout()
    plt.close(_fig2)

    mo.vstack([
        mo.mpl.interactive(_fig2),
        mo.md("### Per-trial cohesion table"),
        mo.ui.table(_agg_df),
    ])
    return


if __name__ == "__main__":
    app.run()
