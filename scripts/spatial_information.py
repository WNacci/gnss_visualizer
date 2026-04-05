"""Spatial Information & Revisit Analysis

Two analyses in one notebook:

1. **Spatial entropy over time** (Todo #5): measures how spread-out the
   sheep's distribution is across the arena at each timestep.  High entropy
   = exploratory; low entropy = clustered around a few locations.

2. **Self-avoiding walk / revisit rate** (Todo #7): quantifies how often
   sheep return to already-visited grid cells.  Low revisit rate ↔ more
   systematic exploration.  Tracks unique cells visited over time and the
   fraction of time spent in previously visited cells.
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
        _assay_str = f" [Assay {_t['assay']}]" if _t['assay'] is not None else ""
        _label = f"[{_i:3d}] {_t['notes'].split(' - ')[0]:20s} {_t['name']}{_assay_str}"
        _options[_label] = _i

    trial_selector = mo.ui.dropdown(options=_options, label="Select trial")
    grid_res_slider = mo.ui.slider(
        start=5, stop=50, step=5, value=20,
        label="Grid resolution (cells per side, 5×5 arena)",
    )
    window_slider = mo.ui.slider(
        start=10, stop=300, step=10, value=60,
        label="Entropy window (seconds)",
    )

    mo.md(f"""
    # Spatial Information & Revisit Analysis

    **Spatial entropy**: computed in a sliding window over discretised arena cells.
    Shannon entropy H of the occupancy distribution; high = spread-out.

    **Revisit rate**: fraction of timesteps where a sheep occupies a cell it has
    visited before.  Decreases over time for efficient explorers.

    {trial_selector}
    {mo.hstack([grid_res_slider, window_slider])}
    """)
    return trial_selector, grid_res_slider, window_slider


@app.cell(hide_code=True)
def _(
    trial_selector, TRIALS, mo,
    load_gnss_date, load_trial_tracks, ARENA_TRANSFORMS,
    np, pd, plt,
    grid_res_slider, window_slider,
):
    if trial_selector.value is None:
        mo.stop(True, mo.md("*Select a trial above.*"))

    _tidx = trial_selector.value
    _trial = TRIALS[_tidx]
    _res = grid_res_slider.value          # cells per side
    _win_s = window_slider.value          # seconds
    _win_min = _win_s / 60.0

    _gnss = load_gnss_date(_trial['date'])
    _tracks = load_trial_tracks(
        _trial, gnss_cache={_trial['date']: _gnss},
        apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
    )
    if not _tracks:
        mo.stop(True, mo.md("*No GPS data found.*"))

    _dur = _trial['duration_min']
    _sheep_ids = sorted(_tracks.keys())
    _n_sheep = len(_sheep_ids)

    # Discretise arena into res×res grid (arena 0-5 each axis)
    def _to_cell(gx, gy):
        """Map grid coords to cell indices."""
        cx = np.clip((gx / 5.0 * _res).astype(int), 0, _res - 1)
        cy = np.clip((gy / 5.0 * _res).astype(int), 0, _res - 1)
        return cx * _res + cy  # flat cell index

    # Common 1-s time grid
    _t_common = np.arange(0, _dur + 1 / 60, 1 / 60)
    _n_t = len(_t_common)

    # Per-sheep: cell index over time, cumulative unique cells, revisit flag
    _cells_all = np.zeros((_n_sheep, _n_t), dtype=int)
    for _ci, _sid in enumerate(_sheep_ids):
        _trk = _tracks[_sid]
        _order = np.argsort(_trk['t'])
        _gxi = np.interp(_t_common, _trk['t'][_order], _trk['gx'][_order])
        _gyi = np.interp(_t_common, _trk['t'][_order], _trk['gy'][_order])
        _cells_all[_ci] = _to_cell(_gxi, _gyi)

    # ------------------------------------------------------------------
    # Spatial entropy (sliding window, aggregate over all sheep)
    # ------------------------------------------------------------------
    _win_steps = max(1, int(_win_min * 60))   # window in time-steps
    _n_cells = _res * _res
    _entropy = np.zeros(_n_t)

    for _ti in range(_n_t):
        _lo = max(0, _ti - _win_steps // 2)
        _hi = min(_n_t, _lo + _win_steps)
        _window_cells = _cells_all[:, _lo:_hi].ravel()
        _counts = np.bincount(_window_cells, minlength=_n_cells).astype(float)
        _total = _counts.sum()
        if _total > 0:
            _p = _counts[_counts > 0] / _total
            _entropy[_ti] = float(-np.sum(_p * np.log2(_p)))

    _max_entropy = np.log2(_n_cells)
    _entropy_norm = _entropy / _max_entropy  # 0–1

    # ------------------------------------------------------------------
    # Revisit rate per sheep
    # ------------------------------------------------------------------
    _revisit_all = []
    _unique_all = []
    for _ci in range(_n_sheep):
        _cells = _cells_all[_ci]
        _visited_so_far = set()
        _revisit = np.zeros(_n_t, dtype=bool)
        _n_unique = np.zeros(_n_t, dtype=int)
        for _ti, _c in enumerate(_cells):
            if _c in _visited_so_far:
                _revisit[_ti] = True
            _visited_so_far.add(_c)
            _n_unique[_ti] = len(_visited_so_far)
        _revisit_all.append(_revisit)
        _unique_all.append(_n_unique)

    _revisit_all = np.array(_revisit_all)  # (n_sheep, n_t)
    _unique_all = np.array(_unique_all)

    # Binned revisit rate
    _bin_min = 1.0
    _t_bin_edges = np.arange(0, _dur + _bin_min, _bin_min)
    _t_bin_c = 0.5 * (_t_bin_edges[:-1] + _t_bin_edges[1:])

    def _bin_mean(arr, t_common, t_edges):
        out = []
        for _b in range(len(t_edges) - 1):
            mask = (t_common >= t_edges[_b]) & (t_common < t_edges[_b + 1])
            out.append(arr[..., mask].mean() if mask.sum() > 0 else np.nan)
        return np.array(out)

    _revisit_rate_b = _bin_mean(_revisit_all.mean(axis=0), _t_common, _t_bin_edges)
    _entropy_b = _bin_mean(_entropy_norm, _t_common, _t_bin_edges)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    COLORS = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#a65628', '#f781bf', '#999999',
    ]

    _fig, _axes = plt.subplots(2, 2, figsize=(13, 8))
    _ax_ent, _ax_rev, _ax_uniq, _ax_heatmap = _axes.flatten()

    # Entropy over time
    _ax_ent.plot(_t_bin_c, _entropy_b, color='#2171b5', lw=1.5)
    _ax_ent.fill_between(_t_bin_c, 0, _entropy_b, alpha=0.2, color='#2171b5')
    _ax_ent.set_ylabel("Normalised spatial entropy (0–1)")
    _ax_ent.set_title(f"Spatial entropy (window={_win_s}s, grid={_res}×{_res})")
    _ax_ent.set_ylim(0, 1.05)
    _ax_ent.set_xlabel("Time (min)")

    # Revisit rate over time
    _ax_rev.plot(_t_bin_c, _revisit_rate_b, color='#d7191c', lw=1.5)
    _ax_rev.fill_between(_t_bin_c, 0, _revisit_rate_b, alpha=0.2, color='#d7191c')
    _ax_rev.set_ylabel("Revisit rate (fraction of timesteps)")
    _ax_rev.set_title("Revisit rate over time")
    _ax_rev.set_ylim(0, 1.05)
    _ax_rev.set_xlabel("Time (min)")

    # Unique cells visited over time per sheep
    _cell_frac = _res * _res  # total cells
    for _ci, _sid in enumerate(_sheep_ids):
        _ax_uniq.plot(
            _t_common, _unique_all[_ci] / _cell_frac,
            color=COLORS[_ci % len(COLORS)], lw=1.2, label=_sid,
        )
    _ax_uniq.set_ylabel("Fraction of arena cells visited")
    _ax_uniq.set_title("Cumulative unique cells explored")
    _ax_uniq.set_xlabel("Time (min)")
    _ax_uniq.set_ylim(0, 1.05)
    _ax_uniq.legend(fontsize=8)

    # Overall occupancy heatmap
    _all_cells = _cells_all.ravel()
    _H_flat = np.bincount(_all_cells, minlength=_n_cells)
    _H = _H_flat.reshape((_res, _res))
    _ax_heatmap.imshow(
        _H.T, origin='lower', extent=[0, 5, 0, 5],
        cmap='Blues', aspect='equal', interpolation='nearest',
    )
    _ax_heatmap.set_xlabel("Grid x")
    _ax_heatmap.set_ylabel("Grid y")
    _ax_heatmap.set_title(f"Occupancy heatmap ({_res}×{_res} grid)")

    for _ax in [_ax_ent, _ax_rev, _ax_uniq]:
        _ax.set_xlim(0, _dur)

    # Summary stats
    _mean_final_cov = float(_unique_all[:, -1].mean()) / _cell_frac
    _mean_revisit = float(_revisit_all.mean())
    _mean_entropy = float(np.nanmean(_entropy_norm))

    _fig.suptitle(
        f"Spatial info — {_trial['name']}\n"
        f"Mean coverage: {_mean_final_cov:.1%}  |  "
        f"Mean revisit rate: {_mean_revisit:.1%}  |  "
        f"Mean entropy: {_mean_entropy:.3f}",
        fontsize=11,
    )
    _fig.tight_layout()
    plt.close(_fig)
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## Aggregate: spatial entropy and revisit rate by assay")
    return


@app.cell(hide_code=True)
def _(
    TRIALS, mo,
    load_gnss_date, load_trial_tracks, ARENA_TRANSFORMS,
    np, pd, plt,
):
    """Aggregate spatial entropy and revisit rate across all test trials."""
    _TEST_CONFIGS = {'A', 'B', 'C', 'D'}
    _RES = 20
    _N_CELLS = _RES * _RES
    _records = []

    def _to_cell(gx, gy, res):
        cx = np.clip((gx / 5.0 * res).astype(int), 0, res - 1)
        cy = np.clip((gy / 5.0 * res).astype(int), 0, res - 1)
        return cx * res + cy

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

        _dur = _trial['duration_min']
        _t_common = np.arange(0, _dur + 1 / 60, 1 / 60)
        _n_t = len(_t_common)
        _n_sheep = len(_tracks)

        _cells_all = []
        for _sid, _trk in _tracks.items():
            _order = np.argsort(_trk['t'])
            _gxi = np.interp(_t_common, _trk['t'][_order], _trk['gx'][_order])
            _gyi = np.interp(_t_common, _trk['t'][_order], _trk['gy'][_order])
            _cells_all.append(_to_cell(_gxi, _gyi, _RES))
        _cells_all = np.array(_cells_all)

        # Overall entropy (all timesteps, all sheep)
        _counts = np.bincount(_cells_all.ravel(), minlength=_N_CELLS).astype(float)
        _total = _counts.sum()
        _p = _counts[_counts > 0] / _total
        _H = float(-np.sum(_p * np.log2(_p))) / np.log2(_N_CELLS)

        # Revisit rate (per sheep, then averaged)
        _rv = []
        _cov = []
        for _ci in range(_n_sheep):
            _cells = _cells_all[_ci]
            _vis = set()
            _rev = 0
            for _c in _cells:
                if _c in _vis:
                    _rev += 1
                _vis.add(_c)
            _rv.append(_rev / _n_t)
            _cov.append(len(_vis) / _N_CELLS)

        _records.append({
            'Trial': _tidx,
            'Date': _trial['date'],
            'Config': _trial['config'],
            'Group size': _trial['group_size'],
            'Assay': str(_trial['assay']),
            'Norm. spatial entropy': round(_H, 3),
            'Mean revisit rate': round(float(np.mean(_rv)), 3),
            'Mean arena coverage': round(float(np.mean(_cov)), 3),
        })

    if not _records:
        mo.stop(True, mo.md("*No test trials with GPS data found.*"))

    _agg_df = pd.DataFrame(_records)

    _fig2, _axes2 = plt.subplots(1, 3, figsize=(14, 4))

    _assays = sorted(_agg_df['Assay'].unique(), key=lambda x: (not x.isdigit(), x))
    for _ax, _col, _title in zip(
        _axes2,
        ['Norm. spatial entropy', 'Mean revisit rate', 'Mean arena coverage'],
        ['Normalised spatial entropy', 'Mean revisit rate', 'Mean arena coverage'],
    ):
        _vals = [_agg_df[_agg_df['Assay'] == a][_col].dropna().values for a in _assays]
        _ax.boxplot(_vals, labels=_assays)
        _ax.set_xlabel("Assay")
        _ax.set_ylabel(_col)
        _ax.set_title(_title)

    _fig2.suptitle(f"Spatial information across test trials (grid={_RES}×{_RES})")
    _fig2.tight_layout()
    plt.close(_fig2)

    mo.vstack([
        mo.mpl.interactive(_fig2),
        mo.md("### Per-trial table"),
        mo.ui.table(_agg_df),
    ])
    return


if __name__ == "__main__":
    app.run()
