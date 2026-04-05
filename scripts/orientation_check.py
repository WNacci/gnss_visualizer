"""Aggregation Orientation Diagnostic

Checks whether per-configuration orientation transforms correctly align
sheep trajectories across configurations.

For each configuration (A, B, C, D), shows:
  - Occupancy heatmap WITHOUT orientation transform applied
  - Occupancy heatmap WITH orientation transform applied
  - Reward-site grid overlay (sites should cluster at known positions after transform)

If the transforms are correct, the WITH-transform heatmaps should look
consistent across configurations — e.g. all show higher occupancy near the
same canonical reward-site positions.

Todos addressed: #1 (aggregation orientation check)
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
    from concurrent.futures import ThreadPoolExecutor
    from analysis_utils import (
        build_trials, load_gnss_date, build_arena_transforms,
        load_trial_tracks, CONFIG_TRANSFORMS, SITE_GRID,
        apply_orientation,
    )

    TRIALS = build_trials()
    ARENA_TRANSFORMS = build_arena_transforms()
    print(f"Loaded {len(TRIALS)} trials")
    return (
        np, pd, plt, mo, matplotlib, ThreadPoolExecutor,
        build_trials, load_gnss_date, build_arena_transforms,
        load_trial_tracks, CONFIG_TRANSFORMS, SITE_GRID, apply_orientation,
        TRIALS, ARENA_TRANSFORMS,
    )


@app.cell(hide_code=True)
def _(mo):
    phase_radio = mo.ui.radio(
        options=["Phase 1 (pre Feb 17)", "Phase 2 (Feb 17+)", "Both"],
        value="Both",
        label="Study phase",
    )
    groupsize_multi = mo.ui.multiselect(
        options=["1", "2", "4", "8"], value=[], label="Group size (all if empty)",
    )
    bins_slider = mo.ui.slider(start=20, stop=300, step=10, value=100, label="Bins")
    assay_multi = mo.ui.multiselect(
        options=[str(i) for i in range(8)] + ["PROB", "PROB_TEST", "SOLO"],
        value=[],
        label="Assay(s) (all if empty)",
    )

    mo.md(f"""
    # Orientation Transform Diagnostic

    Compares occupancy heatmaps **with** and **without** per-configuration
    orientation transforms.  If transforms are correct the WITH-transform
    columns should look consistent across configurations.

    {mo.hstack([phase_radio, groupsize_multi, assay_multi, bins_slider])}
    """)
    return phase_radio, groupsize_multi, bins_slider, assay_multi


@app.cell(hide_code=True)
def _(
    TRIALS, phase_radio, groupsize_multi, assay_multi,
    load_gnss_date, load_trial_tracks, ARENA_TRANSFORMS,
    np, mo,
):
    """Load GPS data for all filtered trials and bin into per-config arrays."""
    _TEST_CONFIGS = ['A', 'B', 'C', 'D']
    _PHASE2_DATE = "2026-02-17"

    _gsizes = {int(g) for g in groupsize_multi.value} if groupsize_multi.value else None
    _assays = set(assay_multi.value) if assay_multi.value else None

    # Collect (config, gx_raw, gy_raw, gx_oriented, gy_oriented) per trial
    config_points_raw = {c: ([], []) for c in _TEST_CONFIGS}
    config_points_ori = {c: ([], []) for c in _TEST_CONFIGS}

    from analysis_utils import CONFIG_TRANSFORMS, apply_orientation

    _loaded = 0
    for _tidx, _trial in enumerate(TRIALS):
        if _trial['config'] not in _TEST_CONFIGS:
            continue
        if phase_radio.value == "Phase 1 (pre Feb 17)" and _trial['date'] >= _PHASE2_DATE:
            continue
        if phase_radio.value == "Phase 2 (Feb 17+)" and _trial['date'] < _PHASE2_DATE:
            continue
        if _gsizes is not None and _trial['group_size'] not in _gsizes:
            continue
        if _assays is not None and str(_trial['assay']) not in _assays:
            continue

        _gnss = load_gnss_date(_trial['date'])
        _tracks = load_trial_tracks(
            _trial, gnss_cache={_trial['date']: _gnss},
            apply_orient=False, arena_transforms=ARENA_TRANSFORMS,
        )
        if not _tracks:
            continue

        _rot, _ref = CONFIG_TRANSFORMS.get(_trial['config'], (0, "none"))
        _cfg = _trial['config']

        for _sid, _trk in _tracks.items():
            _gx, _gy = _trk['gx'], _trk['gy']
            # Raw (no orientation)
            config_points_raw[_cfg][0].append(_gx)
            config_points_raw[_cfg][1].append(_gy)
            # Oriented
            _gxo, _gyo = apply_orientation(_gx, _gy, _rot, _ref)
            config_points_ori[_cfg][0].append(_gxo)
            config_points_ori[_cfg][1].append(_gyo)
        _loaded += 1

    # Concatenate
    for _cfg in _TEST_CONFIGS:
        _rx, _ry = config_points_raw[_cfg]
        config_points_raw[_cfg] = (
            np.concatenate(_rx) if _rx else np.array([]),
            np.concatenate(_ry) if _ry else np.array([]),
        )
        _ox, _oy = config_points_ori[_cfg]
        config_points_ori[_cfg] = (
            np.concatenate(_ox) if _ox else np.array([]),
            np.concatenate(_oy) if _oy else np.array([]),
        )

    print(f"Loaded {_loaded} trials")
    for _cfg in _TEST_CONFIGS:
        print(f"  Config {_cfg}: {len(config_points_raw[_cfg][0]):,} raw points")
    return (config_points_raw, config_points_ori)


@app.cell(hide_code=True)
def _(config_points_raw, config_points_ori, bins_slider, SITE_GRID, np, plt, mo):
    """Plot raw vs oriented heatmaps side by side for each config."""
    _TEST_CONFIGS = ['A', 'B', 'C', 'D']
    _bins = bins_slider.value
    _fig, _axes = plt.subplots(
        len(_TEST_CONFIGS), 2,
        figsize=(12, 4.5 * len(_TEST_CONFIGS)),
    )

    _site_xs = [v[0] for v in SITE_GRID.values()]
    _site_ys = [v[1] for v in SITE_GRID.values()]

    for _ri, _cfg in enumerate(_TEST_CONFIGS):
        _ax_raw = _axes[_ri, 0]
        _ax_ori = _axes[_ri, 1]

        for _ax, (_gx, _gy), _title in [
            (_ax_raw, config_points_raw[_cfg], f"Config {_cfg} — raw (no transform)"),
            (_ax_ori, config_points_ori[_cfg], f"Config {_cfg} — oriented"),
        ]:
            if len(_gx) > 0:
                _in = (_gx >= 0) & (_gx <= 5) & (_gy >= 0) & (_gy <= 5)
                _H, _xe, _ye = np.histogram2d(
                    _gx[_in], _gy[_in], bins=_bins, range=[[0, 5], [0, 5]],
                )
                _ax.imshow(
                    _H.T, origin='lower', extent=[0, 5, 0, 5],
                    cmap='hot_r', aspect='equal', interpolation='nearest',
                )
            else:
                _ax.set_facecolor('#222')

            # Reward site overlay
            _ax.scatter(_site_xs, _site_ys, c='cyan', s=50, zorder=5,
                        edgecolors='k', lw=0.5, label='reward sites')
            _ax.set_xlim(0, 5)
            _ax.set_ylim(0, 5)
            _ax.set_title(_title, fontsize=10)
            _ax.set_xlabel("grid x")
            _ax.set_ylabel("grid y")

        # Draw arena border
        for _ax in [_ax_raw, _ax_ori]:
            _ax.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'w-', lw=1, alpha=0.5)

    _fig.suptitle(
        "Orientation check: raw (left) vs oriented (right)\n"
        "Cyan dots = reward site grid positions; good transforms → right column consistent",
        fontsize=11, y=1.01,
    )
    _fig.tight_layout()
    plt.close(_fig)
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
