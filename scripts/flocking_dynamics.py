"""Flocking & Group Cohesion Analysis

Computes pairwise inter-animal distances over time to quantify how tightly
the group moves together throughout trials.

Single-trial mode shows:
  - Mean / max pairwise inter-animal distance over time
  - Nearest-neighbour distance (NND) over time
  - Per-sheep spread from group centroid over time

Aggregated mode shows mean NND and spread time series averaged across all
filtered trials, plus box-plot summaries by assay / group size.
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
        build_trials, build_gps_cache, build_arena_transforms,
        load_trial_tracks,
    )

    TRIALS = build_trials()
    ARENA_TRANSFORMS = build_arena_transforms()
    print(f"Loaded {len(TRIALS)} trials — building GPS cache (runs once)…")
    GPS_CACHE = build_gps_cache(TRIALS)
    return (
        np, pd, plt, mo,
        build_trials, build_gps_cache, build_arena_transforms,
        load_trial_tracks,
        TRIALS, ARENA_TRANSFORMS, GPS_CACHE,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(TRIALS, np, load_trial_tracks, GPS_CACHE, ARENA_TRANSFORMS):
    """Compute per-trial flocking metrics for all multi-sheep trials."""

    def _compute_cohesion(trial):
        """Return dict of cohesion metrics for one trial, or None if not enough data."""
        tracks = load_trial_tracks(
            trial,
            gnss_cache=GPS_CACHE,
            apply_orient=False,
            arena_transforms=ARENA_TRANSFORMS,
        )
        if len(tracks) < 2:
            return None

        sheep_ids = sorted(tracks.keys())
        n_sheep = len(sheep_ids)
        dur = trial["duration_min"]
        t_common = np.arange(0, dur + 1 / 60, 1 / 60)
        n_t = len(t_common)

        gx_all = np.zeros((n_sheep, n_t))
        gy_all = np.zeros((n_sheep, n_t))
        for ci, sid in enumerate(sheep_ids):
            trk = tracks[sid]
            order = np.argsort(trk["t"])
            gx_all[ci] = np.interp(t_common, trk["t"][order], trk["gx"][order])
            gy_all[ci] = np.interp(t_common, trk["t"][order], trk["gy"][order])

        cx = gx_all.mean(axis=0)
        cy = gy_all.mean(axis=0)
        dist_to_centroid = np.sqrt((gx_all - cx) ** 2 + (gy_all - cy) ** 2)
        spread = dist_to_centroid.mean(axis=0)

        pw = []
        for i in range(n_sheep):
            for j in range(i + 1, n_sheep):
                pw.append(
                    np.sqrt((gx_all[i] - gx_all[j]) ** 2 + (gy_all[i] - gy_all[j]) ** 2)
                )
        pw = np.array(pw)

        return {
            "t": t_common,
            "mean_pw": pw.mean(axis=0),
            "max_pw": pw.max(axis=0),
            "nnd": pw.min(axis=0),
            "spread": spread,
            "gx_all": gx_all,
            "gy_all": gy_all,
            "sheep_ids": sheep_ids,
            "n_sheep": n_sheep,
            "mean_nnd_scalar": float(pw.min(axis=0).mean() * 10),
            "mean_spread_scalar": float(spread.mean() * 10),
        }

    # Pre-compute cohesion for all multi-sheep trials
    COHESION = {}
    for _tidx, _trial in enumerate(TRIALS):
        if _trial["group_size"] >= 2:
            _result = _compute_cohesion(_trial)
            if _result is not None:
                COHESION[_tidx] = _result

    print(f"Cohesion computed for {len(COHESION)} multi-sheep trials")
    return (COHESION, _compute_cohesion)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(TRIALS, mo):
    mode_widget = mo.ui.radio(
        options=["Single trial", "Aggregated"],
        value="Single trial",
        label="Mode",
    )

    # Single-trial selector
    _options = {}
    for _i, _t in enumerate(TRIALS):
        if _t["group_size"] < 2:
            continue
        _assay_str = f" [Assay {_t['assay']}]" if _t["assay"] is not None else ""
        _label = f"[{_i:3d}] {_t['notes'].split(' - ')[0]:20s} {_t['name']}{_assay_str}"
        _options[_label] = _i

    trial_selector = mo.ui.dropdown(options=_options, label="Select trial (≥2 sheep)")
    bin_slider = mo.ui.slider(start=5, stop=120, step=5, value=30, label="Time bin (s)")

    # Aggregation filters
    _PHASE2 = "2026-02-17"
    _all_configs = sorted({t["config"] for t in TRIALS})
    _all_gsizes = sorted({str(t["group_size"]) for t in TRIALS if t["group_size"] >= 2})
    _all_assays = sorted(
        {str(t["assay"]) for t in TRIALS if t["assay"] is not None},
        key=lambda x: (not x.isdigit(), x),
    )

    config_filter = mo.ui.multiselect(
        options=_all_configs, value=["A", "B", "C", "D"], label="Config(s)"
    )
    groupsize_filter = mo.ui.multiselect(
        options=_all_gsizes, value=[], label="Group size(s) (all if empty)"
    )
    assay_filter = mo.ui.multiselect(
        options=_all_assays, value=[], label="Assay(s) (all if empty)"
    )
    phase_filter = mo.ui.dropdown(
        options=["Both phases", "Phase 1 only (pre Feb 17)", "Phase 2 only (Feb 17+)"],
        value="Both phases",
        label="Study phase",
    )
    aggregateby_widget = mo.ui.radio(
        options=["Assay", "Group Size", "Config", "All"],
        value="Assay",
        label="Aggregate by",
    )

    mo.md(f"""
    # Flocking & Group Cohesion

    **Mode:** {mode_widget}

    ---
    ### Single trial
    {trial_selector} {bin_slider}

    ---
    ### Aggregation filters
    {mo.hstack([phase_filter, config_filter, groupsize_filter, assay_filter])}
    {aggregateby_widget}
    """)
    return (
        mode_widget, trial_selector, bin_slider,
        config_filter, groupsize_filter, assay_filter, phase_filter, aggregateby_widget,
    )


# ---------------------------------------------------------------------------
# Single-trial plot
# ---------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(
    mode_widget, trial_selector, TRIALS, COHESION, bin_slider,
    np, pd, plt, mo,
):
    if mode_widget.value != "Single trial":
        mo.stop(True, mo.md(""))

    if trial_selector.value is None:
        mo.stop(True, mo.md("*Select a trial above.*"))

    _tidx = trial_selector.value
    _trial = TRIALS[_tidx]

    if _tidx not in COHESION:
        mo.stop(True, mo.md("*No GPS data or fewer than 2 sheep found.*"))

    _c = COHESION[_tidx]
    _bin_s = bin_slider.value
    _bin_min = _bin_s / 60.0
    _dur = _trial["duration_min"]
    _t_common = _c["t"]

    _t_edges = np.arange(0, _dur + _bin_min, _bin_min)
    _t_centres = 0.5 * (_t_edges[:-1] + _t_edges[1:])

    def _bin(arr):
        return np.array([
            arr[(_t_common >= _t_edges[b]) & (_t_common < _t_edges[b + 1])].mean()
            if ((_t_common >= _t_edges[b]) & (_t_common < _t_edges[b + 1])).sum() > 0
            else np.nan
            for b in range(len(_t_centres))
        ])

    COLORS_SHEEP = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]

    _fig, _axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    _ax_mean, _ax_max, _ax_nnd, _ax_spread = _axes.flatten()

    for (_ax, _key, _color, _title, _ylabel) in [
        (_ax_mean, "mean_pw", "#2171b5", "Mean inter-animal distance", "Mean pairwise dist. (m)"),
        (_ax_max,  "max_pw",  "#d7191c", "Maximum inter-animal distance", "Max pairwise dist. (m)"),
        (_ax_nnd,  "nnd",     "#1a9641", "Nearest-neighbour distance", "NND (m)"),
    ]:
        _b = _bin(_c[_key]) * 10
        _ax.plot(_t_centres, _b, color=_color, lw=1.5)
        _ax.fill_between(_t_centres, 0, _b, alpha=0.2, color=_color)
        _ax.set_ylabel(_ylabel)
        _ax.set_title(_title)
        _ax.set_ylim(bottom=0)

    _cx = _c["gx_all"].mean(axis=0)
    _cy = _c["gy_all"].mean(axis=0)
    for _ci, _sid in enumerate(_c["sheep_ids"]):
        _d = np.sqrt((_c["gx_all"][_ci] - _cx) ** 2 + (_c["gy_all"][_ci] - _cy) ** 2)
        _ax_spread.plot(_t_centres, _bin(_d) * 10,
                        color=COLORS_SHEEP[_ci % len(COLORS_SHEEP)], lw=1.2, label=_sid)
    _ax_spread.set_ylabel("Dist. to centroid (m)")
    _ax_spread.set_title("Spread: each sheep from centroid")
    _ax_spread.legend(loc="upper right", fontsize=8)
    _ax_spread.set_ylim(bottom=0)

    for _ax in _axes.flatten():
        _ax.set_xlim(0, _dur)
    _axes[1, 0].set_xlabel("Time (min)")
    _axes[1, 1].set_xlabel("Time (min)")

    _fig.suptitle(
        f"Flocking — {_trial['name']}\n"
        f"{_c['n_sheep']} sheep  |  "
        f"mean NND {_c['mean_nnd_scalar']:.1f} m, mean spread {_c['mean_spread_scalar']:.1f} m",
        fontsize=11,
    )
    _fig.tight_layout()

    _summary = pd.DataFrame({
        "Metric": [
            "Mean inter-animal distance (m)",
            "Max inter-animal distance (m)",
            "Mean nearest-neighbour distance (m)",
            "Mean group spread (m)",
        ],
        "Value": [
            round(float(_c["mean_pw"].mean() * 10), 2),
            round(float(_c["max_pw"].mean() * 10), 2),
            round(_c["mean_nnd_scalar"], 2),
            round(_c["mean_spread_scalar"], 2),
        ],
    })

    mo.vstack([
        _fig,
        mo.md("### Summary statistics"),
        mo.ui.table(_summary),
    ])
    return


# ---------------------------------------------------------------------------
# Aggregated plot
# ---------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(
    mode_widget, TRIALS, COHESION,
    config_filter, groupsize_filter, assay_filter, phase_filter, aggregateby_widget,
    np, pd, plt, mo,
):
    if mode_widget.value != "Aggregated":
        mo.stop(True, mo.md(""))

    _PHASE2_DATE = "2026-02-17"
    _gsizes = {int(g) for g in groupsize_filter.value} if groupsize_filter.value else None
    _assays = set(assay_filter.value) if assay_filter.value else None
    _configs = set(config_filter.value) if config_filter.value else None

    # Filter trials
    _selected = []
    for _tidx, _trial in enumerate(TRIALS):
        if _trial["group_size"] < 2:
            continue
        if _tidx not in COHESION:
            continue
        if _configs and _trial["config"] not in _configs:
            continue
        if _gsizes is not None and _trial["group_size"] not in _gsizes:
            continue
        if _assays is not None and str(_trial["assay"]) not in _assays:
            continue
        if phase_filter.value == "Phase 1 only (pre Feb 17)" and _trial["date"] >= _PHASE2_DATE:
            continue
        if phase_filter.value == "Phase 2 only (Feb 17+)" and _trial["date"] < _PHASE2_DATE:
            continue
        _selected.append(_tidx)

    if not _selected:
        mo.stop(True, mo.md("*No trials match the current filters.*"))

    print(f"{len(_selected)} trials selected for aggregation")

    # Group key
    def _group_key(trial):
        by = aggregateby_widget.value
        if by == "Assay":       return str(trial["assay"])
        if by == "Group Size":  return str(trial["group_size"])
        if by == "Config":      return trial["config"]
        return "All"

    _groups: dict[str, list] = {}
    for _tidx in _selected:
        _key = _group_key(TRIALS[_tidx])
        _groups.setdefault(_key, []).append(_tidx)

    _sorted_keys = sorted(_groups.keys(), key=lambda x: (not x.isdigit(), x))

    # ----------------------------------------------------------------
    # Figure 1: time-series — mean NND and spread averaged per group
    # ----------------------------------------------------------------
    _DUR = 35
    _t_common = np.arange(0, _DUR + 1 / 60, 1 / 60)
    _CMAP = plt.cm.get_cmap("tab10", len(_sorted_keys))

    _fig1, (_ax_nnd_ts, _ax_sp_ts) = plt.subplots(1, 2, figsize=(13, 4), sharex=True)

    _records = []
    for _ki, _key in enumerate(_sorted_keys):
        _nnd_curves, _sp_curves = [], []
        for _tidx in _groups[_key]:
            _c = COHESION[_tidx]
            _nnd_curves.append(np.interp(_t_common, _c["t"], _c["nnd"]) * 10)
            _sp_curves.append(np.interp(_t_common, _c["t"], _c["spread"]) * 10)
        _mean_nnd = np.nanmean(_nnd_curves, axis=0)
        _mean_sp = np.nanmean(_sp_curves, axis=0)
        _sem_nnd = np.nanstd(_nnd_curves, axis=0) / max(len(_nnd_curves) ** 0.5, 1)
        _sem_sp = np.nanstd(_sp_curves, axis=0) / max(len(_sp_curves) ** 0.5, 1)

        _color = _CMAP(_ki)
        _lbl = f"{aggregateby_widget.value}={_key} (n={len(_groups[_key])})"
        _ax_nnd_ts.plot(_t_common, _mean_nnd, color=_color, lw=1.5, label=_lbl)
        _ax_nnd_ts.fill_between(_t_common, _mean_nnd - _sem_nnd, _mean_nnd + _sem_nnd,
                                 color=_color, alpha=0.15)
        _ax_sp_ts.plot(_t_common, _mean_sp, color=_color, lw=1.5, label=_lbl)
        _ax_sp_ts.fill_between(_t_common, _mean_sp - _sem_sp, _mean_sp + _sem_sp,
                                color=_color, alpha=0.15)

        for _tidx in _groups[_key]:
            _c = COHESION[_tidx]
            _t = TRIALS[_tidx]
            _records.append({
                aggregateby_widget.value: _key,
                "Trial": _tidx,
                "Date": _t["date"],
                "Config": _t["config"],
                "Group size": _t["group_size"],
                "Assay": str(_t["assay"]),
                "Mean NND (m)": round(_c["mean_nnd_scalar"], 2),
                "Mean spread (m)": round(_c["mean_spread_scalar"], 2),
            })

    _ax_nnd_ts.set_xlabel("Time (min)")
    _ax_nnd_ts.set_ylabel("NND (m)")
    _ax_nnd_ts.set_title("Mean nearest-neighbour distance ± SEM")
    _ax_nnd_ts.set_ylim(bottom=0)
    _ax_nnd_ts.legend(fontsize=8)

    _ax_sp_ts.set_xlabel("Time (min)")
    _ax_sp_ts.set_ylabel("Group spread (m)")
    _ax_sp_ts.set_title("Mean group spread ± SEM")
    _ax_sp_ts.set_ylim(bottom=0)
    _ax_sp_ts.legend(fontsize=8)

    for _ax in [_ax_nnd_ts, _ax_sp_ts]:
        _ax.set_xlim(0, _DUR)

    _fig1.suptitle(
        f"Flocking — aggregated by {aggregateby_widget.value}  "
        f"({len(_selected)} trials)",
        fontsize=11,
    )
    _fig1.tight_layout()

    # ----------------------------------------------------------------
    # Figure 2: box plots of scalar summary per group
    # ----------------------------------------------------------------
    _agg_df = pd.DataFrame(_records)

    _fig2, (_bx1, _bx2) = plt.subplots(1, 2, figsize=(12, 4))
    _nnd_by_key = [_agg_df[_agg_df[aggregateby_widget.value] == k]["Mean NND (m)"].dropna().values
                   for k in _sorted_keys]
    _sp_by_key  = [_agg_df[_agg_df[aggregateby_widget.value] == k]["Mean spread (m)"].dropna().values
                   for k in _sorted_keys]

    _bx1.boxplot(_nnd_by_key, labels=_sorted_keys)
    _bx1.set_xlabel(aggregateby_widget.value)
    _bx1.set_ylabel("Mean NND (m)")
    _bx1.set_title("NND distribution")

    _bx2.boxplot(_sp_by_key, labels=_sorted_keys)
    _bx2.set_xlabel(aggregateby_widget.value)
    _bx2.set_ylabel("Mean spread (m)")
    _bx2.set_title("Group spread distribution")

    _fig2.tight_layout()

    mo.vstack([
        _fig1,
        _fig2,
        mo.md("### Per-trial cohesion table"),
        mo.ui.table(_agg_df),
    ])
    return


if __name__ == "__main__":
    app.run()
