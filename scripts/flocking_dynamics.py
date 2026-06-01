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
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import marimo as mo
    from gps_analysis import (
        build_trials, build_tracks_cache,
        load_trial_tracks,
    )

    TRIALS = build_trials()
    print(f"Loaded {len(TRIALS)} trials — building tracks cache (runs once)…")
    TRACKS_CACHE = build_tracks_cache()
    return (
        np, pd, plt, mo,
        build_trials, build_tracks_cache,
        load_trial_tracks,
        TRIALS, TRACKS_CACHE,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@app.cell(hide_code=True)
def _(TRIALS, np, load_trial_tracks, TRACKS_CACHE):
    """Compute per-trial flocking metrics for all multi-sheep trials."""

    def _compute_cohesion(trial, apply_orient=False):
        """Return dict of cohesion metrics for one trial, or None if not enough data."""
        tracks = load_trial_tracks(
            trial,
            tracks_cache=TRACKS_CACHE,
            apply_orient=apply_orient,
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

    # Pre-compute cohesion for all multi-sheep trials.
    # Test configs (A/B/C/D) use apply_orient=True so trajectories are
    # rotated/reflected into a canonical frame for correct aggregation.
    _TEST_CONFIGS = {'A', 'B', 'C', 'D'}
    COHESION = {}
    for _tidx, _trial in enumerate(TRIALS):
        if _trial["group_size"] >= 2:
            _orient = _trial.get("config") in _TEST_CONFIGS
            _result = _compute_cohesion(_trial, apply_orient=_orient)
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


# ---------------------------------------------------------------------------
# Rigour pass: controls, time-reverse, and assay-shuffle nulls
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rigour pass: controls, time-reverse, and assay-shuffle nulls

    Three diagnostic cells follow:

    1. **Controls-aware aggregated panel** — surfaces `CTRL_FAR` and
       `CTRL_BARN` alongside the test configs (A/B/C/D) so we can read off
       any baseline differences in cohesion that have nothing to do with
       the reward-site manipulation.
    2. **Time-reverse null** — because `mean(NND)` and `mean(spread)` are
       trivially invariant under track reversal (they depend only on the
       per-timestep position *set*), we instead compute a temporal-asymmetry
       contrast `Δ = mean(metric_last_third) − mean(metric_first_third)`
       and use reversal as a sanity check (it flips the sign of `Δ`).
    3. **Assay-shuffle null** — non-parametric significance for the
       monotonic trend of cohesion with assay number, shuffling assay
       labels within each `group_num` (preserves group identity).
    """)
    return


@app.cell(hide_code=True)
def _(TRIALS, COHESION, np, pd, plt, mo):
    """Aggregated panel including CTRL_FAR / CTRL_BARN alongside test configs."""

    _CONFIG_KEEP = {"A", "B", "C", "D", "CTRL_FAR", "CTRL_BARN"}
    _TEST_CONFIGS = {"A", "B", "C", "D"}

    _records = []
    for _tidx, _trial in enumerate(TRIALS):
        if _trial["group_size"] < 2 or _tidx not in COHESION:
            continue
        if _trial["config"] not in _CONFIG_KEEP:
            continue
        _c = COHESION[_tidx]
        _records.append({
            "Trial": _tidx,
            "Date": _trial["date"],
            "Config": _trial["config"],
            "Kind": "TEST" if _trial["config"] in _TEST_CONFIGS else "CTRL",
            "Group size": _trial["group_size"],
            "Assay": str(_trial["assay"]),
            "Mean NND (m)": round(_c["mean_nnd_scalar"], 2),
            "Mean spread (m)": round(_c["mean_spread_scalar"], 2),
        })

    if not _records:
        mo.stop(True, mo.md("*No trials match the controls-aware filter.*"))

    _df = pd.DataFrame(_records)

    # Ordered column layout: tests first, then controls
    _ordered_configs = [c for c in ["A", "B", "C", "D", "CTRL_FAR", "CTRL_BARN"]
                        if c in _df["Config"].unique()]
    _split_idx = sum(1 for c in _ordered_configs if c in _TEST_CONFIGS)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    def _draw(ax, col, ylabel, title):
        _data = [_df[_df["Config"] == c][col].dropna().values for c in _ordered_configs]
        _bp = ax.boxplot(_data, labels=_ordered_configs, patch_artist=True)
        for _i, _patch in enumerate(_bp["boxes"]):
            _patch.set_facecolor("#a6cee3" if _i < _split_idx else "#fdbf6f")
            _patch.set_alpha(0.75)
        if 0 < _split_idx < len(_ordered_configs):
            ax.axvline(_split_idx + 0.5, color="0.4", lw=1, ls="--")
            _ymax = ax.get_ylim()[1]
            ax.text(_split_idx / 2 + 0.5, _ymax * 0.97, "TEST",
                    ha="center", va="top", fontsize=9, color="#1f78b4")
            ax.text(_split_idx + (len(_ordered_configs) - _split_idx) / 2 + 0.5,
                    _ymax * 0.97, "CTRL",
                    ha="center", va="top", fontsize=9, color="#ff7f00")
        ax.set_xlabel("Config")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0)

    _draw(_ax1, "Mean NND (m)", "Mean NND (m)", "Nearest-neighbour distance by config")
    _draw(_ax2, "Mean spread (m)", "Mean spread (m)", "Group spread by config")
    _fig.suptitle(
        f"Cohesion: TEST (A/B/C/D) vs CTRL configs  "
        f"({len(_df)} trials, {(_df['Kind']=='CTRL').sum()} controls)",
        fontsize=11,
    )
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md("### Per-trial cohesion (controls-aware)"),
        mo.ui.table(_df.sort_values(["Kind", "Config", "Date"]).reset_index(drop=True)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time-reverse null — early-vs-late framing

    `mean(NND)` and `mean(spread)` over a whole trial are **trivially
    invariant under track reversal**: reversing time does not change the
    multiset of per-timestep positions, so the metric is identical. A naive
    time-reverse null would be tautological.

    To make the null meaningful, we use a temporal-asymmetry contrast:
    `Δ = mean(metric_last_third) − mean(metric_first_third)`. Reversal
    flips the sign of `Δ` (`Δ_rev = −Δ_fwd`), so:

    - The observed `|Δ_fwd|` is the substantive quantity (does cohesion
      drift over the trial?).
    - The reversed version is a sign-flip sanity check, not a separate
      sample.
    """)
    return


@app.cell(hide_code=True)
def _(TRIALS, COHESION, np, pd, plt, mo):
    """Time-reverse null using early-vs-late asymmetry contrast."""

    _CONFIG_KEEP = {"A", "B", "C", "D", "CTRL_FAR", "CTRL_BARN"}
    _rng = np.random.default_rng(seed=42)

    _records = []
    for _tidx, _trial in enumerate(TRIALS):
        if _trial["group_size"] < 2 or _tidx not in COHESION:
            continue
        if _trial["config"] not in _CONFIG_KEEP:
            continue
        _c = COHESION[_tidx]
        _n3 = len(_c["t"]) // 3
        if _n3 < 2:
            continue
        _d_nnd = _c["nnd"][-_n3:].mean() * 10 - _c["nnd"][:_n3].mean() * 10
        _d_sp = _c["spread"][-_n3:].mean() * 10 - _c["spread"][:_n3].mean() * 10
        _records.append({
            "Trial": _tidx,
            "Config": _trial["config"],
            "Assay": str(_trial["assay"]),
            "delta_nnd_fwd": _d_nnd,
            "delta_nnd_rev": -_d_nnd,
            "delta_spread_fwd": _d_sp,
            "delta_spread_rev": -_d_sp,
        })

    if not _records:
        mo.stop(True, mo.md("*No trials available for time-reverse null.*"))

    _df = pd.DataFrame(_records)
    _assays = sorted(_df["Assay"].unique(), key=lambda x: (not x.isdigit(), x))
    _cmap = plt.cm.get_cmap("tab10", max(len(_assays), 1))
    _assay_color = {a: _cmap(i) for i, a in enumerate(_assays)}

    def _bootstrap_ci(values, n=1000):
        if len(values) == 0:
            return (np.nan, np.nan, np.nan)
        _vals = np.asarray(values, dtype=float)
        _samples = _rng.choice(_vals, size=(n, len(_vals)), replace=True).mean(axis=1)
        return float(_vals.mean()), float(np.percentile(_samples, 2.5)), float(np.percentile(_samples, 97.5))

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _ax_scatter, _ax_strip_nnd, _ax_strip_sp = _axes

    # Paired scatter Δ_fwd vs Δ_rev (sits on y = −x by construction)
    for _a in _assays:
        _sub = _df[_df["Assay"] == _a]
        _ax_scatter.scatter(_sub["delta_nnd_fwd"], _sub["delta_nnd_rev"],
                            color=_assay_color[_a], label=f"Assay {_a}",
                            s=30, alpha=0.75, edgecolor="white", linewidth=0.5)
    _lim = max(abs(_df["delta_nnd_fwd"]).max(), 0.1) * 1.1
    _ax_scatter.plot([-_lim, _lim], [_lim, -_lim], color="0.5", lw=0.8, ls="--",
                     label="y = −x")
    _ax_scatter.axhline(0, color="0.7", lw=0.5)
    _ax_scatter.axvline(0, color="0.7", lw=0.5)
    _ax_scatter.set_xlabel("Δ NND forward (m)")
    _ax_scatter.set_ylabel("Δ NND reversed (m)")
    _ax_scatter.set_title("Δ_fwd vs Δ_rev (sanity check)")
    _ax_scatter.legend(fontsize=7, loc="best")
    _ax_scatter.set_xlim(-_lim, _lim)
    _ax_scatter.set_ylim(-_lim, _lim)

    # Strip plots per assay with bootstrap CI for each metric
    def _strip(ax, col, ylabel):
        for _i, _a in enumerate(_assays):
            _vals = _df[_df["Assay"] == _a][col].values
            _jitter = _rng.uniform(-0.15, 0.15, size=len(_vals))
            ax.scatter(np.full(len(_vals), _i) + _jitter, _vals,
                       color=_assay_color[_a], s=22, alpha=0.7,
                       edgecolor="white", linewidth=0.4)
            _mean, _lo, _hi = _bootstrap_ci(_vals)
            if not np.isnan(_mean):
                ax.errorbar([_i], [_mean], yerr=[[_mean - _lo], [_hi - _mean]],
                            fmt="o", color="black", capsize=4, lw=1.2, ms=5,
                            zorder=10)
        ax.axhline(0, color="red", lw=0.8, ls="--")
        ax.set_xticks(range(len(_assays)))
        ax.set_xticklabels(_assays)
        ax.set_xlabel("Assay")
        ax.set_ylabel(ylabel)

    _strip(_ax_strip_nnd, "delta_nnd_fwd", "Δ NND last-first third (m)")
    _ax_strip_nnd.set_title("Forward Δ NND by assay (mean ± 95% boot CI)")
    _strip(_ax_strip_sp, "delta_spread_fwd", "Δ spread last-first third (m)")
    _ax_strip_sp.set_title("Forward Δ spread by assay (mean ± 95% boot CI)")

    _fig.suptitle("Time-reverse null: early-vs-late asymmetry contrast", fontsize=11)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md("### Per-trial Δ table (forward = last_third − first_third)"),
        mo.ui.table(_df.round(3)),
    ])
    return


@app.cell(hide_code=True)
def _(TRIALS, COHESION, np, pd, plt, mo):
    """Assay-shuffle null on Spearman ρ between assay and cohesion."""
    from scipy.stats import spearmanr

    _CONFIG_KEEP = {"A", "B", "C", "D", "CTRL_FAR", "CTRL_BARN"}
    _TEST_CONFIGS = {"A", "B", "C", "D"}
    _N_PERMUTATIONS = 1000
    _rng = np.random.default_rng(seed=42)

    _records = []
    for _tidx, _trial in enumerate(TRIALS):
        if _trial["group_size"] < 2 or _tidx not in COHESION:
            continue
        if _trial["config"] not in _CONFIG_KEEP:
            continue
        _av = _trial["assay"]
        try:
            _aint = int(_av)
        except (TypeError, ValueError):
            continue
        _c = COHESION[_tidx]
        _records.append({
            "Trial": _tidx,
            "Config": _trial["config"],
            "Assay": _aint,
            "group_num": _trial["group_num"],
            "Mean NND (m)": _c["mean_nnd_scalar"],
            "Mean spread (m)": _c["mean_spread_scalar"],
        })

    if not _records:
        mo.stop(True, mo.md("*No trials with digit-castable assay for shuffle null.*"))

    _df = pd.DataFrame(_records)
    _df_test = _df[_df["Config"].isin(_TEST_CONFIGS)].reset_index(drop=True)
    if len(_df_test) < 3:
        mo.stop(True, mo.md("*Not enough test-config trials for Spearman.*"))

    _assay = _df_test["Assay"].to_numpy()
    _groupnum = _df_test["group_num"].to_numpy()
    _y_nnd = _df_test["Mean NND (m)"].to_numpy()
    _y_sp = _df_test["Mean spread (m)"].to_numpy()

    _rho_nnd_obs, _ = spearmanr(_assay, _y_nnd)
    _rho_sp_obs, _ = spearmanr(_assay, _y_sp)

    # Pre-compute group index slices once; permute assay within each group_num
    _unique_groups = np.unique(_groupnum)
    _group_indices = [np.flatnonzero(_groupnum == _g) for _g in _unique_groups]
    _null_nnd = np.empty(_N_PERMUTATIONS)
    _null_sp = np.empty(_N_PERMUTATIONS)
    for _p in range(_N_PERMUTATIONS):
        _shuf = _assay.copy()
        for _idx in _group_indices:
            _shuf[_idx] = _assay[_rng.permutation(_idx)]
        _null_nnd[_p], _ = spearmanr(_shuf, _y_nnd)
        _null_sp[_p], _ = spearmanr(_shuf, _y_sp)

    _p_nnd = float((np.abs(_null_nnd) >= np.abs(_rho_nnd_obs)).mean())
    _p_sp = float((np.abs(_null_sp) >= np.abs(_rho_sp_obs)).mean())
    _ci_nnd = (float(np.percentile(_null_nnd, 2.5)), float(np.percentile(_null_nnd, 97.5)))
    _ci_sp = (float(np.percentile(_null_sp, 2.5)), float(np.percentile(_null_sp, 97.5)))

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    _ax1.hist(_null_nnd, bins=40, color="#a6cee3", edgecolor="white")
    _ax1.axvline(_rho_nnd_obs, color="red", lw=2,
                 label=f"observed ρ = {_rho_nnd_obs:.3f}")
    _ax1.axvline(0, color="0.5", lw=0.7, ls="--")
    _ax1.set_xlabel("Spearman ρ (assay vs mean NND)")
    _ax1.set_ylabel(f"Count (N={_N_PERMUTATIONS})")
    _ax1.set_title(f"NND ρ null (two-sided p = {_p_nnd:.3f})")
    _ax1.legend(fontsize=9)

    _ax2.hist(_null_sp, bins=40, color="#b2df8a", edgecolor="white")
    _ax2.axvline(_rho_sp_obs, color="red", lw=2,
                 label=f"observed ρ = {_rho_sp_obs:.3f}")
    _ax2.axvline(0, color="0.5", lw=0.7, ls="--")
    _ax2.set_xlabel("Spearman ρ (assay vs mean spread)")
    _ax2.set_ylabel(f"Count (N={_N_PERMUTATIONS})")
    _ax2.set_title(f"Spread ρ null (two-sided p = {_p_sp:.3f})")
    _ax2.legend(fontsize=9)

    _fig.suptitle(
        f"Assay-shuffle null (within group_num) — {len(_df_test)} test trials, "
        f"{len(_unique_groups)} groups",
        fontsize=11,
    )
    _fig.tight_layout()

    _summary = pd.DataFrame({
        "Metric": ["Mean NND", "Mean spread"],
        "Observed ρ": [round(_rho_nnd_obs, 4), round(_rho_sp_obs, 4)],
        "Null mean ρ": [round(float(_null_nnd.mean()), 4),
                        round(float(_null_sp.mean()), 4)],
        "Null 95% CI": [f"[{_ci_nnd[0]:.3f}, {_ci_nnd[1]:.3f}]",
                        f"[{_ci_sp[0]:.3f}, {_ci_sp[1]:.3f}]"],
        "Two-sided p": [round(_p_nnd, 4), round(_p_sp, 4)],
    })

    print(f"[assay-shuffle null] NND: observed ρ={_rho_nnd_obs:+.3f}, "
          f"two-sided p={_p_nnd:.3f}  |  "
          f"spread: observed ρ={_rho_sp_obs:+.3f}, two-sided p={_p_sp:.3f}")

    mo.vstack([
        _fig,
        mo.md(
            f"**Assay-shuffle null** — labels shuffled within `group_num` "
            f"({_N_PERMUTATIONS} permutations, seed 42). Spearman ρ computed "
            f"on **test configs only** ({len(_df_test)} trials)."
        ),
        mo.ui.table(_summary),
    ])
    return


if __name__ == "__main__":
    app.run()
