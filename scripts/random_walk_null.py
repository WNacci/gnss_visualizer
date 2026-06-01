"""Random-Walk Null Model

Tests whether observed sheep trajectories differ from per-sheep movement-matched
correlated random walks on coverage, revisit rate, straightness, and time to
discover reward sites.

Each sheep's empirical step-length and turn-angle distributions are fit, then
K random walks are simulated with the same starting position, duration, and
movement statistics. Reflective boundary at [0, 5]^2. Real metrics are
compared to the simulated null envelope via directional empirical p-values.
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
        load_trial_tracks, detect_site_visits, cumulative_path_length,
        SITE_GRID, DATA_DIR,
    )

    KEEP_CONFIGS = {"A", "B", "C", "D", "CTRL_FAR", "CTRL_BARN"}
    K_DEFAULT = 50
    ARENA_LO = 0.0
    ARENA_HI = 5.0
    COVERAGE_BIN = 0.1
    PHASE2_DATE = "2026-02-17"
    SITE_RADIUS = 0.5

    TRIALS = build_trials()
    print(f"Loaded {len(TRIALS)} trials — building tracks cache (runs once)…")
    TRACKS_CACHE = build_tracks_cache()

    return (
        np, pd, plt, mo, matplotlib,
        load_trial_tracks, detect_site_visits, cumulative_path_length,
        SITE_GRID, DATA_DIR,
        TRIALS, TRACKS_CACHE,
        KEEP_CONFIGS, K_DEFAULT, ARENA_LO, ARENA_HI, COVERAGE_BIN,
        PHASE2_DATE, SITE_RADIUS,
    )


@app.cell(hide_code=True)
def _(mo, K_DEFAULT):
    K_slider = mo.ui.slider(
        start=10, stop=100, step=5, value=K_DEFAULT,
        label="K simulated walks per sheep",
    )
    phase_dd = mo.ui.dropdown(
        options=["All trials", "Phase 2 (>=2026-02-17)"],
        value="All trials",
        label="Trial phase",
    )
    time_window_slider = mo.ui.range_slider(
        start=0.0, stop=35.0, step=0.5, value=(0.0, 35.0),
        label="Trial-time window (min)",
    )

    mo.md(f"""
    # Random-walk null model

    For each sheep we fit the empirical step-length and turn-angle distributions
    (stationary suppressed) and simulate **K** correlated random walks of the
    same duration starting at the same position. The arena boundary
    (`[0, 5]^2`) is reflective.

    Metrics compared against the null envelope:

    - **coverage** — number of unique 0.1-unit cells visited (real > null?)
    - **revisit rate** — samples per visited cell (real < null?)
    - **straightness** — net displacement / total path (real > null?)
    - **sites found by time** — cumulative reward-site discovery curve
      (real ahead of null?)

    Use the time-window slider to restrict the analysis to a sub-window of
    the trial (e.g. only the first 5 min, when behaviour may be less
    diffusive). Step-length and turn-angle distributions are refit on the
    selected window and simulated walks have the matching number of steps.

    {K_slider}
    {phase_dd}
    {time_window_slider}
    """)
    return K_slider, phase_dd, time_window_slider


@app.cell(hide_code=True)
def _(np):
    def fit_movement_stats(gx, gy, t):
        """Empirical step lengths and turn angles (stationary-suppressed).

        Returns
        -------
        steps_emp : ndarray, valid non-zero step lengths (grid units).
        turns_emp : ndarray, wrapped turn angles in [-pi, pi].
        """
        gx = np.asarray(gx, dtype=float)
        gy = np.asarray(gy, dtype=float)
        dx = np.diff(gx)
        dy = np.diff(gy)

        steps = np.sqrt(dx ** 2 + dy ** 2)
        valid_steps = np.isfinite(steps) & (steps > 1e-6)
        steps_emp = steps[valid_steps]

        # Only compute heading where there was real motion; otherwise heading
        # is undefined and would inject spurious turns.
        if valid_steps.sum() < 3:
            return steps_emp, np.array([], dtype=float)

        heading = np.arctan2(dy[valid_steps], dx[valid_steps])
        turns = np.diff(np.unwrap(heading))
        # Wrap to [-pi, pi]
        turns_emp = (turns + np.pi) % (2 * np.pi) - np.pi
        turns_emp = turns_emp[np.isfinite(turns_emp)]
        return steps_emp, turns_emp
    return (fit_movement_stats,)


@app.cell(hide_code=True)
def _(np, ARENA_LO, ARENA_HI):
    def _reflect(arr, lo, hi):
        """Fold arr into [lo, hi] via periodic tent-map reflection."""
        span = hi - lo
        u = (arr - lo) % (2 * span)
        return lo + np.where(u <= span, u, 2 * span - u)

    def simulate_walk(start_xy, n_steps, steps_emp, turns_emp, rng):
        """Correlated random walk with reflective boundary on [ARENA_LO, ARENA_HI]^2.

        Vectorised: builds the unbounded trajectory with cumulative ops, then
        folds it into the arena via a tent-map reflection. Equivalent to a
        bouncing-wall reflection for position; the next-step heading detail is
        absorbed because turn angles are resampled IID from the empirical pool.
        """
        if len(steps_emp) == 0 or n_steps <= 0:
            x0, y0 = start_xy
            return np.array([x0], dtype=float), np.array([y0], dtype=float)

        if len(turns_emp) == 0:
            turns_emp = np.array([0.0])

        sampled_steps = rng.choice(steps_emp, size=n_steps)
        sampled_turns = rng.choice(turns_emp, size=n_steps)
        heading0 = rng.uniform(-np.pi, np.pi)
        headings = heading0 + np.cumsum(sampled_turns)

        dx = sampled_steps * np.cos(headings)
        dy = sampled_steps * np.sin(headings)
        x0, y0 = start_xy
        x_unb = x0 + np.concatenate([[0.0], np.cumsum(dx)])
        y_unb = y0 + np.concatenate([[0.0], np.cumsum(dy)])

        gx = _reflect(x_unb, ARENA_LO, ARENA_HI)
        gy = _reflect(y_unb, ARENA_LO, ARENA_HI)
        return gx, gy
    return (simulate_walk,)


@app.cell(hide_code=True)
def _(np, ARENA_LO, ARENA_HI, COVERAGE_BIN, SITE_GRID, SITE_RADIUS):
    _N_BINS = int(round((ARENA_HI - ARENA_LO) / COVERAGE_BIN))

    def coverage(gx, gy):
        """Count of unique 0.1-unit cells visited."""
        if len(gx) == 0:
            return 0
        ix = np.clip(((gx - ARENA_LO) / COVERAGE_BIN).astype(int), 0, _N_BINS - 1)
        iy = np.clip(((gy - ARENA_LO) / COVERAGE_BIN).astype(int), 0, _N_BINS - 1)
        return int(np.unique(ix * _N_BINS + iy).size)

    def revisit_rate(gx, gy):
        """Samples per unique visited cell."""
        c = coverage(gx, gy)
        return float(len(gx)) / float(max(c, 1))

    def straightness(gx, gy):
        if len(gx) < 2:
            return 0.0
        d = float(np.hypot(gx[-1] - gx[0], gy[-1] - gy[0]))
        path = float(np.sum(np.hypot(np.diff(gx), np.diff(gy))))
        return d / path if path > 0 else 0.0

    def sites_found_by_time(gx, gy, t, sites=SITE_GRID, radius=SITE_RADIUS):
        """First-visit times (minutes) for each canonical reward site.

        Mirrors detect_site_visits semantics (distance <= radius), but
        per-sheep and without dwell filtering — used for cumulative
        discovery curves.
        """
        gx = np.asarray(gx, dtype=float)
        gy = np.asarray(gy, dtype=float)
        t = np.asarray(t, dtype=float)
        first_times = []
        for _label, (sx, sy) in sites.items():
            dist = np.sqrt((gx - sx) ** 2 + (gy - sy) ** 2)
            inside = dist <= radius
            if inside.any():
                idx = int(np.argmax(inside))
                first_times.append(float(t[idx]))
        first_times.sort()
        return first_times
    return coverage, revisit_rate, straightness, sites_found_by_time


@app.cell(hide_code=True)
def _(
    np, pd, mo,
    TRIALS, TRACKS_CACHE, load_trial_tracks,
    KEEP_CONFIGS, PHASE2_DATE,
    fit_movement_stats, simulate_walk,
    coverage, revisit_rate, straightness, sites_found_by_time,
    K_slider, phase_dd, time_window_slider,
):
    # Results are cached on (K, phase, window) so re-renders skip the heavy loop.
    _t_start, _t_end = (
        float(time_window_slider.value[0]),
        float(time_window_slider.value[1]),
    )
    _CACHE_KEY = (int(K_slider.value), phase_dd.value, _t_start, _t_end)
    _cache = globals().setdefault("_RW_NULL_CACHE", {})

    if _CACHE_KEY in _cache:
        sim_records, real_records, discovery_curves = _cache[_CACHE_KEY]
    else:
        K = int(K_slider.value)
        _phase = phase_dd.value
        sim_records = []
        real_records = []
        discovery_curves: dict = {}

        _n_trials_used = 0
        _n_sheep_used = 0
        _sim_rng = np.random.default_rng(seed=42)

        for _tidx, _trial in enumerate(TRIALS):
            _cfg = _trial.get("config")
            if _cfg not in KEEP_CONFIGS:
                continue
            if _phase != "All trials" and str(_trial.get("date", "")) < PHASE2_DATE:
                continue
            _tracks = load_trial_tracks(
                _trial, tracks_cache=TRACKS_CACHE, apply_orient=True,
            )
            if not _tracks:
                continue

            _assay = "CTRL" if _cfg.startswith("CTRL") else _cfg
            _bucket = discovery_curves.setdefault(_assay, {"real": [], "sim": []})
            _n_trials_used += 1

            for _sheep_id, _trk in _tracks.items():
                _gx_full = np.asarray(_trk["gx"], dtype=float)
                _gy_full = np.asarray(_trk["gy"], dtype=float)
                _t_full = np.asarray(_trk["t"], dtype=float)
                _mask = (_t_full >= _t_start) & (_t_full <= _t_end)
                _gx = _gx_full[_mask]
                _gy = _gy_full[_mask]
                _t = _t_full[_mask]
                if len(_gx) < 20:
                    continue

                _steps_emp, _turns_emp = fit_movement_stats(_gx, _gy, _t)
                if len(_steps_emp) < 10:
                    continue
                _n_sheep_used += 1

                _real_cov = coverage(_gx, _gy)
                _real_rev = revisit_rate(_gx, _gy)
                _real_str = straightness(_gx, _gy)
                _real_sites = sites_found_by_time(_gx, _gy, _t)
                real_records.append({
                    "assay": _assay,
                    "trial_idx": _tidx,
                    "sheep": _sheep_id,
                    "coverage": _real_cov,
                    "revisit": _real_rev,
                    "straightness": _real_str,
                    "n_sites": len(_real_sites),
                })
                _bucket["real"].append(_real_sites)

                _n_steps = len(_gx) - 1
                _start = (float(_gx[0]), float(_gy[0]))
                for _k in range(K):
                    _sgx, _sgy = simulate_walk(
                        _start, _n_steps, _steps_emp, _turns_emp, _sim_rng,
                    )
                    # Reuse the empirical time-vector for simulated walks so
                    # site discovery curves are directly comparable.
                    _st = _t[: len(_sgx)]
                    _sim_cov = coverage(_sgx, _sgy)
                    _sim_rev = revisit_rate(_sgx, _sgy)
                    _sim_str = straightness(_sgx, _sgy)
                    _sim_sites = sites_found_by_time(_sgx, _sgy, _st)
                    sim_records.append({
                        "assay": _assay,
                        "trial_idx": _tidx,
                        "sheep": _sheep_id,
                        "k": _k,
                        "coverage": _sim_cov,
                        "revisit": _sim_rev,
                        "straightness": _sim_str,
                        "n_sites": len(_sim_sites),
                    })
                    _bucket["sim"].append(_sim_sites)

        _cache[_CACHE_KEY] = (sim_records, real_records, discovery_curves)
        print(
            f"Simulated K={K} walks for {_n_sheep_used} sheep "
            f"across {_n_trials_used} trials "
            f"(phase={_phase}); {len(sim_records)} sim records.",
        )

    real_df = pd.DataFrame(real_records)
    sim_df = pd.DataFrame(sim_records)
    mo.md(
        f"**Real records:** {len(real_df)}  |  "
        f"**Simulated records:** {len(sim_df)}  |  "
        f"**Assays present:** {sorted(real_df['assay'].unique()) if len(real_df) else []}",
    )
    return real_df, sim_df, discovery_curves


@app.cell(hide_code=True)
def _(np, pd, plt, mo, real_df, sim_df, discovery_curves, time_window_slider):
    if len(real_df) == 0:
        mo.stop(True, mo.md("*No sheep matched the current filter.*"))

    _t_start_plot, _t_end_plot = (
        float(time_window_slider.value[0]),
        float(time_window_slider.value[1]),
    )
    _t_eval = 0.5 * (_t_start_plot + _t_end_plot)

    _assays = sorted(real_df["assay"].unique())
    _metric_defs = [
        ("coverage", "Coverage (unique 0.1-unit cells)", "greater"),
        ("revisit", "Revisit rate (samples / cell)", "less"),
        ("straightness", "Straightness (disp / path)", "greater"),
    ]

    _fig, _axes = plt.subplots(2, 2, figsize=(13, 9))
    _ax_map = {
        "coverage": _axes[0, 0],
        "revisit": _axes[0, 1],
        "straightness": _axes[1, 0],
    }

    _pval_rows = []

    for _metric, _title, _direction in _metric_defs:
        _ax = _ax_map[_metric]
        _real_vals = [real_df.loc[real_df["assay"] == a, _metric].values
                      for a in _assays]
        _bp = _ax.boxplot(
            _real_vals, tick_labels=_assays, widths=0.5,
            patch_artist=True,
        )
        for _patch in _bp["boxes"]:
            _patch.set_facecolor("#377eb8")
            _patch.set_alpha(0.55)

        for _i, _assay in enumerate(_assays, start=1):
            _sim_vals = sim_df.loc[sim_df["assay"] == _assay, _metric].values
            _real_assay = real_df.loc[real_df["assay"] == _assay, _metric].values
            if len(_sim_vals) == 0 or len(_real_assay) == 0:
                continue
            _p05, _p50, _p95 = np.percentile(_sim_vals, [5, 50, 95])
            _ax.fill_betweenx(
                [_p05, _p95], _i - 0.3, _i + 0.3,
                color="#e41a1c", alpha=0.18,
                label="Sim p05–p95" if _i == 1 else None,
            )
            _ax.hlines(
                _p50, _i - 0.3, _i + 0.3,
                colors="#e41a1c", linewidth=2.0,
                label="Sim median" if _i == 1 else None,
            )

            # Directional empirical p-value, computed per sheep-trial pair
            # (avg p across all real points for the assay).
            _p_per_real = []
            for _rv in _real_assay:
                if _direction == "greater":
                    _p = float(np.mean(_sim_vals >= _rv))
                else:
                    _p = float(np.mean(_sim_vals <= _rv))
                _p_per_real.append(_p)
            _p_mean = float(np.mean(_p_per_real)) if _p_per_real else float("nan")
            _pval_rows.append({
                "Assay": _assay,
                "Metric": _metric,
                "Real median": round(float(np.median(_real_assay)), 4),
                "Sim median": round(float(_p50), 4),
                "Direction": "real > null" if _direction == "greater" else "real < null",
                "p-value": round(_p_mean, 4),
                "n_sheep": len(_real_assay),
            })
        _ax.set_title(_title, fontsize=10)
        _ax.set_xlabel("Assay")
        _ax.legend(loc="best", fontsize=7)

    # Bottom-right: cumulative discovery curve per assay
    _ax_d = _axes[1, 1]
    _n_sites = 12
    _t_grid = np.linspace(0, 40, 81)  # 0–40 min, 0.5 min steps

    def _cum_curve(list_of_first_times, t_grid):
        if not list_of_first_times:
            return np.zeros_like(t_grid)
        per = []
        for first_times in list_of_first_times:
            if not first_times:
                per.append(np.zeros_like(t_grid))
                continue
            ft = np.asarray(first_times, dtype=float)
            per.append((ft[None, :] <= t_grid[:, None]).sum(axis=1))
        return np.mean(np.stack(per, axis=0), axis=0)

    _colours = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#999999"]
    for _ci, _assay in enumerate(_assays):
        _col = _colours[_ci % len(_colours)]
        _bucket = discovery_curves.get(_assay, {"real": [], "sim": []})
        _real_curve = _cum_curve(_bucket["real"], _t_grid)
        _ax_d.plot(_t_grid, _real_curve, color=_col, lw=2.0,
                   label=f"{_assay} real")
        if _bucket["sim"]:
            _sim_curves = np.stack(
                [_cum_curve([ft], _t_grid) for ft in _bucket["sim"]], axis=0,
            )
            _sp05, _sp50, _sp95 = np.percentile(_sim_curves, [5, 50, 95], axis=0)
            _ax_d.fill_between(
                _t_grid, _sp05, _sp95, color=_col, alpha=0.15,
            )
            _ax_d.plot(_t_grid, _sp50, color=_col, lw=1.0, ls="--", alpha=0.7)

            # p-value for sites_found at the window midpoint
            _t_idx = int(np.argmin(np.abs(_t_grid - _t_eval)))
            _real_mid = float(_real_curve[_t_idx])
            _sim_mid = _sim_curves[:, _t_idx]
            _p = float(np.mean(_sim_mid >= _real_mid)) if len(_sim_mid) else float("nan")
            _pval_rows.append({
                "Assay": _assay,
                "Metric": f"sites_found_t{_t_eval:g}",
                "Real median": round(_real_mid, 3),
                "Sim median": round(float(np.median(_sim_mid)), 3),
                "Direction": "real > null",
                "p-value": round(_p, 4),
                "n_sheep": int(len(_bucket["real"])),
            })

    _ax_d.set_title("Sites found by time (mean over sheep)", fontsize=10)
    _ax_d.set_xlabel("Time (min)")
    _ax_d.set_ylabel("Mean sites found")
    _ax_d.set_ylim(0, _n_sites)
    _ax_d.set_xlim(_t_start_plot, min(_t_end_plot + 0.5, 40.0))
    _ax_d.axvline(_t_eval, color="grey", lw=0.6, ls=":", alpha=0.7)
    _ax_d.legend(loc="best", fontsize=7, ncol=2)

    _fig.suptitle(
        "Random-walk null vs real sheep — per-sheep movement-matched simulations",
        fontsize=12,
    )
    _fig.tight_layout()

    _pval_df = pd.DataFrame(_pval_rows)

    mo.vstack([
        _fig,
        mo.md("### Empirical p-values (directional)"),
        mo.ui.table(_pval_df),
    ])
    return


if __name__ == "__main__":
    app.run()
