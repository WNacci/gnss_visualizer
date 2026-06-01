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
        SITE_GRID, BAITED_CANONICAL, UNBAITED_CANONICAL, DATA_DIR,
    )

    BAITED_SITES = {k: SITE_GRID[k] for k in BAITED_CANONICAL}
    UNBAITED_SITES = {k: SITE_GRID[k] for k in UNBAITED_CANONICAL}
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
        SITE_GRID, BAITED_SITES, UNBAITED_SITES, DATA_DIR,
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
        value="Phase 2 (>=2026-02-17)",
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

    def time_at_sites(gx, gy, sites=SITE_GRID, radius=SITE_RADIUS):
        """Fraction of timesteps within radius of ANY of the passed sites.

        Used in canonical-oriented frame with either BAITED_SITES (3),
        UNBAITED_SITES (9), or SITE_GRID (all 12). Real sheep should
        beat a movement-matched random walker on time_at(BAITED) because
        they navigate to reward locations, but should be near chance on
        time_at(UNBAITED).
        """
        if len(gx) == 0:
            return 0.0
        gx = np.asarray(gx, dtype=float)
        gy = np.asarray(gy, dtype=float)
        in_any = np.zeros(len(gx), dtype=bool)
        for _label, (sx, sy) in sites.items():
            in_any |= np.hypot(gx - sx, gy - sy) <= radius
        return float(in_any.mean())
    return (
        coverage, revisit_rate, straightness,
        sites_found_by_time, time_at_sites,
    )


@app.cell(hide_code=True)
def _(np, ARENA_LO, ARENA_HI, COVERAGE_BIN, SITE_GRID, SITE_RADIUS):
    """Batched (K-walk) simulator and metrics — single numpy op per metric per sheep."""
    _N_BINS_B = int(round((ARENA_HI - ARENA_LO) / COVERAGE_BIN))

    def _reflect_b(arr, lo, hi):
        span = hi - lo
        u = (arr - lo) % (2 * span)
        return lo + np.where(u <= span, u, 2 * span - u)

    def simulate_walks_batch(start_xy, n_steps, steps_emp, turns_emp, K, rng):
        """Run K correlated random walks in parallel.

        Returns gx, gy of shape (K, n_steps + 1). Reuses the empirical pools
        IID across walks (turn-angle order within a walk is preserved via
        cumsum, matching the single-walk simulator).
        """
        x0, y0 = start_xy
        if len(steps_emp) == 0 or n_steps <= 0 or K <= 0:
            return (np.full((max(K, 1), 1), x0, dtype=float),
                    np.full((max(K, 1), 1), y0, dtype=float))
        if len(turns_emp) == 0:
            turns_emp = np.array([0.0])
        sampled_steps = rng.choice(steps_emp, size=(K, n_steps))
        sampled_turns = rng.choice(turns_emp, size=(K, n_steps))
        heading0 = rng.uniform(-np.pi, np.pi, size=K)
        headings = heading0[:, None] + np.cumsum(sampled_turns, axis=1)
        dx = sampled_steps * np.cos(headings)
        dy = sampled_steps * np.sin(headings)
        x_unb = np.concatenate(
            [np.full((K, 1), x0), x0 + np.cumsum(dx, axis=1)], axis=1,
        )
        y_unb = np.concatenate(
            [np.full((K, 1), y0), y0 + np.cumsum(dy, axis=1)], axis=1,
        )
        return (
            _reflect_b(x_unb, ARENA_LO, ARENA_HI),
            _reflect_b(y_unb, ARENA_LO, ARENA_HI),
        )

    _N_CELLS_B = _N_BINS_B * _N_BINS_B

    def coverage_batch(gx, gy):
        K, T = gx.shape
        if T == 0:
            return np.zeros(K, dtype=int)
        ix = np.clip(((gx - ARENA_LO) / COVERAGE_BIN).astype(int), 0, _N_BINS_B - 1)
        iy = np.clip(((gy - ARENA_LO) / COVERAGE_BIN).astype(int), 0, _N_BINS_B - 1)
        # Encode (walk, cell) as walk*N + cell, then one big bincount over all K walks.
        k_off = np.arange(K, dtype=np.int64)[:, None] * _N_CELLS_B
        encoded = (k_off + ix * _N_BINS_B + iy).ravel()
        counts = np.bincount(encoded, minlength=K * _N_CELLS_B).reshape(K, _N_CELLS_B)
        return (counts > 0).sum(axis=1)

    def revisit_rate_batch(gx, gy, cov=None):
        if cov is None:
            cov = coverage_batch(gx, gy)
        return gx.shape[1] / np.maximum(cov, 1).astype(float)

    def straightness_batch(gx, gy):
        if gx.shape[1] < 2:
            return np.zeros(gx.shape[0])
        d = np.hypot(gx[:, -1] - gx[:, 0], gy[:, -1] - gy[:, 0])
        path = np.sum(np.hypot(np.diff(gx, axis=1), np.diff(gy, axis=1)), axis=1)
        return np.where(path > 0, d / path, 0.0)

    def sites_found_by_time_batch(gx, gy, t, sites=SITE_GRID, radius=SITE_RADIUS):
        """First-visit times per walk for each canonical site. Returns list of K sorted lists."""
        K = gx.shape[0]
        t_arr = np.asarray(t, dtype=float)
        first_times_per_k = [[] for _ in range(K)]
        for _label, (sx, sy) in sites.items():
            dist = np.hypot(gx - sx, gy - sy)
            inside = dist <= radius
            any_inside = inside.any(axis=1)
            first_idx = inside.argmax(axis=1)
            for k in range(K):
                if any_inside[k]:
                    first_times_per_k[k].append(float(t_arr[first_idx[k]]))
        for ft in first_times_per_k:
            ft.sort()
        return first_times_per_k

    def time_at_sites_batch(gx, gy, sites=SITE_GRID, radius=SITE_RADIUS):
        """Per-walk fraction of timesteps within radius of any of the passed sites. Shape (K,)."""
        K, T = gx.shape
        if T == 0:
            return np.zeros(K)
        in_any = np.zeros((K, T), dtype=bool)
        for _label, (sx, sy) in sites.items():
            in_any |= np.hypot(gx - sx, gy - sy) <= radius
        return in_any.mean(axis=1)

    return (
        simulate_walks_batch, coverage_batch, revisit_rate_batch,
        straightness_batch, sites_found_by_time_batch, time_at_sites_batch,
    )


@app.cell(hide_code=True)
def _(
    np, pd, mo,
    TRIALS, TRACKS_CACHE, load_trial_tracks,
    KEEP_CONFIGS, PHASE2_DATE,
    BAITED_SITES, UNBAITED_SITES,
    fit_movement_stats,
    coverage, revisit_rate, straightness, sites_found_by_time, time_at_sites,
    simulate_walks_batch, coverage_batch, revisit_rate_batch,
    straightness_batch, sites_found_by_time_batch, time_at_sites_batch,
    K_slider, phase_dd, time_window_slider,
):
    # Decimate 10 Hz tracks to 1 Hz before fitting/simulating. Both real and
    # simulated trajectories are decimated identically so the null comparison
    # stays internally consistent. ~20× speedup over running at full rate.
    _DECIMATE = 10
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

            # NOTE: this groups by *configuration*, not by assay number.
            # `_config_group` is "A"/"B"/"C"/"D" for test trials or "CTRL" for
            # control trials. Real assay is `_trial["assay"]` (per-group trial
            # sequence) and isn't used here.
            _config_group = "CTRL" if _cfg.startswith("CTRL") else _cfg
            _bucket = discovery_curves.setdefault(_config_group, {"real": [], "sim": []})
            _n_trials_used += 1

            for _sheep_id, _trk in _tracks.items():
                _gx_full = np.asarray(_trk["gx"], dtype=float)
                _gy_full = np.asarray(_trk["gy"], dtype=float)
                _t_full = np.asarray(_trk["t"], dtype=float)
                _mask = (_t_full >= _t_start) & (_t_full <= _t_end)
                # Decimate from 10 Hz → 1 Hz: ~20× faster, plenty of resolution
                # for a null model. Real and simulated tracks get the same
                # treatment so the comparison stays internally consistent.
                _gx = _gx_full[_mask][::_DECIMATE]
                _gy = _gy_full[_mask][::_DECIMATE]
                _t = _t_full[_mask][::_DECIMATE]
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
                _real_tas = time_at_sites(_gx, _gy)
                _real_tab = time_at_sites(_gx, _gy, sites=BAITED_SITES)
                _real_tau = time_at_sites(_gx, _gy, sites=UNBAITED_SITES)
                real_records.append({
                    "config": _config_group,
                    "trial_idx": _tidx,
                    "sheep": _sheep_id,
                    "coverage": _real_cov,
                    "revisit": _real_rev,
                    "straightness": _real_str,
                    "n_sites": len(_real_sites),
                    "time_at_sites": _real_tas,
                    "time_at_baited": _real_tab,
                    "time_at_unbaited": _real_tau,
                })
                _bucket["real"].append(_real_sites)

                _n_steps = len(_gx) - 1
                _start = (float(_gx[0]), float(_gy[0]))
                _sgx_all, _sgy_all = simulate_walks_batch(
                    _start, _n_steps, _steps_emp, _turns_emp, K, _sim_rng,
                )
                # Reuse the empirical time-vector for simulated walks so site
                # discovery curves are directly comparable.
                _st = _t[: _sgx_all.shape[1]]
                _sim_covs = coverage_batch(_sgx_all, _sgy_all)
                _sim_revs = revisit_rate_batch(_sgx_all, _sgy_all, cov=_sim_covs)
                _sim_strs = straightness_batch(_sgx_all, _sgy_all)
                _sim_sites_list = sites_found_by_time_batch(
                    _sgx_all, _sgy_all, _st,
                )
                _sim_tas = time_at_sites_batch(_sgx_all, _sgy_all)
                _sim_tab = time_at_sites_batch(_sgx_all, _sgy_all, sites=BAITED_SITES)
                _sim_tau = time_at_sites_batch(_sgx_all, _sgy_all, sites=UNBAITED_SITES)
                for _k in range(K):
                    sim_records.append({
                        "config": _config_group,
                        "trial_idx": _tidx,
                        "sheep": _sheep_id,
                        "k": _k,
                        "coverage": int(_sim_covs[_k]),
                        "revisit": float(_sim_revs[_k]),
                        "straightness": float(_sim_strs[_k]),
                        "n_sites": len(_sim_sites_list[_k]),
                        "time_at_sites": float(_sim_tas[_k]),
                        "time_at_baited": float(_sim_tab[_k]),
                        "time_at_unbaited": float(_sim_tau[_k]),
                    })
                    _bucket["sim"].append(_sim_sites_list[_k])

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
        f"**Configs present:** {sorted(real_df['config'].unique()) if len(real_df) else []}",
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

    _configs = sorted(real_df["config"].unique())
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
        _real_vals = [real_df.loc[real_df["config"] == c, _metric].values
                      for c in _configs]
        _bp = _ax.boxplot(
            _real_vals, tick_labels=_configs, widths=0.5,
            patch_artist=True,
        )
        for _patch in _bp["boxes"]:
            _patch.set_facecolor("#377eb8")
            _patch.set_alpha(0.55)

        for _i, _cfg in enumerate(_configs, start=1):
            _sim_vals = sim_df.loc[sim_df["config"] == _cfg, _metric].values
            _real_cfg = real_df.loc[real_df["config"] == _cfg, _metric].values
            if len(_sim_vals) == 0 or len(_real_cfg) == 0:
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
            # (avg p across all real points for the config).
            _p_per_real = []
            for _rv in _real_cfg:
                if _direction == "greater":
                    _p = float(np.mean(_sim_vals >= _rv))
                else:
                    _p = float(np.mean(_sim_vals <= _rv))
                _p_per_real.append(_p)
            _p_mean = float(np.mean(_p_per_real)) if _p_per_real else float("nan")
            _pval_rows.append({
                "Configuration": _cfg,
                "Metric": _metric,
                "Real median": round(float(np.median(_real_cfg)), 4),
                "Sim median": round(float(_p50), 4),
                "Direction": "real > null" if _direction == "greater" else "real < null",
                "p-value": round(_p_mean, 4),
                "n_sheep": len(_real_cfg),
            })
        _ax.set_title(_title, fontsize=10)
        _ax.set_xlabel("Configuration")
        _ax.legend(loc="best", fontsize=7)

    # Bottom-right: cumulative discovery curve per configuration
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
    for _ci, _cfg in enumerate(_configs):
        _col = _colours[_ci % len(_colours)]
        _bucket = discovery_curves.get(_cfg, {"real": [], "sim": []})
        _real_curve = _cum_curve(_bucket["real"], _t_grid)
        _ax_d.plot(_t_grid, _real_curve, color=_col, lw=2.0,
                   label=f"{_cfg} real")
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
                "Configuration": _cfg,
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


@app.cell(hide_code=True)
def _(np, pd, plt, mo, real_df, sim_df):
    """Diagnostic: real sheep overlaid on per-assay sim distributions.

    Grey violins are the simulated null per assay; blue dots are individual
    real sheep with horizontal jitter. Real dots clustering outside the violin
    bulk indicates the metric distinguishes real movement from a movement-
    matched random walker. Two-sided empirical p-values, the median z-score
    (real − sim_median)/sim_std per sheep, and the % of real points outside
    the sim 5–95% envelope are reported alongside.
    """
    mo.stop(len(real_df) == 0, mo.md("*No sheep matched the current filter — diagnostic skipped.*"))

    _configs_d = sorted(real_df["config"].unique())
    _metrics_d = [
        ("coverage", "Coverage (unique cells)"),
        ("revisit", "Revisit rate (samples/cell)"),
        ("straightness", "Straightness (disp/path)"),
        ("time_at_baited", "Time at BAITED sites (3 sites)"),
        ("time_at_unbaited", "Time at UNBAITED sites (9 sites)"),
    ]

    _fig_d, _axes_d = plt.subplots(1, 5, figsize=(22, 5))
    _summary_rows = []
    _jitter_rng = np.random.default_rng(42)

    for _ax_diag, (_metric_d, _label_d) in zip(_axes_d, _metrics_d):
        _sim_per_cfg = [
            sim_df.loc[sim_df["config"] == _c, _metric_d].values
            for _c in _configs_d
        ]
        _real_per_cfg = [
            real_df.loc[real_df["config"] == _c, _metric_d].values
            for _c in _configs_d
        ]

        _vp = _ax_diag.violinplot(
            _sim_per_cfg,
            positions=range(1, len(_configs_d) + 1),
            widths=0.75, showmedians=True, showextrema=False,
        )
        for _body in _vp["bodies"]:
            _body.set_facecolor("#cccccc")
            _body.set_edgecolor("#888888")
            _body.set_alpha(0.55)
        if "cmedians" in _vp:
            _vp["cmedians"].set_color("#444444")
            _vp["cmedians"].set_linewidth(1.2)

        for _i, (_c, _sv, _rv) in enumerate(
            zip(_configs_d, _sim_per_cfg, _real_per_cfg), start=1,
        ):
            if len(_sv) == 0 or len(_rv) == 0:
                continue
            _jit = _jitter_rng.uniform(-0.18, 0.18, size=len(_rv))
            _ax_diag.scatter(
                _i + _jit, _rv, s=14,
                color="#377eb8", alpha=0.7,
                edgecolor="white", linewidth=0.4, zorder=3,
            )
            _sim_med = float(np.median(_sv))
            _sim_std = float(np.std(_sv))
            _sim_p05, _sim_p95 = np.percentile(_sv, [5, 95])
            _z_per = (_rv - _sim_med) / max(_sim_std, 1e-9)
            _outside = float(
                np.mean((_rv < _sim_p05) | (_rv > _sim_p95)),
            )
            _p_two = float(np.mean([
                2.0 * min(
                    float(np.mean(_sv <= _r)),
                    float(np.mean(_sv >= _r)),
                )
                for _r in _rv
            ]))
            _summary_rows.append({
                "Configuration": _c,
                "Metric": _metric_d,
                "Real median": round(float(np.median(_rv)), 3),
                "Sim median": round(_sim_med, 3),
                "Median z": round(float(np.median(_z_per)), 2),
                "% real outside sim 5–95%": round(100 * _outside, 1),
                "two-sided p": round(_p_two, 4),
                "n_sheep": int(len(_rv)),
            })

        _ax_diag.set_xticks(range(1, len(_configs_d) + 1))
        _ax_diag.set_xticklabels(_configs_d)
        _ax_diag.set_xlabel("Configuration")
        _ax_diag.set_title(_label_d, fontsize=10)

    _fig_d.suptitle(
        "Diagnostic: real sheep (blue) vs simulated-walk distribution (grey)",
        fontsize=12,
    )
    _fig_d.tight_layout()

    _summary_df = pd.DataFrame(_summary_rows)
    mo.vstack([
        mo.md("### Diagnostic: real values vs simulated-walk distributions"),
        mo.md(
            "Median z = (real − sim_median) / sim_std (per sheep, then median "
            "across sheep). |z|>1 ≈ real differs from null by ≥1 sim SD."
        ),
        _fig_d,
        mo.ui.table(_summary_df),
    ])
    return


@app.cell(hide_code=True)
def _(np, pd, plt, mo, real_df, sim_df):
    """Baited preference — the most direct test of 'sheep navigate to reward'.

    For each (sheep-trial, walk), measure the fraction of site-visit time
    that lands on baited rather than unbaited canonical positions:

        baited_fraction = time_at_baited / (time_at_baited + time_at_unbaited)

    A random walker should land at chance ≈ 3/12 = 0.25 (3 of 12 sites are
    baited). Real sheep should be well above 0.25 if they navigate to
    reward. CTRL trials are excluded from the test population since they
    don't have a meaningful baited triplet in the canonical frame
    (orientation transforms map test-config bait → A-prefix positions;
    CTRL trials don't get the same transform).
    """
    mo.stop(
        len(real_df) == 0,
        mo.md("*No sheep matched the current filter — baited preference skipped.*"),
    )

    def _frac(df):
        _tb = df["time_at_baited"].to_numpy()
        _tu = df["time_at_unbaited"].to_numpy()
        _denom = _tb + _tu
        _valid = _denom > 0
        _out = np.full_like(_tb, np.nan, dtype=float)
        _out[_valid] = _tb[_valid] / _denom[_valid]
        return _out

    _real_bp = real_df.assign(baited_fraction=_frac(real_df))
    _sim_bp = sim_df.assign(baited_fraction=_frac(sim_df))

    # Drop CTRL from this panel — see docstring.
    _test_configs = [_c for _c in sorted(_real_bp["config"].unique()) if _c != "CTRL"]
    _chance = 3.0 / 12.0  # 3 baited of 12 total canonical sites

    _real_per = [
        _real_bp.loc[_real_bp["config"] == _c, "baited_fraction"].dropna().to_numpy()
        for _c in _test_configs
    ]
    _sim_per = [
        _sim_bp.loc[_sim_bp["config"] == _c, "baited_fraction"].dropna().to_numpy()
        for _c in _test_configs
    ]

    _fig_bp, (_ax_bp, _ax_dist) = plt.subplots(1, 2, figsize=(15, 5))

    # Left panel: per-config violin (sim) + scatter (real).
    _vp_bp = _ax_bp.violinplot(
        _sim_per, positions=range(1, len(_test_configs) + 1),
        widths=0.75, showmedians=True, showextrema=False,
    )
    for _body in _vp_bp["bodies"]:
        _body.set_facecolor("#cccccc")
        _body.set_edgecolor("#888888")
        _body.set_alpha(0.55)
    if "cmedians" in _vp_bp:
        _vp_bp["cmedians"].set_color("#444444")
        _vp_bp["cmedians"].set_linewidth(1.2)

    _jit_rng_bp = np.random.default_rng(0)
    _rows_bp = []
    for _i, (_c, _sv, _rv) in enumerate(zip(_test_configs, _sim_per, _real_per), start=1):
        if len(_sv) == 0 or len(_rv) == 0:
            continue
        _jit = _jit_rng_bp.uniform(-0.18, 0.18, size=len(_rv))
        _ax_bp.scatter(
            _i + _jit, _rv, s=18,
            color="#d7301f", alpha=0.7,
            edgecolor="white", linewidth=0.4, zorder=3,
        )
        _sim_med = float(np.median(_sv))
        _real_med = float(np.median(_rv))
        _sim_std = float(np.std(_sv))
        _z = (_real_med - _sim_med) / max(_sim_std, 1e-9)
        # One-sided "real > sim" empirical p, per sheep then mean.
        _p = float(np.mean([float(np.mean(_sv >= _r)) for _r in _rv]))
        _rows_bp.append({
            "Configuration": _c,
            "Real median": round(_real_med, 3),
            "Sim median": round(_sim_med, 3),
            "Chance (3/12)": round(_chance, 3),
            "Real − Chance": round(_real_med - _chance, 3),
            "z (real − sim)/σ_sim": round(_z, 2),
            "p (real > null)": round(_p, 4),
            "n_sheep": int(len(_rv)),
        })

    _ax_bp.axhline(_chance, color="#377eb8", lw=1.2, ls="--", alpha=0.7,
                   label=f"Spatial chance ({_chance:.2f})")
    _ax_bp.set_xticks(range(1, len(_test_configs) + 1))
    _ax_bp.set_xticklabels(_test_configs)
    _ax_bp.set_xlabel("Configuration")
    _ax_bp.set_ylabel("baited / (baited + unbaited) site-time")
    _ax_bp.set_title("Baited preference per config\n(grey = sim, red = real sheep)", fontsize=10)
    _ax_bp.set_ylim(-0.05, 1.05)
    _ax_bp.legend(loc="best", fontsize=8)

    # Right panel: pooled distribution real vs sim across all test configs.
    _real_pool = _real_bp.loc[
        _real_bp["config"].isin(_test_configs), "baited_fraction"
    ].dropna().to_numpy()
    _sim_pool = _sim_bp.loc[
        _sim_bp["config"].isin(_test_configs), "baited_fraction"
    ].dropna().to_numpy()
    if len(_real_pool) > 0 and len(_sim_pool) > 0:
        _bins_bp = np.linspace(0.0, 1.0, 31)
        _ax_dist.hist(
            _sim_pool, bins=_bins_bp, density=True,
            color="#888888", alpha=0.55, label=f"Sim (n={len(_sim_pool)})",
        )
        _ax_dist.hist(
            _real_pool, bins=_bins_bp, density=True,
            color="#d7301f", alpha=0.6, label=f"Real (n={len(_real_pool)})",
        )
        _ax_dist.axvline(_chance, color="#377eb8", lw=1.2, ls="--",
                         label=f"Chance ({_chance:.2f})")
        _ax_dist.axvline(float(np.median(_real_pool)), color="#7a0a0a", lw=1.5)
        _ax_dist.axvline(float(np.median(_sim_pool)), color="#444444", lw=1.5)
    _ax_dist.set_xlabel("baited / (baited + unbaited) site-time")
    _ax_dist.set_ylabel("density")
    _ax_dist.set_title("Pooled across A/B/C/D", fontsize=10)
    _ax_dist.legend(loc="best", fontsize=8)

    _fig_bp.suptitle(
        "Baited preference — direct test of 'sheep navigate to reward sites'",
        fontsize=12,
    )
    _fig_bp.tight_layout()

    _bp_df = pd.DataFrame(_rows_bp)
    mo.vstack([
        mo.md("### Baited vs unbaited site preference"),
        mo.md(
            "After `apply_orient=True`, every test trial's baited triplet "
            "maps to the canonical A1/A2/A3 positions, so 'baited' is a "
            "well-defined set in this frame. A random walker should hit "
            "baited at chance ≈ 3/12 = 0.25 of its site-time; sheep that "
            "navigate to reward should sit well above. CTRL configurations "
            "are excluded here because they don't share the canonical "
            "orientation."
        ),
        _fig_bp,
        mo.ui.table(_bp_df),
    ])
    return


if __name__ == "__main__":
    app.run()
