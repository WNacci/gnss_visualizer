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

    rng = np.random.default_rng(seed=42)
    return (
        np, pd, plt, mo, matplotlib,
        load_trial_tracks, detect_site_visits, cumulative_path_length,
        SITE_GRID, DATA_DIR,
        TRIALS, TRACKS_CACHE,
        KEEP_CONFIGS, K_DEFAULT, ARENA_LO, ARENA_HI, COVERAGE_BIN,
        PHASE2_DATE, SITE_RADIUS,
        rng,
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

    {K_slider}
    {phase_dd}
    """)
    return K_slider, phase_dd


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
    def simulate_walk(start_xy, n_steps, steps_emp, turns_emp, rng):
        """Correlated random walk with reflective boundary on [ARENA_LO, ARENA_HI]^2.

        Parameters
        ----------
        start_xy : tuple (x0, y0).
        n_steps : int, number of segments to simulate (returns n_steps+1 samples).
        steps_emp : ndarray of empirical step lengths to resample from.
        turns_emp : ndarray of empirical turn angles to resample from.
        rng : numpy Generator.
        """
        if len(steps_emp) == 0 or n_steps <= 0:
            x0, y0 = start_xy
            return np.array([x0], dtype=float), np.array([y0], dtype=float)

        if len(turns_emp) == 0:
            turns_emp = np.array([0.0])

        gx = np.empty(n_steps + 1, dtype=float)
        gy = np.empty(n_steps + 1, dtype=float)
        gx[0], gy[0] = start_xy

        sampled_steps = rng.choice(steps_emp, size=n_steps)
        sampled_turns = rng.choice(turns_emp, size=n_steps)
        heading = rng.uniform(-np.pi, np.pi)

        for i in range(n_steps):
            heading = heading + sampled_turns[i]
            step = sampled_steps[i]
            x = gx[i] + step * np.cos(heading)
            y = gy[i] + step * np.sin(heading)

            # One-pass reflective fold (typical step << arena width)
            if x < ARENA_LO:
                x = 2 * ARENA_LO - x
                heading = np.pi - heading
            elif x > ARENA_HI:
                x = 2 * ARENA_HI - x
                heading = np.pi - heading
            if y < ARENA_LO:
                y = 2 * ARENA_LO - y
                heading = -heading
            elif y > ARENA_HI:
                y = 2 * ARENA_HI - y
                heading = -heading

            # Clamp in the rare event of a step > arena width
            x = min(max(x, ARENA_LO), ARENA_HI)
            y = min(max(y, ARENA_LO), ARENA_HI)
            gx[i + 1] = x
            gy[i + 1] = y

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


if __name__ == "__main__":
    app.run()
