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


if __name__ == "__main__":
    app.run()
