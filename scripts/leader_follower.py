"""Leader-Follower Dynamics

Identifies leadership behaviour within a group across time by measuring:

  1. **Frontal position**: which sheep is farthest ahead (in the direction of
     group motion) at each timestep.  The "leader" leads from the front.
  2. **Pioneer visits**: which sheep is the *first* to enter the vicinity of
     each reward site.  Consistent pioneers are candidates for leaders.
  3. **Site recruitment**: which sheep initiates group visits to sites?
     Visits are grouped into episodes (2-min gap = new episode); the first
     sheep to enter is the recruiter, others are followers.
  4. **Leadership consistency**: entropy of the leadership distribution.
     Low entropy → one sheep leads most of the time.

Todos addressed: #6 (leader/follower dynamics)
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
        load_trial_tracks, detect_site_visits,
        detect_recruitment_episodes, DATA_DIR,
    )

    TRIALS = build_trials()
    print(f"Loaded {len(TRIALS)} trials — building tracks cache (runs once)…")
    TRACKS_CACHE = build_tracks_cache()
    return (
        np, pd, plt, mo, matplotlib,
        build_trials, build_tracks_cache,
        load_trial_tracks, detect_site_visits,
        detect_recruitment_episodes, DATA_DIR,
        TRIALS, TRACKS_CACHE,
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
    TRACKS_CACHE, load_trial_tracks, detect_site_visits,
    DATA_DIR,
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
        _trial, tracks_cache=TRACKS_CACHE,
        apply_orient=True,
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
        _tracks, _trial['field'], radius=_radius,
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
    # Metric 3: Site recruitment episodes
    # -----------------------------------------------------------------------
    _episodes = detect_recruitment_episodes(_visits)

    # Per-sheep recruitment stats: count total followers attracted
    _recruit_total_followers = {_sid: 0 for _sid in _sheep_ids}
    _recruit_ep_counts = {_sid: 0 for _sid in _sheep_ids}
    for _ep in _episodes:
        _init = _ep['initiator']
        if _init in _recruit_total_followers:
            _recruit_total_followers[_init] += len(_ep['followers'])
            _recruit_ep_counts[_init] += 1

    _total_followers_all = sum(_recruit_total_followers.values())
    _recruit_freq = {
        _sid: _recruit_total_followers[_sid] / max(_total_followers_all, 1)
        for _sid in _sheep_ids
    }
    _recruit_mean_foll = {
        _sid: (_recruit_total_followers[_sid] / _recruit_ep_counts[_sid]
               if _recruit_ep_counts[_sid] > 0 else 0.0)
        for _sid in _sheep_ids
    }

    # Recruitment entropy (over follower share, not episode count)
    _rf = np.array([_recruit_freq[_sid] for _sid in _sheep_ids])
    _rf_pos = _rf[_rf > 0]
    _recruit_entropy = float(-np.sum(_rf_pos * np.log(_rf_pos))) if len(_rf_pos) > 0 else 0.0
    _recruit_norm_entropy = _recruit_entropy / _max_entropy  # reuse from frontal

    _episode_rows = []
    for _ep in _episodes:
        _episode_rows.append({
            'Site': _ep['site'],
            'Initiator': _ep['initiator'],
            'Followers': len(_ep['followers']),
            'Entry time (min)': round(_ep['time'], 2),
        })

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    COLORS = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#a65628', '#f781bf', '#999999',
    ]

    _fig, _axes = plt.subplots(3, 2, figsize=(14, 12))
    _ax_traj, _ax_lead_ts, _ax_frac, _ax_pioneer, _ax_recruit, _ax_ep_timeline = _axes.flatten()

    # --- Trajectories coloured by animal ---
    for _ci, _sid in enumerate(_sheep_ids):
        _ax_traj.plot(_gx_all[_ci], _gy_all[_ci], color=COLORS[_ci % len(COLORS)],
                      alpha=0.5, lw=0.8, label=_sid)
        _ax_traj.scatter(
            _gx_all[_ci, 0], _gy_all[_ci, 0],
            color=COLORS[_ci % len(COLORS)], s=60, zorder=5, marker='o',
        )
    _ax_traj.plot(_cx, _cy, color='k', lw=1, ls='--', label='centroid')

    # Arena styling — mirrors occupancy_heatmap layout.
    # Tracks were loaded with apply_orient=True so all configs are normalised
    # to the canonical orientation where A-prefix sites are the baited ones.
    _ax_traj.set_xlim(-0.05, 5.05)
    _ax_traj.set_ylim(-0.05, 5.05)
    _ax_traj.set_aspect('equal')
    _ax_traj.set_facecolor('#c8c8c8')

    # Faint grid lines at integer positions
    for _v in range(1, 5):
        _ax_traj.axvline(_v, color='white', alpha=0.2, linewidth=0.5)
        _ax_traj.axhline(_v, color='white', alpha=0.2, linewidth=0.5)

    # Reward site overlay (skip assay 0 and control configs)
    _BAITED_COLOR   = '#FFE066'
    _UNBAITED_COLOR = '#888888'
    _is_ctrl = _trial['config'] not in ('A', 'B', 'C', 'D')
    if str(_trial['assay']) != '0' and not _is_ctrl:
        _sites_plot = _rdf[
            (_rdf['field'] == 'A') & (~_rdf['label'].str.startswith('E'))
        ]
        for _, _r in _sites_plot.iterrows():
            # After orient transform, baited sites always land on A-prefix positions
            _ec = _BAITED_COLOR if _r['label'].startswith('A') else _UNBAITED_COLOR
            _ax_traj.scatter(
                _r['grid_x'], _r['grid_y'],
                s=100, zorder=6, marker='o',
                facecolors='none', edgecolors=_ec, linewidths=1.8,
            )
            _ax_traj.annotate(
                _r['label'], (_r['grid_x'], _r['grid_y']),
                xytext=(4, 4), textcoords='offset points',
                fontsize=7, color=_ec, fontweight='bold',
            )

    _ax_traj.set_xlabel("Grid x (10 m/unit)")
    _ax_traj.set_ylabel("Grid y (10 m/unit)")
    _ax_traj.set_title("Trajectories (circle = start, oriented)")
    _ax_traj.legend(fontsize=8)

    # --- Leader time series: fraction of each 1-min bin spent as frontal leader ---
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

    # --- Recruitment: total followers attracted per sheep ---
    _recruit_vals = [_recruit_total_followers[k] for k in _sheep_ids]
    _ax_recruit.bar(
        range(_n_sheep), _recruit_vals,
        color=[COLORS[_i % len(COLORS)] for _i in range(_n_sheep)],
        tick_label=_sheep_ids,
    )
    # Annotate episode count above each bar
    for _bi, _sid in enumerate(_sheep_ids):
        _ec = _recruit_ep_counts[_sid]
        if _recruit_vals[_bi] > 0:
            _ax_recruit.text(
                _bi, _recruit_vals[_bi] + 0.15, f"({_ec} ep)",
                ha='center', va='bottom', fontsize=7, color='#555555',
            )
    _ax_recruit.set_ylabel("Total followers recruited")
    _ax_recruit.set_title(
        f"Site recruitment (entropy={_recruit_norm_entropy:.2f})\n"
        f"Numbers = episodes initiated"
    )

    # --- Episode timeline ---
    _sid_to_idx = {_sid: _i for _i, _sid in enumerate(_sheep_ids)}
    for _site, _init, _foll, _t_ep in _episodes:
        _ci = _sid_to_idx.get(_init, 0)
        _size = 30 + len(_foll) * 40  # scale dot by follower count
        _ax_ep_timeline.scatter(
            _t_ep, _ci, s=_size, alpha=0.6,
            color=COLORS[_ci % len(COLORS)], edgecolors='k', linewidths=0.3,
        )
    _ax_ep_timeline.set_yticks(range(_n_sheep))
    _ax_ep_timeline.set_yticklabels(_sheep_ids, fontsize=8)
    _ax_ep_timeline.set_xlabel("Time (min)")
    _ax_ep_timeline.set_ylabel("Initiator sheep")
    _ax_ep_timeline.set_title("Episode timeline (dot size = # followers)")
    _ax_ep_timeline.set_xlim(0, _dur)

    _fig.suptitle(f"Leader-follower dynamics — {_trial['name']}", fontsize=11)
    _fig.tight_layout()

    _pioneer_df = pd.DataFrame(_pioneer_rows) if _pioneer_rows else pd.DataFrame(
        columns=['Site', 'Pioneer sheep', 'Entry time (min)']
    )
    _episode_df = pd.DataFrame(_episode_rows) if _episode_rows else pd.DataFrame(
        columns=['Site', 'Initiator', 'Followers', 'Entry time (min)']
    )

    mo.vstack([
        _fig,
        mo.md(
            f"**Frontal leadership entropy:** {_norm_entropy:.3f}  "
            f"(0 = one sheep always leads; 1 = equally shared)  \n"
            f"**Recruitment entropy:** {_recruit_norm_entropy:.3f}  "
            f"({_total_episodes} episodes across {sum(1 for v in _visits.values() if v)} sites)  \n"
            f"**Pioneer visits:** {sum(_pioneer_vals)} sites visited"
        ),
        mo.md("### Pioneer visit events"),
        mo.ui.table(_pioneer_df),
        mo.md("### Site recruitment episodes"),
        mo.ui.table(_episode_df),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## Aggregate: leadership consistency across trials by assay")
    return


@app.cell(hide_code=True)
def _(
    TRIALS, mo,
    TRACKS_CACHE, load_trial_tracks, detect_site_visits,
    detect_recruitment_episodes, DATA_DIR,
    np, pd, plt,
):
    """Compute leadership and recruitment entropy for every multi-sheep test trial."""
    _TEST_CONFIGS = {'A', 'B', 'C', 'D'}
    _rdf_agg = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
    _records = []

    for _tidx, _trial in enumerate(TRIALS):
        if _trial['group_size'] < 2:
            continue
        if _trial['config'] not in _TEST_CONFIGS:
            continue

        _tracks = load_trial_tracks(
            _trial, tracks_cache=TRACKS_CACHE,
            apply_orient=True,
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

        # --- Frontal position leadership ---
        _cx = _gx_all.mean(axis=0)
        _cy = _gy_all.mean(axis=0)
        _agg_kernel = np.ones(15) / 15
        _vcx = np.convolve(np.gradient(_cx, _t_common), _agg_kernel, mode='same')
        _vcy = np.convolve(np.gradient(_cy, _t_common), _agg_kernel, mode='same')
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
        _max_ent = np.log(_n_sheep) if _n_sheep > 1 else 1.0
        _norm = _entropy / _max_ent

        # --- Site recruitment ---
        _visits = detect_site_visits(
            _tracks, _trial['field'], radius=0.5,
            reward_sites_df=_rdf_agg,
        )
        _agg_episodes = detect_recruitment_episodes(_visits)
        _recruit_foll = {_sid: 0 for _sid in _sheep_ids}
        for _ep in _agg_episodes:
            if _ep['initiator'] in _recruit_foll:
                _recruit_foll[_ep['initiator']] += len(_ep['followers'])

        _total_foll = sum(_recruit_foll.values())
        _total_ep = len(_agg_episodes)
        _rf = np.array([_recruit_foll[_sid] / max(_total_foll, 1) for _sid in _sheep_ids])
        _rf_pos = _rf[_rf > 0]
        _r_ent = float(-np.sum(_rf_pos * np.log(_rf_pos))) if len(_rf_pos) > 0 else 0.0
        _r_norm = _r_ent / _max_ent

        _records.append({
            'Trial': _tidx,
            'Date': _trial['date'],
            'Config': _trial['config'],
            'Group size': _trial['group_size'],
            'Assay': str(_trial['assay']),
            'Frontal entropy': round(_norm, 3),
            'Dominant leader frac.': round(float(_lf.max()), 3),
            'Recruitment entropy': round(_r_norm, 3),
            'Top recruiter frac.': round(float(_rf.max()), 3),
            'Episodes': _total_ep,
        })

    if not _records:
        mo.stop(True, mo.md("*No multi-sheep test trials with GPS data found.*"))

    _agg_df = pd.DataFrame(_records)

    _fig2, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(15, 4))
    _assays = sorted(_agg_df['Assay'].unique(), key=lambda x: (not x.isdigit(), x))

    _ent_by_assay = [_agg_df[_agg_df['Assay'] == a]['Frontal entropy'].dropna().values
                     for a in _assays]
    _dom_by_assay = [_agg_df[_agg_df['Assay'] == a]['Dominant leader frac.'].dropna().values
                     for a in _assays]
    _rec_by_assay = [_agg_df[_agg_df['Assay'] == a]['Recruitment entropy'].dropna().values
                     for a in _assays]

    _ax1.boxplot(_ent_by_assay, labels=_assays)
    _ax1.set_xlabel("Assay")
    _ax1.set_ylabel("Normalised entropy")
    _ax1.set_title("Frontal leadership entropy\n(low = one sheep leads)")
    _ax1.set_ylim(0, 1.05)

    _ax2.boxplot(_dom_by_assay, labels=_assays)
    _ax2.set_xlabel("Assay")
    _ax2.set_ylabel("Dominant leader fraction")
    _ax2.set_title("Frontal: dominant sheep fraction")
    _ax2.set_ylim(0, 1.05)

    _ax3.boxplot(_rec_by_assay, labels=_assays)
    _ax3.set_xlabel("Assay")
    _ax3.set_ylabel("Normalised entropy")
    _ax3.set_title("Recruitment entropy\n(low = one sheep always initiates)")
    _ax3.set_ylim(0, 1.05)

    _fig2.suptitle("Leadership consistency across test trials")
    _fig2.tight_layout()

    mo.vstack([
        _fig2,
        mo.md("### Per-trial leadership table"),
        mo.ui.table(_agg_df),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "---\n"
        "## Binomial null for frontal leadership\n\n"
        "Under H₀ = \"all sheep lead equally,\" per-sheep leader-frame counts "
        "should follow Multinomial(T, 1/n). Three tests, ordered by how well "
        "they respect the autocorrelation introduced by the 15-s smoothing window:\n\n"
        "1. **Frame-level chi-square** — naive, treats each 1-s frame as "
        "independent. Anti-conservative; reported for reference only.\n"
        "2. **Run-level chi-square / per-sheep binomial** — collapses consecutive "
        "same-leader frames into one run, then tests run counts against "
        "Multinomial(R, 1/n). Treats each leadership switch as an independent draw.\n"
        "3. **Block-permutation empirical p-value** — shuffles run-level leader "
        "identities uniformly, weighting by the observed run-length distribution. "
        "Preserves how long each leadership bout typically lasts, so the null "
        "matches the autocorrelation structure of the data.\n"
    )
    return


@app.cell(hide_code=True)
def _(
    TRIALS, mo,
    TRACKS_CACHE, load_trial_tracks,
    np, pd, plt,
):
    from scipy.stats import binomtest, chisquare

    _TEST_CONFIGS = {'A', 'B', 'C', 'D'}
    _N_PERMS = 2000
    _ALPHA = 0.05
    _RNG = np.random.default_rng(0)

    _rows = []

    for _tidx, _trial in enumerate(TRIALS):
        if _trial['group_size'] < 2:
            continue
        if _trial['config'] not in _TEST_CONFIGS:
            continue

        _tracks = load_trial_tracks(
            _trial, tracks_cache=TRACKS_CACHE, apply_orient=True,
        )
        if len(_tracks) < 2:
            continue

        _sheep_ids = sorted(_tracks.keys())
        _n = len(_sheep_ids)
        _dur = _trial['duration_min']
        _t_common = np.arange(0, _dur + 1 / 60, 1 / 60)

        _gx = np.zeros((_n, len(_t_common)))
        _gy = np.zeros((_n, len(_t_common)))
        for _ci, _sid in enumerate(_sheep_ids):
            _trk = _tracks[_sid]
            _order = np.argsort(_trk['t'])
            _gx[_ci] = np.interp(_t_common, _trk['t'][_order], _trk['gx'][_order])
            _gy[_ci] = np.interp(_t_common, _trk['t'][_order], _trk['gy'][_order])

        _cx = _gx.mean(axis=0)
        _cy = _gy.mean(axis=0)
        _kernel = np.ones(15) / 15
        _vcx = np.convolve(np.gradient(_cx, _t_common), _kernel, mode='same')
        _vcy = np.convolve(np.gradient(_cy, _t_common), _kernel, mode='same')
        _vs = np.sqrt(_vcx**2 + _vcy**2)
        _vss = np.where(_vs > 1e-6, _vs, 1.0)
        _vnx = _vcx / _vss
        _vny = _vcy / _vss
        _dx = _gx - _cx
        _dy = _gy - _cy
        _proj = _dx * _vnx + _dy * _vny
        _moving = _vs > 0.02
        _leader = np.argmax(_proj, axis=0)
        _leader[~_moving] = -1

        _frames = _leader[_leader >= 0]
        _T = len(_frames)
        if _T == 0:
            continue

        _counts = np.bincount(_frames, minlength=_n).astype(float)
        _exp_frame = _T / _n
        _obs_max_frac = float(_counts.max() / _T)

        # Frame-level chi² — kept for reference; anti-conservative under autocorr.
        _chi_frame = float(((_counts - _exp_frame) ** 2 / _exp_frame).sum())
        _chi_frame_p = float(chisquare(_counts).pvalue)

        # Collapse the leader time series to runs of constant identity.
        _change = np.concatenate([[True], np.diff(_frames) != 0])
        _run_starts = np.where(_change)[0]
        _run_leaders = _frames[_run_starts]
        _run_ends = np.concatenate([_run_starts[1:], [_T]])
        _run_lengths = (_run_ends - _run_starts).astype(float)
        _R = len(_run_leaders)
        _run_counts = np.bincount(_run_leaders, minlength=_n).astype(float)

        # Run-level chi² (skip if expected count < 1 makes chi² unreliable).
        _exp_run = _R / _n
        _chi_run_p = float(chisquare(_run_counts).pvalue) if _exp_run >= 1.0 else None
        _binom_ps = [
            binomtest(int(_run_counts[_i]), _R, 1 / _n).pvalue for _i in range(_n)
        ]
        _min_binom_p_run = float(min(_binom_ps))

        # Block-permutation null on the frame-level chi² and the max-fraction.
        # Each run gets a uniformly drawn leader identity; weight by run length
        # to recover frame-equivalent counts.
        _null_chi = np.empty(_N_PERMS)
        _null_max = np.empty(_N_PERMS)
        for _pi in range(_N_PERMS):
            _pl = _RNG.integers(0, _n, size=_R)
            _pc = np.bincount(_pl, weights=_run_lengths, minlength=_n)
            _null_chi[_pi] = ((_pc - _exp_frame) ** 2 / _exp_frame).sum()
            _null_max[_pi] = _pc.max() / _T
        _emp_p_chi = float((_null_chi >= _chi_frame).mean())
        _emp_p_max = float((_null_max >= _counts.max() / _T).mean())
        _null_max_q975 = float(np.quantile(_null_max, 0.975))

        _rows.append({
            'Trial': _tidx,
            'Date': _trial['date'],
            'Config': _trial['config'],
            'Assay': str(_trial['assay']),
            'n': _n,
            'T (frames)': _T,
            'R (runs)': _R,
            'Max leader frac.': round(_obs_max_frac, 3),
            'Null max 97.5%': round(_null_max_q975, 3),
            'Chi² p (frame)': round(_chi_frame_p, 4),
            'Chi² p (run)': None if _chi_run_p is None else round(_chi_run_p, 4),
            'Binom min-p (run)': round(_min_binom_p_run, 4),
            'Block-perm p (chi²)': round(_emp_p_chi, 4),
            'Block-perm p (max)': round(_emp_p_max, 4),
            'Reject H₀': _emp_p_chi < _ALPHA,
        })

    if not _rows:
        mo.stop(True, mo.md("*No multi-sheep test trials found.*"))

    _bdf = pd.DataFrame(_rows)
    _assays = sorted(_bdf['Assay'].unique(), key=lambda x: (not x.isdigit(), x))

    _fig3, _axes3 = plt.subplots(1, 3, figsize=(15, 4.2))
    _axA, _axB, _axC = _axes3
    _floor = 1.0 / (_N_PERMS + 1)
    _jitter_rng = np.random.default_rng(1)

    # A: empirical p-value per trial, log-scaled
    for _ai, _a in enumerate(_assays):
        _sub = _bdf[_bdf['Assay'] == _a]
        _y = np.clip(_sub['Block-perm p (chi²)'].values, _floor, 1.0)
        _x = _ai + _jitter_rng.uniform(-0.15, 0.15, size=len(_y))
        _axA.scatter(_x, _y, s=30, alpha=0.75, color='#3B7DD8',
                     edgecolors='k', linewidths=0.3)
    _axA.axhline(_ALPHA, color='red', ls='--', lw=0.8, label=f'α = {_ALPHA}')
    _axA.set_yscale('log')
    _axA.set_xticks(range(len(_assays)))
    _axA.set_xticklabels(_assays)
    _axA.set_xlabel('Assay')
    _axA.set_ylabel('Block-perm p (chi²)')
    _axA.set_title('Empirical p-value vs. "all-equal" null')
    _axA.legend(fontsize=8)

    # B: observed max-leader fraction vs. null 97.5% bound
    _obs_by_a = [_bdf[_bdf['Assay'] == _a]['Max leader frac.'].values for _a in _assays]
    _null_by_a = [_bdf[_bdf['Assay'] == _a]['Null max 97.5%'].values for _a in _assays]
    _axB.boxplot(
        _obs_by_a, labels=_assays, widths=0.5, patch_artist=True,
        boxprops=dict(facecolor='#3B7DD840', edgecolor='#3B7DD8'),
        medianprops=dict(color='k'),
    )
    _null_med = [float(np.median(v)) if len(v) else np.nan for v in _null_by_a]
    _axB.plot(range(1, len(_assays) + 1), _null_med, 'r--', lw=1.2,
              marker='_', markersize=10,
              label='Null 97.5% (median per assay)')
    _axB.set_xlabel('Assay')
    _axB.set_ylabel('Max leader fraction')
    _axB.set_title('Observed dominance vs. null upper bound')
    _axB.set_ylim(0, 1.05)
    _axB.legend(fontsize=8)

    # C: rejection rate by assay
    _frac_reject = []
    _n_trials = []
    for _a in _assays:
        _sub = _bdf[_bdf['Assay'] == _a]
        _frac_reject.append(float(_sub['Reject H₀'].mean()) if len(_sub) else 0.0)
        _n_trials.append(int(len(_sub)))
    _axC.bar(range(len(_assays)), _frac_reject, color='#E8823A',
             edgecolor='k', linewidth=0.6)
    _axC.axhline(_ALPHA, color='red', ls='--', lw=0.8,
                 label=f'α = {_ALPHA} (chance)')
    for _i, (_fr, _nt) in enumerate(zip(_frac_reject, _n_trials)):
        _axC.text(_i, _fr + 0.02, f'{_nt}', ha='center', va='bottom',
                  fontsize=8, color='#444')
    _axC.set_xticks(range(len(_assays)))
    _axC.set_xticklabels(_assays)
    _axC.set_xlabel('Assay')
    _axC.set_ylabel('Fraction of trials rejecting H₀')
    _axC.set_title('"Consistent leader" rate (numbers = n trials)')
    _axC.set_ylim(0, 1.05)
    _axC.legend(fontsize=8)

    _fig3.suptitle('Binomial / multinomial null for frontal leadership')
    _fig3.tight_layout()

    _pooled_rows = []
    for _a in _assays:
        _sub = _bdf[_bdf['Assay'] == _a]
        if not len(_sub):
            continue
        _pooled_rows.append({
            'Assay': _a,
            'n trials': int(len(_sub)),
            'Median max frac.': round(float(_sub['Max leader frac.'].median()), 3),
            'Median null max 97.5%': round(float(_sub['Null max 97.5%'].median()), 3),
            'Frac. reject @ α=0.05': round(float(_sub['Reject H₀'].mean()), 3),
            'Median block-perm p': round(float(_sub['Block-perm p (chi²)'].median()), 4),
        })
    _pooled_df = pd.DataFrame(_pooled_rows)

    mo.vstack([
        _fig3,
        mo.md(
            f"**Across {len(_bdf)} test trials:** "
            f"{int(_bdf['Reject H₀'].sum())} reject H₀ (\"all sheep lead equally\") "
            f"at α = {_ALPHA} using the block-permutation chi². "
            f"Median observed max-leader fraction "
            f"{_bdf['Max leader frac.'].median():.3f} vs. null 97.5% "
            f"{_bdf['Null max 97.5%'].median():.3f}.\n\n"
            "*The block-permutation null shuffles run-level leader identity "
            "uniformly while preserving the observed run-length distribution, "
            "correcting for the frame-level autocorrelation induced by the 15-s "
            "velocity smoothing.*"
        ),
        mo.md("### Pooled by assay"),
        mo.ui.table(_pooled_df),
        mo.md("### Per-trial detail"),
        mo.ui.table(_bdf),
    ])
    return


if __name__ == "__main__":
    app.run()
