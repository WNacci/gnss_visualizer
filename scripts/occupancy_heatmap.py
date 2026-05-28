import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import re
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    COLORS = [
        [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
        [255, 127, 0], [255, 200, 51], [166, 86, 40], [247, 129, 191],
        [153, 153, 153], [0, 206, 209], [139, 69, 19], [0, 100, 0],
        [75, 0, 130], [220, 20, 60], [0, 139, 139], [184, 134, 11],
    ]

    def load_device_data_from_dir(device_dir):
        """Load GPS data from a GNSS device subdirectory."""
        _lats, _lons, _times = [], [], []
        for f in sorted(device_dir.iterdir(), key=lambda x: x.name):
            if re.match(r"LOGS\d+\.TXT", f.name, re.IGNORECASE) and f.stat().st_size > 0:
                for line in open(f, errors='ignore'):
                    parts = line.split(":")
                    if len(parts) >= 6:
                        try:
                            _lats.append(float(parts[3]))
                            _lons.append(float(parts[4]))
                            _times.append(float(parts[5]))
                        except ValueError:
                            pass
        return np.array(_lats), np.array(_lons), np.array(_times)

    def _load_device_data_from_files(files):
        _lats, _lons, _times = [], [], []
        for f in sorted(files, key=lambda x: x.name):
            if f.stat().st_size > 0:
                for line in open(f, errors='ignore'):
                    parts = line.split(":")
                    if len(parts) >= 6:
                        try:
                            _lats.append(float(parts[3]))
                            _lons.append(float(parts[4]))
                            _times.append(float(parts[5]))
                        except ValueError:
                            pass
        return np.array(_lats), np.array(_lons), np.array(_times)

    def detect_format_and_load(data_dir):
        """Detect directory format and load all device data."""
        if not data_dir.exists():
            return {}
        flat_files = list(data_dir.glob("GNSS_*_LOGS*.TXT"))
        if flat_files:
            _lats, _lons, _times = _load_device_data_from_files(flat_files)
            if len(_lats) > 0:
                return {data_dir.name: (_lats, _lons, _times)}
            return {}
        device_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("GNSS")])
        _data = {}
        for device_dir in device_dirs:
            _lats, _lons, _times = load_device_data_from_dir(device_dir)
            if len(_lats) > 0:
                _data[device_dir.name] = (_lats, _lons, _times)
        return _data

    def normalize_device_id(device_id):
        if isinstance(device_id, int):
            return [f"GNSS-{device_id}", f"GNSS_{device_id}"]
        _id = str(device_id)
        if '-' in _id or '_' in _id:
            _num = _id.replace('GNSS-', '').replace('GNSS_', '')
            return [f"GNSS-{_num}", f"GNSS_{_num}"]
        return [_id]

    def find_matching_devices(requested_devices, available_devices):
        if requested_devices is None:
            return sorted(available_devices)
        _matched = []
        for _req in requested_devices:
            for _fmt in normalize_device_id(_req):
                if _fmt in available_devices:
                    _matched.append(_fmt)
                    break
                for _avail in available_devices:
                    if _avail.lower() == _fmt.lower():
                        _matched.append(_avail)
                        break
        return _matched

    return (
        Path, np, pd, plt, gaussian_filter, COLORS,
        detect_format_and_load, find_matching_devices,
    )


@app.cell(hide_code=True)
def _():
    """Load trial metadata."""
    from gps_analysis import build_trials
    TRIALS = build_trials()
    print(f"Loaded {len(TRIALS)} trials")
    return (TRIALS,)


@app.cell(hide_code=True)
def _(pd):
    """Build arena coordinate transforms from fitted reward sites."""
    from gps_analysis import latlon_to_grid, apply_orientation, build_arena_transforms, DATA_DIR
    reward_sites_df = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
    ARENA_TRANSFORMS = build_arena_transforms(reward_sites_df)
    return (reward_sites_df, ARENA_TRANSFORMS, latlon_to_grid, apply_orientation)


@app.cell(hide_code=True)
def _(TRIALS, detect_format_and_load):
    """Pre-load all GNSS directories once at startup (does NOT depend on selected_indices)."""
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from gps_analysis import DATA_DIR
    from pathlib import Path
    _base = DATA_DIR / "gnss"
    _all_dirs = sorted(set(
        str(_base / f"{int(t['date'].split('-')[2])}-02-26")
        for t in TRIALS
    ))
    _existing = [d for d in _all_dirs if Path(d).exists()]

    def _load(path_str):
        return path_str, detect_format_and_load(Path(path_str))

    GPS_CACHE = {}
    _n = min(16, max(1, len(_existing)))
    with _TPE(max_workers=_n) as _pool:
        for _p, _raw in _pool.map(_load, _existing):
            GPS_CACHE[_p] = _raw

    _total = sum(len(v[0]) for _raw in GPS_CACHE.values() for v in _raw.values())
    print(f"GPS cache ready: {len(GPS_CACHE)} dirs, {_total:,} total points")
    return (GPS_CACHE,)


@app.cell(hide_code=True)
def _(mo):
    """Main UI controls: mode and visualization parameters."""
    mode_widget = mo.ui.radio(
        options=["Single trial", "Aggregated"],
        value="Single trial",
        label="Mode",
    )
    bins_slider = mo.ui.slider(start=20, stop=500, step=5, value=200, label="Bins")
    sigma_slider = mo.ui.slider(start=0, stop=0.5, step=0.01, value=0, label="Smoothing σ (grid units, 0=none)")
    duration_slider = mo.ui.slider(start=1, stop=35, step=1, value=35, label="Max duration (min)")
    cmap_dropdown = mo.ui.dropdown(
        options=["Blues", "hot_r", "viridis", "plasma", "YlOrRd"],
        value="Blues",
        label="Colormap",
    )
    log_scale_checkbox = mo.ui.checkbox(label="Log scale", value=False)
    show_sites_checkbox = mo.ui.checkbox(label="Show reward sites", value=True)

    mo.md(f"""
    # Occupancy Heatmap

    **Mode:** {mode_widget}

    ---
    ### Visualization Parameters
    {mo.hstack([bins_slider, sigma_slider, duration_slider])}
    {mo.hstack([cmap_dropdown, log_scale_checkbox, show_sites_checkbox])}
    """)
    return (
        mode_widget, bins_slider, sigma_slider, duration_slider,
        cmap_dropdown, log_scale_checkbox, show_sites_checkbox,
    )


@app.cell(hide_code=True)
def _(TRIALS, mo, mode_widget):
    """Single-trial mode controls: trial selector and per-sheep toggle."""
    _options = {}
    for _i, _trial in enumerate(TRIALS):
        _assay_str = f" [Assay {_trial['assay']}]" if _trial['assay'] is not None else ""
        _label = f"[{_i:2d}] {_trial['notes'].split(' - ')[0]:20s} {_trial['name']}{_assay_str}"
        _options[_label] = _i

    trial_selector = mo.ui.multiselect(options=_options, label="Select trials", full_width=True)
    per_sheep_checkbox = mo.ui.checkbox(label="Per-sheep subplots", value=False)

    (mo.vstack([
        mo.md("### Trial Selection"),
        trial_selector,
        per_sheep_checkbox,
    ]) if mode_widget.value == "Single trial" else mo.md(""))
    return (trial_selector, per_sheep_checkbox)


@app.cell(hide_code=True)
def _(TRIALS, mo, mode_widget):
    """Aggregated mode controls: filters and aggregate-by selector."""
    TEST_CONFIGS = sorted({"A", "B", "C", "D"})
    CTRL_CONFIGS = sorted(c for c in set(t["config"] for t in TRIALS) if c not in {"A", "B", "C", "D"})
    _all_configs = sorted(set(t["config"] for t in TRIALS))

    _fields = ["Both"] + sorted(set(t["field"] for t in TRIALS))
    _gsizes = [str(g) for g in sorted(set(t["group_size"] for t in TRIALS))]
    _assays = sorted(set(str(t["assay"]) for t in TRIALS if t["assay"] is not None))

    field_filter = mo.ui.dropdown(options=_fields, value="Both", label="Field")
    config_preset = mo.ui.radio(
        options=["All", "Test (A/B/C/D)", "Control", "Custom"],
        value="Test (A/B/C/D)",
        label="Configurations",
    )
    config_filter = mo.ui.multiselect(
        options=_all_configs, value=[], label="Custom config(s)",
    )
    groupsize_filter = mo.ui.multiselect(options=_gsizes, value=[], label="Group Size(s)")
    assay_filter = mo.ui.multiselect(options=_assays, value=[], label="Assay(s)")
    # Phase 2 began 2026-02-17 (trial #56 onwards): group-size-4-only sessions.
    phase_filter = mo.ui.dropdown(
        options=["Both phases", "Phase 1 only (pre Feb 17)", "Phase 2 only (Feb 17+)"],
        value="Phase 2 only (Feb 17+)",
        label="Study phase",
    )
    aggregateby_widget = mo.ui.radio(
        options=["All", "Field", "Configuration", "Group Size", "Assay"],
        value="Assay",
        label="Aggregate by",
    )
    transform_mode_widget = mo.ui.radio(
        options=["None", "Per configuration"],
        value="Per configuration",
        label="Orientation transforms",
    )

    (mo.vstack([
        mo.md("### Filters"),
        mo.hstack([field_filter, config_preset, groupsize_filter, assay_filter]),
        config_filter,
        mo.hstack([phase_filter]),
        mo.hstack([aggregateby_widget, transform_mode_widget]),
    ]) if mode_widget.value == "Aggregated" else mo.md(""))
    return (
        field_filter, config_preset, config_filter, groupsize_filter, assay_filter,
        phase_filter, aggregateby_widget, transform_mode_widget,
        TEST_CONFIGS, CTRL_CONFIGS,
    )




@app.cell(hide_code=True)
def _(
    TRIALS, mode_widget, trial_selector,
    field_filter, config_preset, config_filter, groupsize_filter, assay_filter,
    phase_filter, TEST_CONFIGS, CTRL_CONFIGS,
):
    """Compute the list of active trial indices based on current mode and filters."""
    _PHASE2_DATE = "2026-02-17"

    # Resolve the effective config set from preset or custom multiselect
    if config_preset.value == "Test (A/B/C/D)":
        _active_configs = set(TEST_CONFIGS)
    elif config_preset.value == "Control":
        _active_configs = set(CTRL_CONFIGS)
    elif config_preset.value == "Custom":
        _active_configs = set(config_filter.value) if config_filter.value else None
    else:  # "All"
        _active_configs = None

    if mode_widget.value == "Single trial":
        selected_indices = list(trial_selector.value)
    else:
        selected_indices = []
        for _i, _trial in enumerate(TRIALS):
            _date, _field, _config = _trial["date"], _trial["field"], _trial["config"]
            _assay, _gsize = _trial["assay"], _trial["group_size"]
            if field_filter.value != "Both" and _field != field_filter.value:
                continue
            if _active_configs is not None and _config not in _active_configs:
                continue
            if groupsize_filter.value and str(_gsize) not in groupsize_filter.value:
                continue
            if assay_filter.value and str(_assay) not in assay_filter.value:
                continue
            if phase_filter.value == "Phase 1 only (pre Feb 17)" and _date >= _PHASE2_DATE:
                continue
            if phase_filter.value == "Phase 2 only (Feb 17+)" and _date < _PHASE2_DATE:
                continue
            selected_indices.append(_i)
    print(f"Active trials: {len(selected_indices)}")
    return (selected_indices,)


@app.cell(hide_code=True)
def _(
    TRIALS, ARENA_TRANSFORMS,
    GPS_CACHE,
    selected_indices,
    transform_mode_widget,
    np, pd,
    find_matching_devices, latlon_to_grid, apply_orientation,
):
    """Project GPS data to arena grid coords and apply orientation transforms.

    Uses GPS_CACHE (pre-loaded at startup) so no disk I/O on filter changes.
    Auto-detects the correct field transform if the CSV label is wrong.
    Averages gx/gy for sheep carrying 2 GPS devices.
    """
    from gps_analysis import CONFIG_TRANSFORMS, DATA_DIR

    def _get_orientation(config):
        if transform_mode_widget.value == "Per configuration":
            return CONFIG_TRANSFORMS.get(config, (0, "none"))
        return (0, "none")

    def _in_arena_count(lats, lons, field_key):
        _gx_t, _gy_t = latlon_to_grid(lats, lons, ARENA_TRANSFORMS[field_key])
        return int(np.sum((_gx_t >= 0) & (_gx_t <= 5) & (_gy_t >= 0) & (_gy_t <= 5)))

    def _best_field(lats, lons, csv_field):
        """Return (field_key, was_corrected) using whichever transform keeps more points in-arena."""
        _best, _best_n = csv_field, _in_arena_count(lats, lons, csv_field)
        for _f_try in ARENA_TRANSFORMS:
            if _f_try == csv_field:
                continue
            _n = _in_arena_count(lats, lons, _f_try)
            if _n > _best_n:
                _best_n, _best = _n, _f_try
        return _best, (_best != csv_field)

    all_points = []
    _n_loaded = 0
    _n_autocorrected = 0
    _n_low_arena = 0
    _n_no_data = 0

    for _tidx in selected_indices:
        _t = TRIALS[_tidx]
        _name, _date, _csv_field = _t["name"], _t["date"], _t["field"]
        _config, _start_str, _dur_min = _t["config"], _t["start_time"], _t["duration_min"]
        _devs, _assay, _notes = _t["devices"], _t["assay"], _t["notes"]
        _gnum, _gsize = _t["group_num"], _t["group_size"]
        _day = int(_date.split("-")[2])
        _path_key = str(DATA_DIR / "gnss" / f"{_day}-02-26")
        _raw = GPS_CACHE.get(_path_key, {})
        if not _raw:
            _n_no_data += 1
            continue

        _dt_str = f"{_date} {_start_str}"
        _start_unix = pd.to_datetime(_dt_str).tz_localize('Europe/Paris').tz_convert('UTC').timestamp()
        _end_unix = _start_unix + _dur_min * 60

        _devices = find_matching_devices(_devs, list(_raw.keys()))
        _d2s = _t["device_to_sheep"]

        # Auto-detect correct field using the first device with time-filtered data
        _field = _csv_field
        _corrected = False
        for _dev in _devices:
            if _dev not in _raw:
                continue
            _lats_d, _lons_d, _times_d = _raw[_dev]
            _mask_d = (_times_d >= _start_unix) & (_times_d <= _end_unix)
            if _mask_d.sum() < 10:
                continue
            _field, _corrected = _best_field(_lats_d[_mask_d], _lons_d[_mask_d], _csv_field)
            break

        _rot, _ref = _get_orientation(_config)

        # Collect per-device projected arrays, grouped by sheep_id
        _sheep_tracks = {}  # sheep_id → list of (gx, gy, t_rel) arrays
        for _dev in _devices:
            if _dev not in _raw:
                continue
            _lats_d, _lons_d, _times_d = _raw[_dev]
            _mask_d = (_times_d >= _start_unix) & (_times_d <= _end_unix)
            _lf, _lnf = _lats_d[_mask_d], _lons_d[_mask_d]
            if len(_lf) == 0:
                continue
            _t_rel = (_times_d[_mask_d] - _start_unix) / 60.0
            _gx, _gy = latlon_to_grid(_lf, _lnf, ARENA_TRANSFORMS[_field])
            _gx, _gy = apply_orientation(_gx, _gy, _rot, _ref)

            _dev_parts = _dev.replace('GNSS-', '').replace('GNSS_', '')
            try:
                _dev_num = int(_dev_parts)
            except ValueError:
                _dev_num = None
            _sheep_id = _d2s.get(_dev_num, 'Unknown') if _dev_num is not None else 'Unknown'
            _sheep_tracks.setdefault(_sheep_id, []).append((_gx, _gy, _t_rel))

        if not _sheep_tracks:
            _n_no_data += 1
            continue

        # Average multi-device sheep; emit one all_points entry per sheep
        _trial_pts_total = 0
        _trial_pts_in_arena = 0
        _trial_pts_5min = 0
        _trial_pts_in_arena_5min = 0
        _trial_max_t = 0.0
        _sheep_summaries = []

        for _sheep_id, _tracks in _sheep_tracks.items():
            if len(_tracks) == 1:
                _gx_f, _gy_f, _t_f = _tracks[0]
            else:
                _t_min = min(t.min() for _, _, t in _tracks)
                _t_max = max(t.max() for _, _, t in _tracks)
                _t_grid = np.arange(_t_min, _t_max + 1/600, 1/600)  # 0.1 s steps (native 10 Hz)
                _stacks_gx, _stacks_gy = [], []
                for _gx_d, _gy_d, _t_d in _tracks:
                    _order = np.argsort(_t_d)
                    _stacks_gx.append(np.interp(_t_grid, _t_d[_order], _gx_d[_order]))
                    _stacks_gy.append(np.interp(_t_grid, _t_d[_order], _gy_d[_order]))
                _gx_f = np.mean(_stacks_gx, axis=0)
                _gy_f = np.mean(_stacks_gy, axis=0)
                _t_f = _t_grid

            _in_arena_mask = (_gx_f >= 0) & (_gx_f <= 5) & (_gy_f >= 0) & (_gy_f <= 5)
            _mask_5 = _t_f <= 5.0
            _n_total = len(_gx_f)
            _n_arena = int(_in_arena_mask.sum())
            _n_5min = int(_mask_5.sum())
            _n_arena_5min = int((_in_arena_mask & _mask_5).sum())
            _sheep_dur = float(_t_f.max()) if _n_total > 0 else 0.0

            _trial_pts_total += _n_total
            _trial_pts_in_arena += _n_arena
            _trial_pts_5min += _n_5min
            _trial_pts_in_arena_5min += _n_arena_5min
            _trial_max_t = max(_trial_max_t, _sheep_dur)

            _pct_s = 100 * _n_arena / _n_total if _n_total else 0
            _pct_5_s = 100 * _n_arena_5min / _n_5min if _n_5min else 0
            _sheep_summaries.append(
                f"      sheep {_sheep_id}: {_n_total:,} pts, {_sheep_dur:.1f} min, "
                f"{_pct_s:.0f}% in-arena, 0-5min: {_n_5min} pts {_pct_5_s:.0f}% in-arena"
            )

            all_points.append({
                'gx': _gx_f, 'gy': _gy_f, 't_rel_min': _t_f,
                'field': _field, 'config': _config,
                'assay': _assay, 'group_size': _gsize, 'group_num': _gnum,
                'sheep_id': _sheep_id, 'trial_idx': _tidx, 'trial_name': _name,
            })

        _pct = 100 * _trial_pts_in_arena / _trial_pts_total if _trial_pts_total else 0
        _pct_5 = 100 * _trial_pts_in_arena_5min / _trial_pts_5min if _trial_pts_5min else 0
        _flag = f"→{_field}*" if _corrected else f" {_field} "
        _n_sheep = len(_sheep_tracks)
        print(f"  Trial {_tidx:3d} [{_date} assay={_assay} {_config} field={_csv_field}{_flag}]: "
              f"{_n_sheep} sheep, {_trial_max_t:.1f} min, {_trial_pts_total:,} pts, "
              f"{_pct:.0f}% in-arena | 0-5min: {_trial_pts_5min} pts, {_pct_5:.0f}% in-arena")
        for _s in _sheep_summaries:
            print(_s)
        _n_loaded += 1
        if _corrected:
            _n_autocorrected += 1
        if _pct < 50:
            _n_low_arena += 1

    _total_pts = sum(len(p['gx']) for p in all_points)
    print(f"Projected {_n_loaded}/{len(selected_indices)} trials, {_total_pts:,} pts total")
    if _n_autocorrected:
        print(f"  {_n_autocorrected} trial(s) had field auto-corrected (CSV label didn't match GPS location)")
    if _n_low_arena:
        print(f"  {_n_low_arena} trial(s) had <50% pts in-arena")
    if _n_no_data:
        print(f"  {_n_no_data} trial(s) had no GPS data")
    return (all_points,)


@app.cell(hide_code=True)
def _(
    all_points, selected_indices,
    mode_widget, per_sheep_checkbox, aggregateby_widget,
    bins_slider, sigma_slider, duration_slider,
    np, gaussian_filter,
):
    """Bin projected GPS points into 2D histograms.

    Separated from GPS loading so that changing bins/sigma/colormap/duration does not
    re-read files.
    """

    def _compute_heatmap(gx, gy, bins, sigma):
        _H, _xe, _ye = np.histogram2d(gx, gy, bins=bins, range=[[0, 5], [0, 5]])
        _H = _H.T  # shape (bins_y, bins_x) for imshow with origin='lower'
        if sigma > 0:
            _sigma_bins = sigma / (5.0 / bins)  # convert grid units → bins
            _H = gaussian_filter(_H.astype(float), sigma=_sigma_bins)
        return _H.astype(float), _xe, _ye

    def _concat_dur(pts, dur_limit):
        """Concatenate gx/gy arrays from a list of points, filtered to dur_limit minutes."""
        _gx_parts, _gy_parts = [], []
        for _p in pts:
            _m = _p['t_rel_min'] <= dur_limit
            _gx_parts.append(_p['gx'][_m])
            _gy_parts.append(_p['gy'][_m])
        return np.concatenate(_gx_parts), np.concatenate(_gy_parts)

    _bins = bins_slider.value
    _sigma = sigma_slider.value
    _dur_limit = duration_slider.value

    # Build heatmap subplots: list of (title, H, xedges, yedges, field, is_aggregated, config, assay)
    heatmap_subplots = []

    if mode_widget.value == "Single trial":
        if all_points:
            _fields_present = set(p['field'] for p in all_points)
            _subplot_field = list(_fields_present)[0] if len(_fields_present) == 1 else "A+B"
            _configs_present = set(p['config'] for p in all_points)
            _subplot_config = list(_configs_present)[0] if len(_configs_present) == 1 else 'mixed'
            _assays_present = set(p['assay'] for p in all_points)
            _subplot_assay = list(_assays_present)[0] if len(_assays_present) == 1 else 'mixed'
            _title_base = all_points[0]['trial_name'] if len(selected_indices) == 1 else f"{len(selected_indices)} trials"

            if per_sheep_checkbox.value:
                _unique_sheep = sorted(set(p['sheep_id'] for p in all_points))
                if len(_unique_sheep) > 1:
                    for _sheep in _unique_sheep:
                        _sp = [p for p in all_points if p['sheep_id'] == _sheep]
                        _gx, _gy = _concat_dur(_sp, _dur_limit)
                        _H, _xe, _ye = _compute_heatmap(_gx, _gy, _bins, _sigma)
                        heatmap_subplots.append((
                            f"{_title_base}\nSheep {_sheep} ({len(_gx):,} pts)",
                            _H, _xe, _ye, _subplot_field, False, _subplot_config, _subplot_assay,
                        ))
                    _gx_all, _gy_all = _concat_dur(all_points, _dur_limit)
                    _H_c, _xe_c, _ye_c = _compute_heatmap(_gx_all, _gy_all, _bins, _sigma)
                    heatmap_subplots.append((
                        f"{_title_base}\nAll sheep ({len(_gx_all):,} pts)",
                        _H_c, _xe_c, _ye_c, _subplot_field, False, _subplot_config, _subplot_assay,
                    ))

            if not heatmap_subplots:
                _gx, _gy = _concat_dur(all_points, _dur_limit)
                _H, _xe, _ye = _compute_heatmap(_gx, _gy, _bins, _sigma)
                heatmap_subplots.append((
                    f"{_title_base} ({len(_gx):,} pts)",
                    _H, _xe, _ye, _subplot_field, False, _subplot_config, _subplot_assay,
                ))

    else:  # Aggregated
        _agg_by = aggregateby_widget.value

        def _group_key(p):
            if _agg_by == "Field": return p['field']
            if _agg_by == "Configuration": return p['config']
            if _agg_by == "Group Size": return str(p['group_size'])
            if _agg_by == "Assay": return str(p['assay'])
            return "All"

        _groups = {}
        for _p in all_points:
            _groups.setdefault(_group_key(_p), []).append(_p)

        for _key in sorted(_groups.keys()):
            _pts = _groups[_key]
            _gx, _gy = _concat_dur(_pts, _dur_limit)
            _H, _xe, _ye = _compute_heatmap(_gx, _gy, _bins, _sigma)
            _fields_in_group = set(p['field'] for p in _pts)
            _f = list(_fields_in_group)[0] if len(_fields_in_group) == 1 else "A+B"
            _n_trials = len(set(p['trial_idx'] for p in _pts))
            _agg_config = _key if _agg_by == "Configuration" else 'mixed'
            _agg_assay = _key if _agg_by == "Assay" else 'mixed'
            heatmap_subplots.append((
                f"{_agg_by}: {_key}\n({_n_trials} trials, {len(_gx):,} pts)",
                _H, _xe, _ye, _f, True, _agg_config, _agg_assay,
            ))

    print(f"{len(heatmap_subplots)} subplot(s), {sum(len(p['gx']) for p in all_points):,} total points binned")
    return (heatmap_subplots,)


@app.cell(hide_code=True)
def _(
    heatmap_subplots, reward_sites_df,
    cmap_dropdown, log_scale_checkbox, show_sites_checkbox,
    np, plt,
):
    """Render occupancy heatmap figure."""

    # Config A reward sites sorted bottom-left → top-right (by gx + gy) for aggregated view.
    # These are the 3 sites rewarded in config A and serve as the reference after transforms.
    _ref_sites = reward_sites_df[
        (reward_sites_df['field'] == 'A') &
        (reward_sites_df['label'].str.match(r'^A\d+$'))
    ].copy()
    _ref_sites['_sort'] = _ref_sites['grid_x'] + _ref_sites['grid_y']
    _ref_sites = _ref_sites.sort_values('_sort').reset_index(drop=True)
    _ref_sites['site_num'] = range(1, len(_ref_sites) + 1)

    if not heatmap_subplots:
        _fig, _ax = plt.subplots(1, 1, figsize=(6, 6))
        _ax.text(
            0.5, 0.5,
            "No data selected.\nChoose trials or adjust filters.",
            ha='center', va='center', transform=_ax.transAxes, fontsize=12,
        )
        _ax.set_axis_off()
    else:
        _n = len(heatmap_subplots)
        _ncols = min(_n, 3)
        _nrows = (_n + _ncols - 1) // _ncols
        _fig, _axes = plt.subplots(
            _nrows, _ncols, figsize=(7 * _ncols, 7 * _nrows),
            squeeze=False, constrained_layout=True, dpi=150,
        )
        _fig.patch.set_alpha(0)

        # Shared colour scale across all subplots
        _all_display = [np.log1p(H) if log_scale_checkbox.value else H
                        for _, H, *_ in heatmap_subplots]
        _vmax = max((d.max() for d in _all_display), default=1)
        _vmax = _vmax if _vmax > 0 else 1
        _cbar_label = 'log(1+count)' if log_scale_checkbox.value else 'count'
        _last_im = None

        for _idx, (_title, _H, _xe, _ye, _field, _is_aggregated, _config, _assay) in enumerate(heatmap_subplots):
            _ax = _axes[_idx // _ncols][_idx % _ncols]

            _display_H = np.log1p(_H) if log_scale_checkbox.value else _H

            if _display_H.max() > 0:
                _im = _ax.imshow(
                    _display_H,
                    extent=[_xe[0], _xe[-1], _ye[0], _ye[-1]],
                    origin='lower',
                    cmap=cmap_dropdown.value,
                    aspect='equal',
                    interpolation='nearest',
                    vmin=0, vmax=_vmax,
                )
                _last_im = _im
            else:
                _ax.text(0.5, 0.5, "No points in arena bounds",
                        ha='center', va='center', transform=_ax.transAxes)

            # Reward sites overlay (hidden for assay 0 and control configurations)
            _BAITED_COLOR = '#FFE066'   # soft yellow
            _UNBAITED_COLOR = '#888888' # grey
            _is_assay0 = (str(_assay) == '0')
            _is_ctrl = _config not in ('A', 'B', 'C', 'D', 'mixed')
            if show_sites_checkbox.value and not _is_assay0 and not _is_ctrl:
                if _is_aggregated:
                    # Aggregated mode: orientation transforms normalise to config A,
                    # so all shown sites (A1/A2/A3) are baited → soft yellow.
                    for _, _r in _ref_sites.iterrows():
                        _ax.scatter(
                            _r['grid_x'], _r['grid_y'],
                            s=120, zorder=5, marker='o',
                            facecolors='none', edgecolors=_BAITED_COLOR, linewidths=2,
                        )
                        _ax.annotate(
                            str(int(_r['site_num'])), (_r['grid_x'], _r['grid_y']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, color=_BAITED_COLOR, fontweight='bold',
                        )
                elif _field in ('A', 'B'):
                    # Single-trial mode: baited sites are those whose label prefix
                    # matches the trial config (e.g. config A → A1/A2/A3).
                    _fd = reward_sites_df[
                        (reward_sites_df['field'] == _field) &
                        (~reward_sites_df['label'].str.startswith('E'))
                    ]
                    for _, _r in _fd.iterrows():
                        _is_baited = _r['label'].startswith(_config) and _config in ('A', 'B', 'C', 'D')
                        _ec = _BAITED_COLOR if _is_baited else _UNBAITED_COLOR
                        _ax.scatter(
                            _r['grid_x'], _r['grid_y'],
                            s=100, zorder=5, marker='o',
                            facecolors='none', edgecolors=_ec, linewidths=2,
                        )
                        _ax.annotate(
                            _r['label'], (_r['grid_x'], _r['grid_y']),
                            xytext=(4, 4), textcoords='offset points',
                            fontsize=7, color=_ec, fontweight='bold',
                        )

            # Faint grid lines at integer positions (reward site grid)
            for _v in range(1, 5):
                _ax.axvline(_v, color='white', alpha=0.15, linewidth=0.5)
                _ax.axhline(_v, color='white', alpha=0.15, linewidth=0.5)

            _ax.set_xlim(-0.05, 5.05)
            _ax.set_ylim(-0.05, 5.05)
            _ax.set_xlabel("Grid X (10 m/unit)")
            _ax.set_ylabel("Grid Y (10 m/unit)")
            _ax.set_title(_title, fontsize=9)
            _ax.set_facecolor('none')

        # Hide unused subplot axes
        for _idx in range(_n, _nrows * _ncols):
            _axes[_idx // _ncols][_idx % _ncols].set_visible(False)

        # Single shared colorbar for all subplots
        if _last_im is not None:
            _fig.colorbar(
                _last_im,
                ax=_axes.ravel().tolist(),
                label=_cbar_label,
                shrink=0.6,
                pad=0.02,
            )


    _fig


if __name__ == "__main__":
    app.run()
