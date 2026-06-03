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
    from gps_analysis import latlon_to_grid, apply_orientation, build_arena_transforms, DATA_DIR as _DATA_DIR
    reward_sites_df = pd.read_csv(_DATA_DIR / "fitted_reward_sites.csv")
    ARENA_TRANSFORMS = build_arena_transforms(reward_sites_df)
    return (reward_sites_df, ARENA_TRANSFORMS, latlon_to_grid, apply_orientation)


@app.cell(hide_code=True)
def _(TRIALS, detect_format_and_load, Path):
    """Pre-load all GNSS directories once at startup (does NOT depend on selected_indices)."""
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from gps_analysis import DATA_DIR as _DATA_DIR
    _base = _DATA_DIR / "gnss"
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
    from gps_analysis import CONFIG_TRANSFORMS as _CONFIG_TRANSFORMS, DATA_DIR as _DATA_DIR

    def _get_orientation(config):
        if transform_mode_widget.value == "Per configuration":
            return _CONFIG_TRANSFORMS.get(config, (0, "none"))
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
        _path_key = str(_DATA_DIR / "gnss" / f"{_day}-02-26")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Rigour pass: Test vs Control + sheep-ID shuffle null

        Two additions independent of the UI filters above:

        1. **Test vs Control occupancy panel** — pools all Phase 2 trials by whether
           the configuration is a baited test (`A/B/C/D`) or a control
           (`CTRL_FAR/CTRL_BARN`) and renders test, control, and (test − control)
           occupancy on a shared grid.
        2. **Sheep-ID-shuffle null** — tests whether sheep within a trial occupy
           statistically distinct sub-regions, using mean per-sheep Shannon entropy
           as the metric (see explanatory note in the null cell below).
        """
    )
    return


@app.cell(hide_code=True)
def _(
    TRIALS, ARENA_TRANSFORMS, GPS_CACHE,
    np, pd,
    find_matching_devices, latlon_to_grid, apply_orientation,
):
    """Build Test vs Control point pools across Phase 2, oriented per-config.

    Independent of the UI filters above: enumerates ALL Phase 2 trials
    (date >= 2026-02-17) with config in the test or control sets, projects
    GPS to arena grid coords, applies per-configuration orientation,
    averages multi-device sheep, and aggregates points per pool.
    """
    from gps_analysis import CONFIG_TRANSFORMS as _CONFIG_TRANSFORMS
    from gps_analysis import DATA_DIR as _DATA_DIR

    _PHASE2_DATE = "2026-02-17"
    _TEST_SET = {"A", "B", "C", "D"}
    _CTRL_SET = {"CTRL_FAR", "CTRL_BARN"}

    def _in_arena_count_tc(lats, lons, field_key):
        _gxc, _gyc = latlon_to_grid(lats, lons, ARENA_TRANSFORMS[field_key])
        return int(np.sum((_gxc >= 0) & (_gxc <= 5) & (_gyc >= 0) & (_gyc <= 5)))

    def _best_field_tc(lats, lons, csv_field):
        _best, _best_n = csv_field, _in_arena_count_tc(lats, lons, csv_field)
        for _f_try in ARENA_TRANSFORMS:
            if _f_try == csv_field:
                continue
            _n = _in_arena_count_tc(lats, lons, _f_try)
            if _n > _best_n:
                _best_n, _best = _n, _f_try
        return _best

    def _project_trial(t):
        """Return list of all_points-style dicts (one per sheep) for trial t, or []."""
        _date = t["date"]
        _day = int(_date.split("-")[2])
        _raw = GPS_CACHE.get(str(_DATA_DIR / "gnss" / f"{_day}-02-26"), {})
        if not _raw:
            return []
        _start_unix = (
            pd.to_datetime(f"{_date} {t['start_time']}")
            .tz_localize('Europe/Paris').tz_convert('UTC').timestamp()
        )
        _end_unix = _start_unix + t["duration_min"] * 60
        _devices = find_matching_devices(t["devices"], list(_raw.keys()))
        _d2s = t["device_to_sheep"]

        _field = t["field"]
        for _dev in _devices:
            if _dev not in _raw:
                continue
            _lats_d, _lons_d, _times_d = _raw[_dev]
            _mask_d = (_times_d >= _start_unix) & (_times_d <= _end_unix)
            if _mask_d.sum() < 10:
                continue
            _field = _best_field_tc(_lats_d[_mask_d], _lons_d[_mask_d], t["field"])
            break

        _rot, _ref = _CONFIG_TRANSFORMS.get(t["config"], (0, "none"))

        _sheep_tracks = {}
        for _dev in _devices:
            if _dev not in _raw:
                continue
            _lats_d, _lons_d, _times_d = _raw[_dev]
            _mask_d = (_times_d >= _start_unix) & (_times_d <= _end_unix)
            if _mask_d.sum() == 0:
                continue
            _t_rel = (_times_d[_mask_d] - _start_unix) / 60.0
            _gx, _gy = latlon_to_grid(
                _lats_d[_mask_d], _lons_d[_mask_d], ARENA_TRANSFORMS[_field],
            )
            _gx, _gy = apply_orientation(_gx, _gy, _rot, _ref)

            _dev_parts = _dev.replace('GNSS-', '').replace('GNSS_', '')
            try:
                _dev_num = int(_dev_parts)
            except ValueError:
                _dev_num = None
            _sheep_id = _d2s.get(_dev_num, 'Unknown') if _dev_num is not None else 'Unknown'
            _sheep_tracks.setdefault(_sheep_id, []).append((_gx, _gy, _t_rel))

        _out = []
        for _sheep_id, _tracks in _sheep_tracks.items():
            if len(_tracks) == 1:
                _gx_f, _gy_f, _t_f = _tracks[0]
            else:
                _t_min = min(tt.min() for _, _, tt in _tracks)
                _t_max = max(tt.max() for _, _, tt in _tracks)
                _t_grid = np.arange(_t_min, _t_max + 1/600, 1/600)
                _stacks_gx, _stacks_gy = [], []
                for _gx_d, _gy_d, _t_d in _tracks:
                    _order = np.argsort(_t_d)
                    _stacks_gx.append(np.interp(_t_grid, _t_d[_order], _gx_d[_order]))
                    _stacks_gy.append(np.interp(_t_grid, _t_d[_order], _gy_d[_order]))
                _gx_f = np.mean(_stacks_gx, axis=0)
                _gy_f = np.mean(_stacks_gy, axis=0)
                _t_f = _t_grid
            _out.append({
                'gx': _gx_f, 'gy': _gy_f, 't_rel_min': _t_f,
                'field': _field, 'config': t["config"],
                'assay': t["assay"], 'group_size': t["group_size"],
                'group_num': t["group_num"],
                'sheep_id': _sheep_id, 'trial_idx': None, 'trial_name': t["name"],
            })
        return _out

    test_points, ctrl_points = [], []
    _n_test_trials = _n_ctrl_trials = 0
    for _i, _t in enumerate(TRIALS):
        if _t["date"] < _PHASE2_DATE:
            continue
        _cfg = _t["config"]
        if _cfg in _TEST_SET:
            _pts = _project_trial(_t)
            if _pts:
                for _p in _pts:
                    _p['trial_idx'] = _i
                test_points.extend(_pts)
                _n_test_trials += 1
        elif _cfg in _CTRL_SET:
            _pts = _project_trial(_t)
            if _pts:
                for _p in _pts:
                    _p['trial_idx'] = _i
                ctrl_points.extend(_pts)
                _n_ctrl_trials += 1

    _n_test_pts = sum(len(p['gx']) for p in test_points)
    _n_ctrl_pts = sum(len(p['gx']) for p in ctrl_points)
    print(
        f"Test pool: {_n_test_trials} trials, {_n_test_pts:,} pts | "
        f"Control pool: {_n_ctrl_trials} trials, {_n_ctrl_pts:,} pts"
    )
    return (test_points, ctrl_points)


@app.cell(hide_code=True)
def _(
    test_points, ctrl_points, reward_sites_df,
    bins_slider, cmap_dropdown, log_scale_checkbox, duration_slider,
    np, plt,
):
    """1x3 panel: test occupancy, control occupancy, and (test - control)."""
    _bins = bins_slider.value
    _dur_limit = duration_slider.value

    def _concat_pool(pool):
        if not pool:
            return np.array([]), np.array([])
        _gxs, _gys = [], []
        for _p in pool:
            _m = _p['t_rel_min'] <= _dur_limit
            _gxs.append(_p['gx'][_m])
            _gys.append(_p['gy'][_m])
        return np.concatenate(_gxs), np.concatenate(_gys)

    def _hist(gx, gy):
        if len(gx) == 0:
            return np.zeros((_bins, _bins))
        _H, _, _ = np.histogram2d(gx, gy, bins=_bins, range=[[0, 5], [0, 5]])
        return _H.T

    _gx_t, _gy_t = _concat_pool(test_points)
    _gx_c, _gy_c = _concat_pool(ctrl_points)
    _H_t = _hist(_gx_t, _gy_t)
    _H_c = _hist(_gx_c, _gy_c)

    # Normalize each to probability so the difference is scale-comparable
    _Pt = _H_t / _H_t.sum() if _H_t.sum() > 0 else _H_t
    _Pc = _H_c / _H_c.sum() if _H_c.sum() > 0 else _H_c
    _D = _Pt - _Pc

    _ref_sites_tc = reward_sites_df[
        (reward_sites_df['field'] == 'A') &
        (reward_sites_df['label'].str.match(r'^A\d+$'))
    ].copy()
    _ref_sites_tc['_sort'] = _ref_sites_tc['grid_x'] + _ref_sites_tc['grid_y']
    _ref_sites_tc = _ref_sites_tc.sort_values('_sort').reset_index(drop=True)

    _fig_tc, _axes_tc = plt.subplots(
        1, 3, figsize=(20, 6.5), constrained_layout=True, dpi=140,
    )
    _fig_tc.patch.set_alpha(0)

    _disp_t = np.log1p(_H_t) if log_scale_checkbox.value else _H_t
    _disp_c = np.log1p(_H_c) if log_scale_checkbox.value else _H_c
    _vmax_tc = float(max(_disp_t.max(), _disp_c.max(), 1))
    _cbar_label_tc = 'log(1+count)' if log_scale_checkbox.value else 'count'

    _im_t = _axes_tc[0].imshow(
        _disp_t, extent=[0, 5, 0, 5], origin='lower',
        cmap=cmap_dropdown.value, aspect='equal',
        interpolation='nearest', vmin=0, vmax=_vmax_tc,
    )
    _axes_tc[0].set_title(
        f"Test (A/B/C/D, canonical frame)\n"
        f"{len(test_points)} sheep-trials, {int(_H_t.sum()):,} pts",
        fontsize=10,
    )
    _fig_tc.colorbar(_im_t, ax=_axes_tc[0], shrink=0.7, label=_cbar_label_tc)

    # Reward markers in canonical Config-A frame on the test panel only
    for _, _r in _ref_sites_tc.iterrows():
        _axes_tc[0].scatter(
            _r['grid_x'], _r['grid_y'],
            s=120, zorder=5, marker='o',
            facecolors='none', edgecolors='#FFE066', linewidths=2,
        )

    _im_c = _axes_tc[1].imshow(
        _disp_c, extent=[0, 5, 0, 5], origin='lower',
        cmap=cmap_dropdown.value, aspect='equal',
        interpolation='nearest', vmin=0, vmax=_vmax_tc,
    )
    _axes_tc[1].set_title(
        f"Control (CTRL_FAR/CTRL_BARN)\n"
        f"{len(ctrl_points)} sheep-trials, {int(_H_c.sum()):,} pts",
        fontsize=10,
    )
    _fig_tc.colorbar(_im_c, ax=_axes_tc[1], shrink=0.7, label=_cbar_label_tc)

    _absmax = float(max(abs(_D.min()), abs(_D.max()), 1e-12))
    _im_d = _axes_tc[2].imshow(
        _D, extent=[0, 5, 0, 5], origin='lower',
        cmap='RdBu_r', aspect='equal',
        interpolation='nearest', vmin=-_absmax, vmax=_absmax,
    )
    _axes_tc[2].set_title(
        "Difference: P(test) − P(control)\n(linear, divergent)",
        fontsize=10,
    )
    _fig_tc.colorbar(_im_d, ax=_axes_tc[2], shrink=0.7, label='Δ probability')

    for _ax_tc in _axes_tc:
        for _v in range(1, 5):
            _ax_tc.axvline(_v, color='white', alpha=0.15, linewidth=0.5)
            _ax_tc.axhline(_v, color='white', alpha=0.15, linewidth=0.5)
        _ax_tc.set_xlim(-0.05, 5.05)
        _ax_tc.set_ylim(-0.05, 5.05)
        _ax_tc.set_xlabel("Grid X (10 m/unit)")
        _ax_tc.set_ylabel("Grid Y (10 m/unit)")
        _ax_tc.set_facecolor('none')

    _fig_tc


@app.cell(hide_code=True)
def _(
    test_points, ctrl_points,
    bins_slider, duration_slider,
    np, pd, mo,
):
    """Sheep-ID-shuffle null on mean per-sheep occupancy entropy.

    Note on methodology: shuffling sheep_id labels per point leaves the
    aggregate trial histogram invariant, so the literal label-shuffle null on
    aggregate entropy is degenerate. We instead use the **mean per-sheep
    entropy** as the metric. This tests the substantive question:
    "do sheep within a trial occupy distinct sub-regions, or are they
    statistically interchangeable?"

    For each Phase 2 trial (test or control):
      1. Bin each sheep's points into a 2D histogram on bins_slider grid
         and normalize to probability; H_s = -sum p log p over non-zero bins.
      2. Observed metric = mean of H_s across sheep.
      3. Null: permute sheep_id labels across the trial's points (preserving
         per-sheep counts), recompute each sheep's entropy, take the mean.
      4. N=1000, seed=42; report observed, null mean, 95% interval, p_emp.
    """
    _N_PERM = 1000
    _rng = np.random.default_rng(seed=42)
    _bins_e = bins_slider.value
    _dur_limit_e = duration_slider.value

    def _entropy(H):
        _s = H.sum()
        if _s <= 0:
            return 0.0
        _p = H.ravel() / _s
        _nz = _p[_p > 0]
        return float(-(_nz * np.log(_nz)).sum())

    def _trial_groups(pool):
        _by_trial = {}
        for _p in pool:
            _by_trial.setdefault(_p['trial_idx'], []).append(_p)
        return _by_trial

    def _mean_per_sheep_entropy(gx_all, gy_all, labels):
        """Compute mean per-sheep entropy given concatenated coords + sheep labels."""
        _vals = []
        for _lab in np.unique(labels):
            _m = labels == _lab
            if _m.sum() == 0:
                continue
            _H, _, _ = np.histogram2d(
                gx_all[_m], gy_all[_m],
                bins=_bins_e, range=[[0, 5], [0, 5]],
            )
            _vals.append(_entropy(_H))
        return float(np.mean(_vals)) if _vals else 0.0

    _rows = []
    for _pool in (test_points, ctrl_points):
        for _tidx, _pts in _trial_groups(_pool).items():
            _gx_parts, _gy_parts, _lab_parts = [], [], []
            for _p in _pts:
                _m = _p['t_rel_min'] <= _dur_limit_e
                _gx_parts.append(_p['gx'][_m])
                _gy_parts.append(_p['gy'][_m])
                _lab_parts.append(np.full(int(_m.sum()), str(_p['sheep_id'])))
            if not _gx_parts:
                continue
            _gx_all = np.concatenate(_gx_parts)
            _gy_all = np.concatenate(_gy_parts)
            _labels = np.concatenate(_lab_parts)
            if len(_gx_all) == 0 or len(np.unique(_labels)) < 2:
                continue

            _obs = _mean_per_sheep_entropy(_gx_all, _gy_all, _labels)

            _null = np.empty(_N_PERM, dtype=float)
            _shuf = _labels.copy()
            for _k in range(_N_PERM):
                _rng.shuffle(_shuf)
                _null[_k] = _mean_per_sheep_entropy(_gx_all, _gy_all, _shuf)

            _p_emp = max(
                1.0 / (_N_PERM + 1),
                2.0 * min((_null >= _obs).mean(), (_null <= _obs).mean()),
            )
            _first = _pts[0]
            _rows.append({
                'trial_idx': _tidx,
                'name': _first['trial_name'],
                'config': _first['config'],
                'n_sheep': int(len(np.unique(_labels))),
                'observed': _obs,
                'null_mean': float(_null.mean()),
                'null_low': float(np.percentile(_null, 2.5)),
                'null_high': float(np.percentile(_null, 97.5)),
                'p_emp': _p_emp,
            })

    entropy_results = pd.DataFrame(_rows)
    print(
        f"Entropy null: {len(entropy_results)} trials evaluated "
        f"(N_PERMUTATIONS={_N_PERM})"
    )
    if len(entropy_results):
        _n_sig = int((entropy_results['p_emp'] < 0.05).sum())
        print(
            f"  observed vs null (mean): "
            f"{entropy_results['observed'].mean():.3f} vs "
            f"{entropy_results['null_mean'].mean():.3f}"
        )
        print(f"  {_n_sig}/{len(entropy_results)} trials with p_emp < 0.05")

    mo.md(
        r"""
        ### Sheep-ID-shuffle null on mean per-sheep occupancy entropy

        Aggregate occupancy entropy is invariant under sheep-ID permutation,
        so we test the substantive question: **"do sheep within a trial
        occupy distinct sub-regions, or are they statistically
        interchangeable?"** via mean per-sheep entropy. Lower observed
        entropy than the null indicates per-sheep occupancy is more
        spatially concentrated than chance would predict if sheep were
        interchangeable (i.e., sheep have individual sub-regions).
        """
    )
    return (entropy_results,)


@app.cell(hide_code=True)
def _(entropy_results, mo, np, plt):
    """Bar plot of observed mean entropy per config with 95% null whiskers + dots."""
    if len(entropy_results) == 0:
        _fig_e, _ax_e = plt.subplots(figsize=(8, 5))
        _ax_e.text(
            0.5, 0.5, "No entropy results — empty Phase 2 pools.",
            ha='center', va='center', transform=_ax_e.transAxes,
        )
        _ax_e.set_axis_off()
        _table = mo.md("_no rows_")
    else:
        _configs_order = ["A", "B", "C", "D", "CTRL_FAR", "CTRL_BARN"]
        _present = [c for c in _configs_order if (entropy_results['config'] == c).any()]

        _fig_e, _ax_e = plt.subplots(figsize=(9, 5.5), dpi=140)
        _fig_e.patch.set_alpha(0)
        _x = np.arange(len(_present))

        _obs_means = []
        _null_means = []
        _null_los = []
        _null_his = []
        for _c in _present:
            _df = entropy_results[entropy_results['config'] == _c]
            _obs_means.append(_df['observed'].mean())
            _null_means.append(_df['null_mean'].mean())
            _null_los.append(_df['null_low'].mean())
            _null_his.append(_df['null_high'].mean())

        _obs_arr = np.array(_obs_means)
        _nm_arr = np.array(_null_means)
        _yerr = np.vstack([_nm_arr - np.array(_null_los), np.array(_null_his) - _nm_arr])

        _ax_e.bar(
            _x - 0.18, _obs_arr, width=0.36, color='#377eb8',
            label='Observed (mean across trials)',
        )
        _ax_e.bar(
            _x + 0.18, _nm_arr, width=0.36, color='#bbbbbb',
            yerr=_yerr, capsize=4,
            label='Null mean ± 95% (label-shuffle)',
        )

        # Per-trial dots overlaid on the observed bar
        _jitter_rng = np.random.default_rng(0)
        for _i_c, _c in enumerate(_present):
            _df = entropy_results[entropy_results['config'] == _c]
            _xs = (_x[_i_c] - 0.18) + _jitter_rng.uniform(-0.06, 0.06, len(_df))
            _ax_e.scatter(_xs, _df['observed'], s=24, color='#222222', zorder=5, alpha=0.85)

        _ax_e.set_xticks(_x)
        _ax_e.set_xticklabels(_present)
        _ax_e.set_ylabel("Mean per-sheep entropy (nats)")
        _ax_e.set_title("Mean per-sheep occupancy entropy: observed vs sheep-ID-shuffle null")
        _ax_e.legend(loc='best', fontsize=8)
        _ax_e.grid(axis='y', alpha=0.25)

        _table = mo.ui.table(entropy_results, page_size=25)

    mo.vstack([mo.as_html(_fig_e), _table])


@app.cell(hide_code=True)
def _(
    test_points, ctrl_points, reward_sites_df,
    bins_slider, cmap_dropdown, log_scale_checkbox, duration_slider,
    np, plt, mo,
):
    """Real vs movement-matched random-walk occupancy heatmaps.

    For every sheep-trial in the test/control pools we fit per-sheep empirical
    step-length and turn-angle distributions, then simulate _K_SIM correlated
    random walks of the same length starting from the same position with
    reflective arena boundary. The aggregate occupancy of the simulated walks
    is rendered next to the real occupancy, plus a difference panel (red =
    sheep concentrate more than null; blue = less).

    If sheep navigate to reward sites, the real heatmap should show hotspots
    on the canonical Config-A site positions that the simulated heatmap lacks.
    """
    _K_SIM = 20
    _DECIMATE_RW = 10
    _arena_lo, _arena_hi = 0.0, 5.0
    _bins_rw = bins_slider.value
    _dur_rw = duration_slider.value
    _rng_rw = np.random.default_rng(42)

    def _fit_rw(gx, gy):
        _dx = np.diff(gx)
        _dy = np.diff(gy)
        _s = np.sqrt(_dx**2 + _dy**2)
        _v = np.isfinite(_s) & (_s > 1e-6)
        if _v.sum() < 3:
            return _s[_v], np.array([], dtype=float)
        _h = np.arctan2(_dy[_v], _dx[_v])
        _tu = np.diff(np.unwrap(_h))
        _tu = (_tu + np.pi) % (2 * np.pi) - np.pi
        return _s[_v], _tu[np.isfinite(_tu)]

    def _reflect_rw(arr, lo, hi):
        _span = hi - lo
        _u = (arr - lo) % (2 * _span)
        return lo + np.where(_u <= _span, _u, 2 * _span - _u)

    def _sim_walks_rw(start_xy, n_steps, steps_emp, turns_emp, K, rng):
        if len(steps_emp) == 0 or n_steps <= 0:
            return np.array([]), np.array([])
        if len(turns_emp) == 0:
            turns_emp = np.array([0.0])
        _ss = rng.choice(steps_emp, size=(K, n_steps))
        _tt = rng.choice(turns_emp, size=(K, n_steps))
        _h0 = rng.uniform(-np.pi, np.pi, size=K)
        _h = _h0[:, None] + np.cumsum(_tt, axis=1)
        _dxs = _ss * np.cos(_h)
        _dys = _ss * np.sin(_h)
        _x0, _y0 = start_xy
        _xu = np.concatenate(
            [np.full((K, 1), _x0), _x0 + np.cumsum(_dxs, axis=1)], axis=1,
        )
        _yu = np.concatenate(
            [np.full((K, 1), _y0), _y0 + np.cumsum(_dys, axis=1)], axis=1,
        )
        return (
            _reflect_rw(_xu, _arena_lo, _arena_hi),
            _reflect_rw(_yu, _arena_lo, _arena_hi),
        )

    def _simulated_pool(pool, label):
        _gx_out, _gy_out = [], []
        _n_used = 0
        for _p in pool:
            _m = _p['t_rel_min'] <= _dur_rw
            _gx_r = np.asarray(_p['gx'][_m], dtype=float)[::_DECIMATE_RW]
            _gy_r = np.asarray(_p['gy'][_m], dtype=float)[::_DECIMATE_RW]
            if len(_gx_r) < 20:
                continue
            _steps, _turns = _fit_rw(_gx_r, _gy_r)
            if len(_steps) < 10:
                continue
            _sgx, _sgy = _sim_walks_rw(
                (float(_gx_r[0]), float(_gy_r[0])),
                len(_gx_r) - 1, _steps, _turns, _K_SIM, _rng_rw,
            )
            if _sgx.size == 0:
                continue
            _gx_out.append(_sgx.ravel())
            _gy_out.append(_sgy.ravel())
            _n_used += 1
        if not _gx_out:
            return np.array([]), np.array([]), 0
        return np.concatenate(_gx_out), np.concatenate(_gy_out), _n_used

    def _concat_real(pool):
        if not pool:
            return np.array([]), np.array([])
        _gxs, _gys = [], []
        for _p in pool:
            _m = _p['t_rel_min'] <= _dur_rw
            _gxs.append(_p['gx'][_m])
            _gys.append(_p['gy'][_m])
        return np.concatenate(_gxs), np.concatenate(_gys)

    _gx_real_t, _gy_real_t = _concat_real(test_points)
    _gx_sim_t, _gy_sim_t, _n_test_sheep = _simulated_pool(test_points, "test")
    _gx_real_c, _gy_real_c = _concat_real(ctrl_points)
    _gx_sim_c, _gy_sim_c, _n_ctrl_sheep = _simulated_pool(ctrl_points, "ctrl")

    def _hist_rw(gx, gy):
        if len(gx) == 0:
            return np.zeros((_bins_rw, _bins_rw))
        _H, _, _ = np.histogram2d(
            gx, gy, bins=_bins_rw, range=[[0, 5], [0, 5]],
        )
        return _H.T

    _H_rt = _hist_rw(_gx_real_t, _gy_real_t)
    _H_st = _hist_rw(_gx_sim_t, _gy_sim_t)
    # Probability normalised so real-vs-sim difference is scale-comparable.
    _P_rt = _H_rt / _H_rt.sum() if _H_rt.sum() > 0 else _H_rt
    _P_st = _H_st / _H_st.sum() if _H_st.sum() > 0 else _H_st
    _D_t = _P_rt - _P_st

    _ref_sites_rw = reward_sites_df[
        (reward_sites_df['field'] == 'A') &
        (reward_sites_df['label'].str.match(r'^A\d+$'))
    ]

    _fig_rw, _axes_rw = plt.subplots(
        1, 3, figsize=(20, 6.5), constrained_layout=True, dpi=140,
    )

    _disp_rt = np.log1p(_H_rt) if log_scale_checkbox.value else _H_rt
    _disp_st = np.log1p(_H_st) if log_scale_checkbox.value else _H_st
    _vmax_rw = float(max(_disp_rt.max(), _disp_st.max(), 1))
    _cbar_label_rw = 'log(1+count)' if log_scale_checkbox.value else 'count'

    _im_rt = _axes_rw[0].imshow(
        _disp_rt, extent=[0, 5, 0, 5], origin='lower',
        cmap=cmap_dropdown.value, aspect='equal',
        interpolation='nearest', vmin=0, vmax=_vmax_rw,
    )
    _axes_rw[0].set_title(
        f"Real test occupancy\n"
        f"{len(test_points)} sheep-trials, {int(_H_rt.sum()):,} pts",
        fontsize=10,
    )
    _fig_rw.colorbar(_im_rt, ax=_axes_rw[0], shrink=0.7, label=_cbar_label_rw)

    _im_st = _axes_rw[1].imshow(
        _disp_st, extent=[0, 5, 0, 5], origin='lower',
        cmap=cmap_dropdown.value, aspect='equal',
        interpolation='nearest', vmin=0, vmax=_vmax_rw,
    )
    _axes_rw[1].set_title(
        f"Movement-matched random-walk occupancy\n"
        f"{_n_test_sheep} sheep × K={_K_SIM}, {int(_H_st.sum()):,} pts",
        fontsize=10,
    )
    _fig_rw.colorbar(_im_st, ax=_axes_rw[1], shrink=0.7, label=_cbar_label_rw)

    _absmax_rw = float(max(abs(_D_t.min()), abs(_D_t.max()), 1e-12))
    _im_dt = _axes_rw[2].imshow(
        _D_t, extent=[0, 5, 0, 5], origin='lower',
        cmap='RdBu_r', aspect='equal',
        interpolation='nearest', vmin=-_absmax_rw, vmax=_absmax_rw,
    )
    _axes_rw[2].set_title(
        "Difference: P(real) − P(sim)\nred = sheep > null, blue = sheep < null",
        fontsize=10,
    )
    _fig_rw.colorbar(_im_dt, ax=_axes_rw[2], shrink=0.7, label='Δ probability')

    # Reward markers on real and diff panels (canonical Config-A frame).
    for _ax_rw in (_axes_rw[0], _axes_rw[2]):
        for _, _r in _ref_sites_rw.iterrows():
            _ax_rw.scatter(
                _r['grid_x'], _r['grid_y'],
                s=120, zorder=5, marker='o',
                facecolors='none', edgecolors='#FFE066', linewidths=2,
            )

    for _ax_rw in _axes_rw:
        for _v in range(1, 5):
            _ax_rw.axvline(_v, color='white', alpha=0.15, linewidth=0.5)
            _ax_rw.axhline(_v, color='white', alpha=0.15, linewidth=0.5)
        _ax_rw.set_xlim(-0.05, 5.05)
        _ax_rw.set_ylim(-0.05, 5.05)
        _ax_rw.set_xlabel("Grid X (10 m/unit)")
        _ax_rw.set_ylabel("Grid Y (10 m/unit)")

    _fig_rw.suptitle(
        f"Real vs movement-matched random-walk occupancy "
        f"(Phase 2 test pool, K={_K_SIM}/sheep)",
        fontsize=12,
    )

    print(
        f"[RW heatmap] Real test: {int(_H_rt.sum()):,} pts | "
        f"Sim test: {int(_H_st.sum()):,} pts ({_n_test_sheep} sheep × K={_K_SIM})"
    )

    mo.vstack([
        mo.md("### Random-walk occupancy comparison (test pool)"),
        mo.md(
            "Per-sheep movement-matched correlated random walks aggregated "
            "into a heatmap; compare against the real heatmap. Red regions "
            "in the difference panel are where sheep spend MORE time than "
            "their own movement statistics would predict — these are the "
            "spatial-memory hotspots, expected near reward sites."
        ),
        _fig_rw,
    ])


if __name__ == "__main__":
    app.run()
