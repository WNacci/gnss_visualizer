import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(Path, pd):
    # Load trial data from the cleaned CSV
    _csv_path = str(Path(__file__).parent.parent / "data" / "experimental" / "Sheep_Trial_Data.csv")
    _all_data = pd.read_csv(_csv_path, dtype={"Sheep ID": str})

    # Build device-to-sheep mappings and trial definitions from CSV
    def _build_mappings_and_trials(df):
        _mappings = {}
        _trials = []
        _trial_num = 0

        # Only rows with a start_time define a trial
        _with_start = df[df['start_time'].notna()].copy()

        for (_date, _start_time, _group_num, _group_size), _group in _with_start.groupby(
            ['date', 'start_time', 'Group #', 'Group Size'], dropna=False
        ):
            # Device-to-sheep mapping for this trial
            _device_to_sheep = {}
            for _, _row in _group.iterrows():
                _sheep_id = _row['Sheep ID'] if pd.notna(_row['Sheep ID']) else 'Unknown'
                if pd.notna(_row['GNSS_SN1']):
                    _device_to_sheep[int(_row['GNSS_SN1'])] = _sheep_id
                if pd.notna(_row['GNSS_SN2']):
                    _device_to_sheep[int(_row['GNSS_SN2'])] = _sheep_id
            _mappings[_trial_num] = _device_to_sheep

            # Trial definition
            _devices = sorted(set(
                [int(v) for v in _group['GNSS_SN1'].dropna()] +
                [int(v) for v in _group['GNSS_SN2'].dropna()]
            ))
            _field = str(_group['field'].iloc[0]) if pd.notna(_group['field'].iloc[0]) else 'Unknown'
            _config = str(_group['configuration'].iloc[0]) if pd.notna(_group['configuration'].iloc[0]) else 'Unknown'
            _g_num = int(_group_num) if pd.notna(_group_num) else 0
            _g_size = int(_group_size) if pd.notna(_group_size) else 0
            _assay_val = _group['assay'].iloc[0]
            _assay = None
            if pd.notna(_assay_val):
                try:
                    _assay = int(_assay_val)
                except (ValueError, TypeError):
                    _assay = str(_assay_val)
            _notes_list = _group['note'].dropna().unique()
            _notes = f"Group {_g_num}, Size {_g_size}"
            if len(_notes_list) > 0:
                _notes += f" - {'; '.join(_notes_list)}"

            _name = f"{_date} - Field {_field}, {_config}, {_g_size} sheep"
            _trials.append((_name, _date, _field, _config, str(_start_time), 35, _devices, _assay, _notes))
            _trial_num += 1

        return _mappings, _trials

    DEVICE_TO_SHEEP, TRIALS = _build_mappings_and_trials(_all_data)

    print(f"Loading experimental data from CSV...")
    print(f"\n{'='*70}")
    print(f"Available Trials ({len(TRIALS)} total):")
    print('='*70)
    for _i, _trial in enumerate(TRIALS):
        _name, _date, _field, _config, _start, _dur, _devs, _assay, _notes = _trial
        _assay_str = f" [Assay {_assay}]" if _assay is not None else ""
        print(f"  [{_i:2d}] {_notes.split(' - ')[0]:20s} {_name}{_assay_str}")
    print('='*70)
    print("\nTo select trials, use their IDs in ACTIVE_TRIALS below.")
    print("Example: ACTIVE_TRIALS = [0, 4, 10]")

    # Function to convert trial info to data source format
    def trial_to_source(trial_name, date, field, config, start_time_str, duration_min, devices, assay, notes, decimation=10):
        """Convert trial definition to data source tuple."""
        import pandas as pd

        # Construct path based on date
        _date_parts = date.split("-")
        _day = int(_date_parts[2])
        _path = str(Path(__file__).parent.parent / "data" / "gnss" / f"{_day}-02-26")

        # Convert Paris time to UTC Unix timestamp
        _datetime_str = f"{date} {start_time_str}"
        _start_paris = pd.to_datetime(_datetime_str)
        _start_utc = _start_paris.tz_localize('Europe/Paris').tz_convert('UTC')
        _start_unix = _start_utc.timestamp()

        # Calculate end time
        _end_unix = _start_unix + (duration_min * 60)

        return (_path, _start_unix, _end_unix, devices, decimation)
    return DEVICE_TO_SHEEP, TRIALS, trial_to_source


@app.cell(hide_code=True)
def _(TRIALS, mo):
    # Build options dict: display label -> trial index
    _options = {}
    for _i, _trial in enumerate(TRIALS):
        _name, _date, _field, _config, _start, _dur, _devs, _assay, _notes = _trial
        _assay_str = f" [Assay {_assay}]" if _assay is not None else ""
        _label = f"[{_i}] {_notes.split(' - ')[0]:20s} {_name}{_assay_str}"
        _options[_label] = _i

    trial_selector = mo.ui.multiselect(options=_options, label="Select trials", full_width=True)
    include_field = mo.ui.checkbox(label="Include field data", value=False)
    include_field_merged = mo.ui.checkbox(label="Include field_merged data", value=False)
    decimation_slider = mo.ui.slider(start=1, stop=50, step=1, value=10, label="Decimation factor")

    mo.md(f"""
    ### Dataset Selection
    {trial_selector}

    {include_field} {include_field_merged}

    {decimation_slider}
    """)
    return decimation_slider, include_field, include_field_merged, trial_selector


@app.cell(hide_code=True)
def _(Path, TRIALS, decimation_slider, include_field, include_field_merged, trial_to_source, trial_selector):
    # Convert selected trials to data sources
    DATA_SOURCES = []
    _decimation = decimation_slider.value

    for _trial_idx in trial_selector.value:
        _trial = TRIALS[_trial_idx]
        # Override decimation in trial_to_source
        _src = trial_to_source(*_trial)
        # Replace the default decimation with user-selected value
        DATA_SOURCES.append((_src[0], _src[1], _src[2], _src[3], _decimation))

    if include_field_merged.value:
        _path = str(Path(__file__).parent.parent / "data" / "gnss" / "field_merged")
        DATA_SOURCES.append((_path, None, None, None, _decimation))

    if include_field.value:
        _path = str(Path(__file__).parent.parent / "data" / "gnss" / "gnss_data_field")
        DATA_SOURCES.append((_path, None, None, None, _decimation))

    print(f"Selected {len(DATA_SOURCES)} source(s), decimation={_decimation}")
    return (DATA_SOURCES,)


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
    import geopandas as gpd
    from lonboard import Map, PathLayer, ScatterplotLayer
    from lonboard.basemap import MaplibreBasemap
    from lonboard.view_state import MapViewState

    COLORS = [
        [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
        [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
        [153, 153, 153], [0, 206, 209], [139, 69, 19], [0, 100, 0],
        [75, 0, 130], [220, 20, 60], [0, 139, 139], [184, 134, 11],
    ]

    def load_device_data_from_dir(device_dir):
        """Load GPS data from a device subdirectory."""
        _lats, _lons, _times = [], [], []
        for f in sorted(device_dir.iterdir(), key=lambda x: x.name):
            if re.match(r"LOGS\d+\.TXT", f.name, re.IGNORECASE) and f.stat().st_size > 0:
                for line in open(f, errors='ignore'):
                    parts = line.split(":")
                    if len(parts) >= 6:
                        try:
                            lat = float(parts[3])
                            lon = float(parts[4])
                            time = float(parts[5])
                            _lats.append(lat)
                            _lons.append(lon)
                            _times.append(time)
                        except ValueError:
                            pass
        return np.array(_lats), np.array(_lons), np.array(_times)

    def load_device_data_from_files(files):
        """Load GPS data from a list of files."""
        _lats, _lons, _times = [], [], []
        for f in sorted(files, key=lambda x: x.name):
            if f.stat().st_size > 0:
                for line in open(f, errors='ignore'):
                    parts = line.split(":")
                    if len(parts) >= 6:
                        try:
                            lat = float(parts[3])
                            lon = float(parts[4])
                            time = float(parts[5])
                            _lats.append(lat)
                            _lons.append(lon)
                            _times.append(time)
                        except ValueError:
                            pass
        return np.array(_lats), np.array(_lons), np.array(_times)

    def detect_format_and_load(data_dir):
        """Detect directory format and load data."""
        if not data_dir.exists():
            return {}, 0.0, 1.0

        flat_files = list(data_dir.glob("GNSS_*_LOGS*.TXT"))

        if flat_files:
            device_name = data_dir.name
            _lats, _lons, _times = load_device_data_from_files(flat_files)
            if len(_lats) > 0:
                return {device_name: (_lats, _lons, _times)}, float(_times.min()), float(_times.max())
            return {}, 0.0, 1.0
        else:
            device_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("GNSS")])
            if device_dirs:
                _data = {}
                _all_times = []
                for device_dir in device_dirs:
                    _lats, _lons, _times = load_device_data_from_dir(device_dir)
                    if len(_lats) > 0:
                        _data[device_dir.name] = (_lats, _lons, _times)
                        _all_times.append(_times)
                if _all_times:
                    _combined = np.concatenate(_all_times)
                    return _data, float(_combined.min()), float(_combined.max())
            return {}, 0.0, 1.0
    return (
        COLORS,
        Map,
        MapViewState,
        MaplibreBasemap,
        Path,
        PathLayer,
        ScatterplotLayer,
        detect_format_and_load,
        gpd,
        np,
        pd,
        re,
    )


@app.cell(hide_code=True)
def _(Path, PathLayer, ScatterplotLayer, gpd, np, pd):
    from shapely.geometry import LineString

    _csv_path = str(Path(__file__).parent.parent / "data" / "fitted_reward_sites.csv")
    reward_sites_df = pd.read_csv(_csv_path)

    def build_reward_layers(field, radius, site_color, border_color):
        """Build scatterplot + border layers for a given field's reward sites."""
        _field_df = reward_sites_df[reward_sites_df['field'] == field]
        _sites = _field_df[~_field_df['label'].str.startswith('E')]
        _corners = _field_df[_field_df['label'].str.startswith('E')].sort_values('label')

        layers = []

        # Reward site circles
        if len(_sites) > 0:
            _site_colors = np.tile(np.array(site_color + [200], dtype=np.uint8), (len(_sites), 1))
            _gdf = gpd.GeoDataFrame(
                {'label': _sites['label'].values},
                geometry=gpd.points_from_xy(_sites['longitude'], _sites['latitude']),
                crs="EPSG:4326",
            )
            layers.append(ScatterplotLayer.from_geopandas(
                _gdf, get_fill_color=_site_colors, get_radius=radius,
                radius_min_pixels=2, radius_max_pixels=50,
                pickable=True, auto_highlight=True,
            ))

        # Arena border as a closed path
        if len(_corners) >= 3:
            _coords = list(zip(_corners['longitude'], _corners['latitude']))
            _coords.append(_coords[0])  # close the loop
            _line = LineString(_coords)
            _border_gdf = gpd.GeoDataFrame(
                {'label': [f'Field {field} border']},
                geometry=[_line],
                crs="EPSG:4326",
            )
            _border_colors = np.array([[border_color[0], border_color[1], border_color[2], 180]], dtype=np.uint8)
            layers.append(PathLayer.from_geopandas(
                _border_gdf, get_color=_border_colors, width_min_pixels=2,
                pickable=True, auto_highlight=True,
            ))

        return layers
    return (build_reward_layers,)


@app.cell(hide_code=True)
def _(mo):
    show_reward_sites = mo.ui.checkbox(label="Show reward sites & arena border", value=True)
    reward_site_radius = mo.ui.slider(start=1, stop=20, step=0.5, value=5, label="Reward site radius (m)")
    mo.md(f"""
    ### Reward Site Overlay
    {show_reward_sites}
    {reward_site_radius}
    """)
    return reward_site_radius, show_reward_sites


@app.cell
def _(mo):
    mo.md("""
    # Multi-Source GPS Data Visualization
    """)
    return


@app.cell(hide_code=True)
def _(DATA_SOURCES, Path, TRIALS, detect_format_and_load, pd, re):
    def normalize_device_id(device_id):
        """Convert device ID to standard format. Accepts int (1-16) or str ('GNSS_1'/'GNSS-1')."""
        if isinstance(device_id, int):
            # Try both hyphen and underscore formats
            return [f"GNSS-{device_id}", f"GNSS_{device_id}"]
        # Return both formats for string input too
        _id = str(device_id)
        if '-' in _id or '_' in _id:
            _num = _id.replace('GNSS-', '').replace('GNSS_', '')
            return [f"GNSS-{_num}", f"GNSS_{_num}"]
        return [_id]

    def find_matching_devices(requested_devices, available_devices):
        """Find available devices that match the requested list."""
        if requested_devices is None:
            return sorted(available_devices)

        _matched = []
        for _req in requested_devices:
            _possible_formats = normalize_device_id(_req)
            # Try all possible formats
            for _format in _possible_formats:
                if _format in available_devices:
                    _matched.append(_format)
                    break
                else:
                    # Try case-insensitive match
                    for _avail in available_devices:
                        if _avail.lower() == _format.lower():
                            _matched.append(_avail)
                            break
                    if _matched and _matched[-1]:  # If we found a match, break
                        break
        return _matched

    def find_trial_info(path, start_time):
        """Find trial info matching the path and start time."""
        for trial_idx, trial in enumerate(TRIALS):
            _name, _date, _field, _config, _start_str, _dur, _devs, _assay, _notes = trial
            _date_parts = _date.split("-")
            _day = int(_date_parts[2])
            _expected_path = str(Path(__file__).parent.parent / "data" / "gnss" / f"{_day}-02-26")

            # Convert trial start time to unix
            _datetime_str = f"{_date} {_start_str}"
            _start_paris = pd.to_datetime(_datetime_str)
            _start_utc = _start_paris.tz_localize('Europe/Paris').tz_convert('UTC')
            _start_unix = _start_utc.timestamp()

            if path == _expected_path and abs(start_time - _start_unix) < 60:  # Within 1 minute
                # Extract group number and size from notes
                _group_match = re.search(r'Group (\d+)', _notes)
                _size_match = re.search(r'Size (\d+)', _notes)
                _group_num = int(_group_match.group(1)) if _group_match else None
                _group_size = int(_size_match.group(1)) if _size_match else None
                return _name, _assay, _notes, _group_num, _group_size, trial_idx
        return None, None, None, None, None, None

    # Auto-load all data sources
    sources = []
    for i, (path, start_time, end_time, device_list, decimation) in enumerate(DATA_SOURCES):
        _p = Path(path)
        _data, _min_time, _max_time = detect_format_and_load(_p)

        # Use provided times or fall back to data min/max
        _start = start_time if start_time is not None else _min_time
        _end = end_time if end_time is not None else _max_time

        # Match requested devices to available devices
        _devices = find_matching_devices(device_list, list(_data.keys()))

        # Try to find trial info
        _trial_name, _assay, _notes, _group_num, _group_size, _trial_idx = find_trial_info(str(path), _start)

        if _data:
            if _trial_name:
                print(f"\n✓ {_trial_name}")
                if _assay is not None:
                    try:
                        print(f"  Assay: {int(float(_assay))}")
                    except (ValueError, TypeError):
                        print(f"  Assay: {_assay}")
                if _notes:
                    print(f"  Notes: {_notes}")
            else:
                print(f"\n✓ Source {i+1}: {_p.name}")

            print(f"  Available devices ({len(_data)}): {', '.join(sorted(_data.keys()))}")
            if device_list is not None:
                print(f"  Requested: {device_list}")
                print(f"  Using devices ({len(_devices)}): {', '.join(_devices)}")
            else:
                print(f"  Using all devices ({len(_devices)})")
            print(f"  Data range UTC: {pd.Timestamp(_min_time, unit='s', tz='UTC')} to {pd.Timestamp(_max_time, unit='s', tz='UTC')}")
            print(f"  Using time range UTC: {pd.Timestamp(_start, unit='s', tz='UTC')} to {pd.Timestamp(_end, unit='s', tz='UTC')}")
            print(f"  Using time range Paris: {pd.Timestamp(_start, unit='s', tz='UTC').tz_convert('Europe/Paris')} to {pd.Timestamp(_end, unit='s', tz='UTC').tz_convert('Europe/Paris')}")
            print(f"  Decimation: {decimation}")
            sources.append({
                'name': _trial_name if _trial_name else _p.name,
                'data': _data,
                'start_time': _start,
                'end_time': _end,
                'devices': _devices,
                'decimation': decimation,
                'assay': _assay,
                'group_num': _group_num,
                'group_size': _group_size,
                'trial_idx': _trial_idx,
                'index': i
            })
        else:
            print(f"\n✗ No data in: {path}")

    print(f"\n{'='*60}")
    print(f"Loaded {len(sources)} sources total")
    return (sources,)


@app.cell
def _(mo):
    focus_btn = mo.ui.button(label="🎯 Focus Map")
    mo.md(f"## Map\n{focus_btn}")
    return (focus_btn,)


@app.cell(hide_code=True)
def _(
    COLORS,
    DEVICE_TO_SHEEP,
    Map,
    MapViewState,
    MaplibreBasemap,
    ScatterplotLayer,
    build_reward_layers,
    focus_btn,
    gpd,
    np,
    pd,
    re,
    reward_site_radius,
    show_reward_sites,
    sources,
):
    _layers = []
    _color_idx = 0
    _total = 0
    _all_lats = []
    _all_lons = []

    # Render all sources
    for _src in sources:
        for _dev in _src['devices']:
            if _dev in _src['data']:
                _lats, _lons, _times = _src['data'][_dev]

                # Apply time filter
                _mask = (_times >= _src['start_time']) & (_times <= _src['end_time'])
                _lats, _lons, _times = _lats[_mask], _lons[_mask], _times[_mask]

                if len(_lats) > 0:
                    # Apply decimation
                    _dec = _src['decimation']
                    _lats, _lons, _times = _lats[::_dec], _lons[::_dec], _times[::_dec]
                    _total += len(_lats)

                    # Track for bounds calculation
                    _all_lats.extend(_lats)
                    _all_lons.extend(_lons)

                    # Assign color
                    _color = COLORS[_color_idx % len(COLORS)]
                    _colors = np.column_stack([
                        np.full(len(_lats), _color[0], dtype=np.uint8),
                        np.full(len(_lats), _color[1], dtype=np.uint8),
                        np.full(len(_lats), _color[2], dtype=np.uint8),
                        np.full(len(_lats), 200, dtype=np.uint8)
                    ])

                    # Extract device number and get sheep ID
                    _dev_match = re.search(r'(\d+)', _dev)
                    _dev_num = int(_dev_match.group(1)) if _dev_match else None
                    _sheep_id = 'Unknown'
                    if _src['trial_idx'] is not None and _dev_num is not None:
                        _sheep_id = DEVICE_TO_SHEEP.get(_src['trial_idx'], {}).get(_dev_num, 'Unknown')

                    # Create tooltip data
                    _tooltip_data = {
                        'device': _dev,
                        'sheep_id': _sheep_id,
                        'group': _src['group_num'] if _src['group_num'] is not None else 'Unknown',
                        'group_size': _src['group_size'] if _src['group_size'] is not None else 'Unknown',
                        'assay': _src['assay'] if _src['assay'] is not None else 'N/A',
                        'time_utc': pd.to_datetime(_times, unit='s', utc=True).strftime('%Y-%m-%d %H:%M:%S UTC'),
                        'time_paris': pd.to_datetime(_times, unit='s', utc=True).tz_convert('Europe/Paris').strftime('%Y-%m-%d %H:%M:%S CET'),
                    }

                    # Create layer
                    _gdf = gpd.GeoDataFrame(_tooltip_data, geometry=gpd.points_from_xy(_lons, _lats), crs="EPSG:4326")

                    _layers.append(ScatterplotLayer.from_geopandas(
                        _gdf, get_fill_color=_colors, get_radius=2,
                        radius_min_pixels=1, radius_max_pixels=3,
                        pickable=True, auto_highlight=True
                    ))
                    _color_idx += 1

    # Add reward site overlay if toggled on
    if show_reward_sites.value:
        _radius = reward_site_radius.value
        for _field in ['A', 'B']:
            _site_color = [255, 0, 0] if _field == 'A' else [0, 0, 255]
            _border_color = [255, 100, 100] if _field == 'A' else [100, 100, 255]
            _reward_layers = build_reward_layers(_field, _radius, _site_color, _border_color)
            _layers.extend(_reward_layers)
        print(f"Reward site overlay: fields A, B, radius={_radius}m")

    print(f"Displaying {_total:,} points from {len(_layers)} layers")

    # Calculate view state based on focus button
    _view_state = None
    _clicked = focus_btn.value
    if _clicked is not None and _clicked > 0 and _all_lats and _all_lons:
        # Calculate center and bounds
        _center_lat = (min(_all_lats) + max(_all_lats)) / 2
        _center_lon = (min(_all_lons) + max(_all_lons)) / 2

        # Calculate zoom level based on bounds
        _lat_span = max(_all_lats) - min(_all_lats)
        _lon_span = max(_all_lons) - min(_all_lons)
        _max_span = max(_lat_span, _lon_span)

        if _max_span > 1:
            _zoom = 8
        elif _max_span > 0.1:
            _zoom = 12
        elif _max_span > 0.01:
            _zoom = 14
        else:
            _zoom = 16

        _view_state = MapViewState(
            longitude=_center_lon,
            latitude=_center_lat,
            zoom=_zoom,
            bearing=100,
        )
        print(f"🎯 Focused map on center: ({_center_lat:.6f}, {_center_lon:.6f}), zoom: {_zoom}")

    # Default view state with bearing rotation so grid Y-axis faces north
    if _view_state is None and _all_lats and _all_lons:
        _view_state = MapViewState(
            longitude=(min(_all_lons) + max(_all_lons)) / 2,
            latitude=(min(_all_lats) + max(_all_lats)) / 2,
            zoom=16,
            bearing=100,
        )

    # Create map with layers and view state
    _basemap = MaplibreBasemap(style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json")
    _map_kwargs = dict(
        layers=_layers,
        basemap=_basemap,
        height=800,
        picking_radius=5,
        show_tooltip=True,
    )
    gps_map = Map(**_map_kwargs)
    if _view_state:
        gps_map.set_view_state(view_state=_view_state)

    gps_map
    return


if __name__ == "__main__":
    app.run()
