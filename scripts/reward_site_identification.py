import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    # All imports at the top - marimo requires explicit return of all objects
    # that will be used in subsequent cells
    import pandas as pd
    from datetime import datetime, timezone
    import numpy as np
    from pathlib import Path
    import geopandas as gpd
    from lonboard import Map, ScatterplotLayer
    from lonboard.basemap import MaplibreBasemap
    import marimo as mo
    from gps_analysis import DATA_DIR, load_gnss_dir
    return (
        DATA_DIR,
        Map,
        MaplibreBasemap,
        Path,
        ScatterplotLayer,
        datetime,
        gpd,
        load_gnss_dir,
        mo,
        np,
        pd,
        timezone,
    )


@app.cell(hide_code=True)
def _(datetime, mo, pd, timezone):
    # Reward site configuration from reward_sites.md
    # Field A: GNSS time 09:38-10:05 UTC
    # Field B: GNSS time 09:00-09:30 UTC

    # Convert time strings to Unix timestamps for 2026-01-26
    # Using UTC timezone to match GNSS timestamps
    date_str = "2026-01-26"

    # Field A time range: 09:38-10:09 UTC (extended to capture more data)
    field_a_start = datetime(2026, 1, 26, 9, 38, 0, tzinfo=timezone.utc).timestamp()
    field_a_end = datetime(2026, 1, 26, 10, 9, 0, tzinfo=timezone.utc).timestamp()

    # Field B time range: 09:00-09:30 UTC
    field_b_start = datetime(2026, 1, 26, 9, 0, 0, tzinfo=timezone.utc).timestamp()
    field_b_end = datetime(2026, 1, 26, 9, 30, 0, tzinfo=timezone.utc).timestamp()

    print(f"Field A time range (UTC): {pd.Timestamp(field_a_start, unit='s', tz='UTC')} to {pd.Timestamp(field_a_end, unit='s', tz='UTC')}")
    print(f"Field B time range (UTC): {pd.Timestamp(field_b_start, unit='s', tz='UTC')} to {pd.Timestamp(field_b_end, unit='s', tz='UTC')}")

    # Reward site definitions: (label, gnss_id, field)
    # From reward_sites.md - each site has a unique GNSS device for calibration
    reward_sites = [
        # Field A sites (recorded 09:38-10:05 UTC)
        ("B1", 4, "A"),
        ("A3", 10, "A"),
        ("C1", 6, "A"),
        ("D2", 7, "A"),
        ("C2", 1, "A"),
        ("D3", 3, "A"),
        ("A1", 16, "A"),
        ("B2", 15, "A"),
        ("A2", 8, "A"),
        ("B3", 12, "A"),
        ("D1", 9, "A"),
        ("C3", 11, "A"),
        # Field B sites (recorded 09:00-09:30 UTC)
        ("B1", 9, "B"),
        ("A3", 6, "B"),
        ("C1", 13, "B"),
        ("D2", 3, "B"),
        ("C2", 8, "B"),
        ("D3", 16, "B"),
        ("A1", 14, "B"),
        ("B2", 12, "B"),
        ("A2", 1, "B"),
        ("B3", 5, "B"),
        ("D1", 11, "B"),
        ("C3", 2, "B"),
    ]

    # Define expected grid positions for each label (x, y in 10m units)
    # The grid is 50x50m with 10m spacing in an octagonal configuration (4x4 with corners missing)
    # Origin (0,0) would be at the bottom-left if it existed
    # y increases upward (North), x increases rightward (East)
    label_to_grid = {
        'B1': (2, 4), 'A3': (3, 4),
        'C1': (1, 3), 'D2': (2, 3), 'C2': (3, 3), 'D3': (4, 3),
        'A1': (1, 2), 'B2': (2, 2), 'A2': (3, 2), 'B3': (4, 2),
        'D1': (2, 1), 'C3': (3, 1)
    }

    mo.md("""
    # Reward Site Location Identification

    This script identifies the precise GPS coordinates of reward sites by:
    1. Loading GNSS data for each device during the calibration period
    2. Filtering by the recorded time ranges for each field
    3. Averaging coordinates using different portions:
       - Field A: last 10% of data
       - Field B: last 50% of data
    """)
    return (
        field_a_end,
        field_a_start,
        field_b_end,
        field_b_start,
        label_to_grid,
        reward_sites,
    )


@app.cell(hide_code=True)
def _(
    Path,
    field_a_end,
    field_a_start,
    field_b_end,
    field_b_start,
    label_to_grid,
    np,
    pd,
    reward_sites,
):
    def load_device_data(device_id):
        """Load GPS data for a specific GNSS device from gnss_data_field.

        Tries both naming formats: GNSS_01 (zero-padded) and GNSS-1 (no padding).
        Returns (lats, lons, times) or (None, None, None) if no data found.
        """
        base_path = DATA_DIR / "gnss" / "gnss_data_field"
        for device_dir in [base_path / f"GNSS_{device_id:02d}", base_path / f"GNSS-{device_id}"]:
            if device_dir.exists():
                lats, lons, times = load_gnss_dir(device_dir)
                if len(lats) > 0:
                    return lats, lons, times
        print(f"Warning: No data directory found for device {device_id}")
        return None, None, None

    def filter_data_by_time(lats, lons, times, start_time, end_time):
        """Filter GPS data by Unix timestamp range."""
        mask = (times >= start_time) & (times <= end_time)
        return lats[mask], lons[mask], times[mask]

    def get_average_and_raw_data(lats, lons, times, use_later_portion=0.5):
        """
        Calculate average location from a subset of data points.

        Args:
            lats: Array of latitudes
            lons: Array of longitudes
            times: Array of timestamps
            use_later_portion: Fraction of data to use from the end (0.5 = last 50%)
                              Higher precision is expected later in recordings
                              as GNSS devices acquire better satellite locks

        Returns:
            Tuple of (avg_lat, avg_lon, raw_lats, raw_lons)
        """
        if len(lats) == 0:
            return None, None, None, None

        n_points = len(lats)
        start_idx = int(n_points * (1 - use_later_portion))

        raw_lats = lats[start_idx:]
        raw_lons = lons[start_idx:]
        avg_lat = np.median(raw_lats)
        avg_lon = np.median(raw_lons)

        return avg_lat, avg_lon, raw_lats, raw_lons

    # =============================================================================
    # Process All Reward Sites
    # =============================================================================
    # For each reward site:
    # 1. Load GPS data from the corresponding GNSS device
    # 2. Filter by the field's time range
    # 3. Calculate average position using the later portion of data
    # 4. Store results including raw points for visualization

    reward_site_locations = []

    print("Processing reward sites...")
    print("=" * 70)

    for _label, _gnss_id, _field in reward_sites:
        # Load data for this device
        _lats, _lons, _times = load_device_data(_gnss_id)

        if _lats is None:
            continue

        # Get time range for this field
        if _field == "A":
            _start_time, _end_time = field_a_start, field_a_end
        else:
            _start_time, _end_time = field_b_start, field_b_end

        # Filter data by time range
        _filtered_lats, _filtered_lons, _filtered_times = filter_data_by_time(
            _lats, _lons, _times, _start_time, _end_time
        )

        if len(_filtered_lats) == 0:
            print(f"Warning: No data in time range for {_label} (GNSS {_gnss_id}, Field {_field})")
            continue

        # Calculate average location and get raw data
        # Field A uses last 10%, Field B uses last 50%
        _portion = 0.1 if _field == "A" else 0.5
        _avg_lat, _avg_lon, _raw_lats, _raw_lons = get_average_and_raw_data(
            _filtered_lats, _filtered_lons, _filtered_times, use_later_portion=_portion
        )

        # Get expected grid position
        _grid_x, _grid_y = label_to_grid.get(_label, (0, 0))

        # Store result with raw data for visualization
        reward_site_locations.append({
            'label': _label,
            'field': _field,
            'gnss_id': _gnss_id,
            'grid_x': _grid_x,
            'grid_y': _grid_y,
            'latitude': _avg_lat,
            'longitude': _avg_lon,
            'raw_lats': _raw_lats,
            'raw_lons': _raw_lons,
            'n_points': len(_filtered_lats),
            'n_raw_points': len(_raw_lats),
            'time_start': pd.Timestamp(_filtered_times[0], unit='s', tz='UTC'),
            'time_end': pd.Timestamp(_filtered_times[-1], unit='s', tz='UTC'),
        })

        _portion_str = "10%" if _field == "A" else "50%"
        print(f"✓ {_label} (Field {_field}, GNSS {_gnss_id:02d}, grid {_grid_x},{_grid_y}): {_avg_lat:.7f}, {_avg_lon:.7f} ({len(_filtered_lats)} points, {len(_raw_lats)} used for avg - last {_portion_str})")

    print("=" * 70)
    print(f"Processed {len(reward_site_locations)} reward sites")
    return (reward_site_locations,)


@app.cell(hide_code=True)
def _(datetime, np, pd, reward_site_locations, timezone):
    # Reference timestamps for grid alignment verification
    # Format: (field, gnss_id, timestamp1, timestamp2, description)
    # These timestamps capture when devices were moved along grid axes
    reference_points = [
        # Field A: GNSS 14 (A1 in Field A) - Y-axis movement
        ("A", 14, datetime(2026, 1, 26, 9, 49, 4, tzinfo=timezone.utc).timestamp(), 
         datetime(2026, 1, 26, 9, 49, 30, tzinfo=timezone.utc).timestamp(), "Field A reference"),
        # Field B: GNSS 10 (A3 in Field B) - Y-axis movement
        ("B", 10, datetime(2026, 1, 26, 9, 22, 4, tzinfo=timezone.utc).timestamp(),
         datetime(2026, 1, 26, 9, 22, 30, tzinfo=timezone.utc).timestamp(), "Field B reference"),
    ]

    def get_location_at_time(gnss_id, target_time, tolerance_seconds=5):
        """
        Get the location of a device at a specific timestamp.

        Searches through log files to find the closest GPS reading
        within the specified tolerance window.
        """
        base_path = DATA_DIR / "gnss" / "gnss_data_field"

        device_dirs = [
            base_path / f"GNSS_{gnss_id:02d}",
            base_path / f"GNSS-{gnss_id}",
        ]

        device_dir = None
        for d in device_dirs:
            if d.exists():
                device_dir = d
                break

        if device_dir is None:
            return None, None

        # Find closest point in time
        closest_lat, closest_lon, closest_diff = None, None, float('inf')

        for log_file in sorted(device_dir.glob("LOGS*.TXT")):
            if log_file.stat().st_size == 0:
                continue
            with open(log_file, errors='ignore') as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) >= 6:
                        try:
                            lat = float(parts[3])
                            lon = float(parts[4])
                            timestamp = float(parts[5])
                            diff = abs(timestamp - target_time)
                            if diff < closest_diff and diff <= tolerance_seconds:
                                closest_diff = diff
                                closest_lat = lat
                                closest_lon = lon
                        except ValueError:
                            pass

        return closest_lat, closest_lon

    def rotate_points(lats, lons, angle_rad):
        """Rotate points around their centroid by given angle in radians."""
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)

        # Convert to local meters (approximate)
        lat_m = 111000  # meters per degree latitude
        lon_m = 111000 * np.cos(np.radians(center_lat))  # meters per degree longitude

        # Center points
        x = (lons - center_lon) * lon_m
        y = (lats - center_lat) * lat_m

        # Rotate
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

        # Convert back to lat/lon
        lons_rot = center_lon + x_rot / lon_m
        lats_rot = center_lat + y_rot / lat_m

        return lats_rot, lons_rot

    def calculate_grid_error(locs, expected_spacing_m=10):
        """
        Calculate how well points fit expected 10m grid.

        Analyzes pairwise distances to verify if they match expected grid spacing.
        """
        # Group by field
        for field in ['A', 'B']:
            field_locs = [l for l in locs if l['field'] == field]
            if len(field_locs) < 2:
                continue

            # Calculate all pairwise distances manually
            dists_m = []
            lat_m = 111000  # meters per degree latitude

            for i in range(len(field_locs)):
                for j in range(i+1, len(field_locs)):
                    lat1, lon1 = field_locs[i]['latitude'], field_locs[i]['longitude']
                    lat2, lon2 = field_locs[j]['latitude'], field_locs[j]['longitude']
                    lon_m = 111000 * np.cos(np.radians((lat1 + lat2) / 2))
                    dist = np.sqrt(((lat2 - lat1) * lat_m)**2 + ((lon2 - lon1) * lon_m)**2)
                    dists_m.append(dist)

            dists_m = np.array(dists_m)

            print(f"\n{field} Grid distance analysis:")
            print(f"  Measured distances (m): min={dists_m.min():.1f}, max={dists_m.max():.1f}")
            print(f"  Sample distances: {sorted(dists_m)[:5]}")

            # Check how many are close to expected 10m spacing
            close_to_10m = np.sum(np.abs(dists_m - 10) < 2)
            close_to_14m = np.sum(np.abs(dists_m - 14.14) < 2)
            close_to_20m = np.sum(np.abs(dists_m - 20) < 2)
            print(f"  Pairs close to 10m (±2m): {close_to_10m}")
            print(f"  Pairs close to 14.14m diagonal (±2m): {close_to_14m}")
            print(f"  Pairs close to 20m (±2m): {close_to_20m}")

    # Get reference points
    print("\n" + "=" * 70)
    print("Reference Point Analysis")
    print("=" * 70)

    ref_locations = []
    for field, gnss_id, t1, t2, desc in reference_points:
        lat1, lon1 = get_location_at_time(gnss_id, t1)
        lat2, lon2 = get_location_at_time(gnss_id, t2)

        if lat1 is not None and lat2 is not None:
            # Calculate vector between the two reference points
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            dist_m = np.sqrt((dlat * 111000)**2 + (dlon * 111000 * np.cos(np.radians(lat1)))**2)
            angle = np.arctan2(dlon, dlat)

            print(f"\n{desc} (GNSS {gnss_id:02d}):")
            print(f"  Time 1 ({pd.Timestamp(t1, unit='s', tz='UTC').strftime('%H:%M:%S')}): {lat1:.7f}, {lon1:.7f}")
            print(f"  Time 2 ({pd.Timestamp(t2, unit='s', tz='UTC').strftime('%H:%M:%S')}): {lat2:.7f}, {lon2:.7f}")
            print(f"  Distance: {dist_m:.2f}m, Angle: {np.degrees(angle):.1f}°")

            ref_locations.append({
                'field': field,
                'gnss_id': gnss_id,
                'lat1': lat1, 'lon1': lon1,
                'lat2': lat2, 'lon2': lon2,
                'angle': angle,
                'dist_m': dist_m
            })
        else:
            print(f"  Could not find data for {desc} (GNSS {gnss_id:02d})")

    # Calculate grid errors
    calculate_grid_error(reward_site_locations)

    return


@app.cell
def _(mo, pd, reward_site_locations):
    # Display results as a table
    _df = pd.DataFrame(reward_site_locations)
    _df = _df[['label', 'field', 'gnss_id', 'grid_x', 'grid_y', 'latitude', 'longitude', 'n_points']]
    _df.columns = ['Label', 'Field', 'GNSS ID', 'Grid X', 'Grid Y', 'Latitude', 'Longitude', 'Data Points']

    mo.md(f"## Reward Site Coordinates\n\n{mo.as_html(_df)}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Reward Site Map

    Hover over points to see label, field, and coordinate information.

    - **Small semi-transparent circles**: Averaged location for each reward site
    - **Small semi-transparent dots**: Raw GPS data points (Field A: last 10%, Field B: last 50%)
    - **Blue**: Field A
    - **Red**: Field B
    """)
    return


@app.cell(hide_code=True)
def _(Map, MaplibreBasemap, ScatterplotLayer, gpd, np, reward_site_locations):
    # Create visualization of reward sites with raw data

    # Colors for different fields - using ColorBrewer Set1 colors
    field_colors = {
        'A': [55, 126, 184],   # Blue for Field A
        'B': [228, 26, 28],    # Red for Field B
    }

    _layers = []
    _all_lats = []
    _all_lons = []
    _total_raw_points = 0

    # First pass: add raw data points (semi-transparent, smaller)
    for _site in reward_site_locations:
        _raw_lats = _site['raw_lats']
        _raw_lons = _site['raw_lons']
        _field = _site['field']
        _label = _site['label']

        if len(_raw_lats) == 0:
            continue

        # Track for bounds calculation
        _all_lats.extend(_raw_lats)
        _all_lons.extend(_raw_lons)
        _total_raw_points += len(_raw_lats)

        # Create color array (lighter, semi-transparent version of field color)
        _color = field_colors[_field]
        _colors = np.column_stack([
            np.full(len(_raw_lats), _color[0], dtype=np.uint8),
            np.full(len(_raw_lats), _color[1], dtype=np.uint8),
            np.full(len(_raw_lats), _color[2], dtype=np.uint8),
            np.full(len(_raw_lats), 80, dtype=np.uint8)  # Semi-transparent
        ])

        # Create tooltip data for raw points
        _tooltip_data = {
            'type': ['Raw'] * len(_raw_lats),
            'site_label': [_label] * len(_raw_lats),
            'field': [_field] * len(_raw_lats),
        }

        # Create GeoDataFrame for raw points
        _gdf_raw = gpd.GeoDataFrame(
            _tooltip_data,
            geometry=gpd.points_from_xy(_raw_lons, _raw_lats),
            crs="EPSG:4326"
        )

        # Create layer for raw points (smaller, semi-transparent)
        _layer_raw = ScatterplotLayer.from_geopandas(
            _gdf_raw,
            get_fill_color=_colors,
            get_radius=2,
            radius_min_pixels=1,
            radius_max_pixels=3,
            pickable=True,
            auto_highlight=True
        )
        _layers.append(_layer_raw)

    # Second pass: add average location points (small, semi-transparent circle markers)
    for _field in ['A', 'B']:
        _field_sites = [s for s in reward_site_locations if s['field'] == _field]

        if len(_field_sites) == 0:
            continue

        _lats = [s['latitude'] for s in _field_sites]
        _lons = [s['longitude'] for s in _field_sites]
        _labels = [s['label'] for s in _field_sites]
        _gnss_ids = [s['gnss_id'] for s in _field_sites]

        # Create color array (semi-transparent)
        _color = field_colors[_field]
        _colors = np.column_stack([
            np.full(len(_lats), _color[0], dtype=np.uint8),
            np.full(len(_lats), _color[1], dtype=np.uint8),
            np.full(len(_lats), _color[2], dtype=np.uint8),
            np.full(len(_lats), 180, dtype=np.uint8)  # Semi-transparent (180/255)
        ])

        # Create tooltip data
        _tooltip_data = {
            'type': ['Average'] * len(_lats),
            'label': _labels,
            'field': [_field] * len(_labels),
            'gnss_id': _gnss_ids,
            'latitude': [f"{lat:.7f}" for lat in _lats],
            'longitude': [f"{lon:.7f}" for lon in _lons],
        }

        # Create GeoDataFrame
        _gdf = gpd.GeoDataFrame(
            _tooltip_data,
            geometry=gpd.points_from_xy(_lons, _lats),
            crs="EPSG:4326"
        )

        # Create layer with small, semi-transparent circle markers
        _layer = ScatterplotLayer.from_geopandas(
            _gdf,
            get_fill_color=_colors,
            get_radius=3,  # Small radius
            radius_min_pixels=3,
            radius_max_pixels=8,
            filled=True,
            stroked=False,
            pickable=True,
            auto_highlight=True
        )
        _layers.append(_layer)

    # Calculate center and zoom
    _center_lat = (min(_all_lats) + max(_all_lats)) / 2
    _center_lon = (min(_all_lons) + max(_all_lons)) / 2

    # Calculate zoom based on spread
    _lat_span = max(_all_lats) - min(_all_lats)
    _lon_span = max(_all_lons) - min(_all_lons)
    _max_span = max(_lat_span, _lon_span)

    if _max_span > 0.1:
        _zoom = 14
    elif _max_span > 0.01:
        _zoom = 16
    else:
        _zoom = 18

    print(f"Displaying {len(reward_site_locations)} reward sites on map")
    print(f"Raw data points: {_total_raw_points:,}")
    print(f"Map center: ({_center_lat:.6f}, {_center_lon:.6f}), zoom: {_zoom}")

    # Create map
    _basemap = MaplibreBasemap(style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json")
    reward_map = Map(
        layers=_layers,
        basemap=_basemap,
        height=800,
        picking_radius=5,
        show_tooltip=True,
        view_state={
            'longitude': _center_lon,
            'latitude': _center_lat,
            'zoom': _zoom
        }
    )

    reward_map
    return


@app.cell
def _(mo):
    # Create state store for view state (bearing will be set by map cell)
    # mo.state() returns a tuple (getter, setter)
    get_view_state, set_view_state = mo.state({
        "longitude": 0,
        "latitude": 0, 
        "zoom": 18,
        "bearing": None,  # Will be set by map cell
        "pitch": 0.0
    })
    
    show_measurements_checkbox = mo.ui.checkbox(label="Show actual measurements", value=True)
    mo.md(f"""
    ## Grid-Aligned Visualization

    {show_measurements_checkbox}

    This view shows actual reward site locations with estimated ideal grid positions.
    - **Blue/Red dots**: Actual reward site locations (Field A/B) - shown when checked
    - **Gray circles**: Estimated ideal grid positions (rigid fit with fixed 10m spacing) - always shown
    - **Gray square outline**: 50x50m arena boundary (fit to grid) - always shown
    - **Red/Green lines**: X/Y axis references - shown when checked
    - **Orange/Cyan dots**: Raw GNSS data used to derive axes - shown when checked
    - **Small gray dots**: Offset lines between actual and expected - shown when checked
    
    The grid is fit using a rigid body transformation (translation + rotation only) 
    with fixed 10m spacing, maintaining correct label-to-point matching.

    **The map is automatically rotated so the grid Y-axis points UP.**
    Uncheck the box to hide all calibration/measurement data and see only the fitted grid.
    """)
    return (show_measurements_checkbox, get_view_state, set_view_state)


@app.cell(hide_code=True)
def _(
    DATA_DIR,
    Map,
    MaplibreBasemap,
    ScatterplotLayer,
    datetime,
    gpd,
    label_to_grid,
    mo,
    np,
    reward_site_locations,
    show_measurements_checkbox,
    timezone,
    get_view_state,
    set_view_state,
):
    # =============================================================================
    # Grid Reference Data Collection
    # =============================================================================
    # Load calibration movement data to establish grid axes.
    # These timestamps capture when specific GNSS devices were moved along
    # the grid axes (X and Y directions) for calibration purposes.

    # Grid reference definitions: (field, axis, gnss_id, t1, t2)
    # These define line segments representing the X and Y axes of the grid
    _grid_references = [
        # Field A - Y axis movement (GNSS 14 moved North-South)
        ("A", "y", 14, 
         datetime(2026, 1, 26, 9, 49, 4, tzinfo=timezone.utc).timestamp(), 
         datetime(2026, 1, 26, 9, 49, 30, tzinfo=timezone.utc).timestamp()),
        # Field A - X axis movement (GNSS 02 moved East-West)
        ("A", "x", 2,
         datetime(2026, 1, 26, 9, 50, 0, tzinfo=timezone.utc).timestamp(),
         datetime(2026, 1, 26, 9, 59, 40, tzinfo=timezone.utc).timestamp()),
        # Field B - Y axis movement (GNSS 10 moved North-South)
        ("B", "y", 10,
         datetime(2026, 1, 26, 9, 22, 2, tzinfo=timezone.utc).timestamp(),
         datetime(2026, 1, 26, 9, 22, 30, tzinfo=timezone.utc).timestamp()),
        # Field B - X axis movement (GNSS 10 moved East-West, different time)
        # Start time shifted from :16 to :22 to exclude initial points that looked wrong
        ("B", "x", 10,
         datetime(2026, 1, 26, 9, 20, 22, tzinfo=timezone.utc).timestamp(),
         datetime(2026, 1, 26, 9, 20, 30, tzinfo=timezone.utc).timestamp()),
    ]

    def _get_location_at_time(gnss_id, target_time, tolerance_seconds=10):
        """
        Get the GPS location of a device at a specific Unix timestamp.

        Searches through all LOGS*.TXT files for the device to find the
        reading closest to the target time within the specified tolerance.
        """
        _base_path = DATA_DIR / "gnss" / "gnss_data_field"

        _device_dirs = [
            _base_path / f"GNSS_{gnss_id:02d}",
            _base_path / f"GNSS-{gnss_id}",
        ]

        _device_dir = None
        for _d in _device_dirs:
            if _d.exists():
                _device_dir = _d
                break

        if _device_dir is None:
            return None, None

        _closest_lat, _closest_lon, _closest_diff = None, None, float('inf')

        for _log_file in sorted(_device_dir.glob("LOGS*.TXT")):
            if _log_file.stat().st_size == 0:
                continue
            with open(_log_file, errors='ignore') as _f:
                for _line in _f:
                    _parts = _line.strip().split(":")
                    if len(_parts) >= 6:
                        try:
                            _lat = float(_parts[3])
                            _lon = float(_parts[4])
                            _timestamp = float(_parts[5])
                            _diff = abs(_timestamp - target_time)
                            if _diff < _closest_diff and _diff <= tolerance_seconds:
                                _closest_diff = _diff
                                _closest_lat = _lat
                                _closest_lon = _lon
                        except ValueError:
                            pass

        return _closest_lat, _closest_lon

    def _get_axis_data_points(gnss_id, start_time, end_time):
        """
        Get all GPS data points for a device within a time range.
        
        Used to visualize the actual GNSS data that was used to derive
        the X and Y axes during calibration movements.
        
        Returns:
            Tuple of (lats, lons, timestamps) arrays, or (None, None, None) if no data
        """
        _base_path = DATA_DIR / "gnss" / "gnss_data_field"

        _device_dirs = [
            _base_path / f"GNSS_{gnss_id:02d}",
            _base_path / f"GNSS-{gnss_id}",
        ]

        _device_dir = None
        for _d in _device_dirs:
            if _d.exists():
                _device_dir = _d
                break

        if _device_dir is None:
            return None, None, None

        _lats, _lons, _times = [], [], []

        for _log_file in sorted(_device_dir.glob("LOGS*.TXT")):
            if _log_file.stat().st_size == 0:
                continue
            with open(_log_file, errors='ignore') as _f:
                for _line in _f:
                    _parts = _line.strip().split(":")
                    if len(_parts) >= 6:
                        try:
                            _lat = float(_parts[3])
                            _lon = float(_parts[4])
                            _timestamp = float(_parts[5])
                            if start_time <= _timestamp <= end_time:
                                _lats.append(_lat)
                                _lons.append(_lon)
                                _times.append(_timestamp)
                        except ValueError:
                            pass

        if len(_lats) == 0:
            return None, None, None

        return np.array(_lats), np.array(_lons), np.array(_times)

    # =============================================================================
    # Grid Analysis - Calculate Grid Alignment from Reference Movements
    # =============================================================================
    # For each field, calculate:
    # 1. The angle of the Y-axis from North (for potential map rotation)
    # 2. A least-squares fit of measured points to the expected grid
    # 3. Offset distances between measured and expected positions

    print("\n" + "=" * 70)
    print("Grid Reference Analysis")
    print("=" * 70)

    # Load reference line endpoints from calibration movements
    _grid_lines = {}
    _axis_data_points = {}  # Store actual GNSS data used for axis derivation
    
    for _field, _axis, _gnss_id, _t1, _t2 in _grid_references:
        _lat1, _lon1 = _get_location_at_time(_gnss_id, _t1)
        _lat2, _lon2 = _get_location_at_time(_gnss_id, _t2)

        if _lat1 is not None and _lat2 is not None:
            # Calculate vector between reference points
            _dlat = _lat2 - _lat1
            _dlon = _lon2 - _lon1
            # Convert to meters for distance calculation
            _dist_m = np.sqrt((_dlat * 111000)**2 + (_dlon * 111000 * np.cos(np.radians(_lat1)))**2)
            # Calculate angle from North (clockwise, in radians)
            _angle = np.arctan2(_dlon, _dlat)

            if _field not in _grid_lines:
                _grid_lines[_field] = {}
            _grid_lines[_field][_axis] = {
                'lat1': _lat1, 'lon1': _lon1,
                'lat2': _lat2, 'lon2': _lon2,
                'angle': _angle,
                'dist_m': _dist_m
            }
            
            # Collect all GNSS data points used to derive this axis
            _axis_lats, _axis_lons, _axis_times = _get_axis_data_points(_gnss_id, _t1, _t2)
            if _axis_lats is not None:
                if _field not in _axis_data_points:
                    _axis_data_points[_field] = {}
                _axis_data_points[_field][_axis] = {
                    'lats': _axis_lats,
                    'lons': _axis_lons,
                    'times': _axis_times,
                    'gnss_id': _gnss_id
                }

    # Analyze each field's grid alignment
    grid_analysis = []

    for _field_key in ['A', 'B']:
        if _field_key not in _grid_lines:
            continue

        _field_sites = [s for s in reward_site_locations if s['field'] == _field_key]
        if len(_field_sites) == 0:
            continue

        _y_line = _grid_lines[_field_key].get('y')
        _x_line = _grid_lines[_field_key].get('x')

        if _y_line is None:
            continue

        # Calculate grid angle from Y-line
        # The Y-line points at angle _grid_angle from North
        # To make it point UP on screen, we would rotate the map by -_grid_angle
        # (negative because bearing is clockwise, but we want counter-clockwise rotation)
        _grid_angle = _y_line['angle']
        _bearing = -np.degrees(_grid_angle)  # Map bearing to align Y-axis up

        print(f"\n{_field_key} Field:")
        print(f"  Y-line angle: {np.degrees(_grid_angle):.1f}° from North")
        print(f"  Map bearing: {_bearing:.1f}°")

        # =============================================================================
        # Rigid Grid Fit (Fixed 10m spacing)
        # =============================================================================
        # Fit a rigid grid (translation + rotation only) with fixed 10m spacing
        # to the measured points. This preserves the relative positions of all
        # grid points while finding the best orientation and position.
        
        _gxs = np.array([label_to_grid[s['label']][0] for s in _field_sites])
        _gys = np.array([label_to_grid[s['label']][1] for s in _field_sites])
        _lats = np.array([s['latitude'] for s in _field_sites])
        _lons = np.array([s['longitude'] for s in _field_sites])
        
        # Convert grid indices to actual meters (10m spacing)
        _grid_spacing_m = 10.0
        _X = _gxs * _grid_spacing_m  # Grid X positions in meters
        _Y = _gys * _grid_spacing_m  # Grid Y positions in meters
        
        # Convert measured lat/lon to local meters (centered on mean)
        _mean_lat = np.mean(_lats)
        _mean_lon = np.mean(_lons)
        _lat_m = 111000
        _lon_m = 111000 * np.cos(np.radians(_mean_lat))
        
        # Measured points in local meters (centered)
        _meas_x = (_lons - _mean_lon) * _lon_m
        _meas_y = (_lats - _mean_lat) * _lat_m
        
        # Center the grid points
        _mean_X = np.mean(_X)
        _mean_Y = np.mean(_Y)
        _X_centered = _X - _mean_X
        _Y_centered = _Y - _mean_Y
        
        # =============================================================================
        # Kabsch Algorithm for Optimal Rotation
        # =============================================================================
        # Compute the optimal rotation matrix that aligns grid points to measured points
        # H = sum(grid_point_i * measured_point_i^T)
        _H = np.zeros((2, 2))
        for i in range(len(_X_centered)):
            _grid_vec = np.array([_X_centered[i], _Y_centered[i]])
            _meas_vec = np.array([_meas_x[i], _meas_y[i]])
            _H += np.outer(_grid_vec, _meas_vec)
        
        # SVD to find optimal rotation
        _U, _S, _Vt = np.linalg.svd(_H)
        _R = _Vt.T @ _U.T
        
        # Ensure proper rotation (det(R) = 1, not -1 which would be a reflection)
        if np.linalg.det(_R) < 0:
            _Vt[1, :] *= -1
            _R = _Vt.T @ _U.T
        
        # Rotation angle
        _rotation_angle = np.arctan2(_R[1, 0], _R[0, 0])
        
        # Compute optimal translation
        # Transform grid center by rotation, then find translation to measured center
        _rotated_grid_center = _R @ np.array([_mean_X, _mean_Y])
        _translation = np.array([np.mean(_meas_x), np.mean(_meas_y)]) - _rotated_grid_center
        
        print(f"  Grid spacing: {_grid_spacing_m}m (fixed)")
        print(f"  Rotation angle: {np.degrees(_rotation_angle):.2f}°")
        print(f"  Translation: ({_translation[0]:.2f}m, {_translation[1]:.2f}m)")
        
        # Calculate estimated positions and offsets
        _offsets = []
        for i, _s in enumerate(_field_sites):
            _lat, _lon = _s['latitude'], _s['longitude']
            _grid_x, _grid_y = label_to_grid.get(_s['label'], (0, 0))
            
            # Grid point in meters
            _grid_point_m = np.array([_grid_x * _grid_spacing_m, _grid_y * _grid_spacing_m])
            
            # Apply rigid transformation: rotate then translate
            _rotated_point = _R @ _grid_point_m
            _transformed_point = _rotated_point + _translation
            
            # Convert back to lat/lon
            _est_lon = _mean_lon + _transformed_point[0] / _lon_m
            _est_lat = _mean_lat + _transformed_point[1] / _lat_m
            
            # Calculate offset in meters
            _dx = (_lon - _est_lon) * _lon_m
            _dy = (_lat - _est_lat) * _lat_m
            _offset = np.sqrt(_dx**2 + _dy**2)
            
            _offsets.append({
                'label': _s['label'],
                'grid_x': _grid_x, 'grid_y': _grid_y,
                'lat': _lat, 'lon': _lon,
                'expected_lat': _est_lat,
                'expected_lon': _est_lon,
                'offset_m': _offset
            })
        
        _avg_offset = np.mean([o['offset_m'] for o in _offsets])
        _max_offset = np.max([o['offset_m'] for o in _offsets])
        _rms_offset = np.sqrt(np.mean([o['offset_m']**2 for o in _offsets]))
        print(f"  Average offset: {_avg_offset:.2f}m")
        print(f"  Max offset: {_max_offset:.2f}m")
        print(f"  RMS offset: {_rms_offset:.2f}m")

        grid_analysis.append({
            'field': _field_key,
            'bearing': _bearing,
            'y_line': _y_line,
            'x_line': _x_line,
            'offsets': _offsets,
            # Store rigid transformation parameters for arena outline
            'R': _R,
            'translation': _translation,
            'mean_lat': _mean_lat,
            'mean_lon': _mean_lon,
            'lat_m': _lat_m,
            'lon_m': _lon_m,
            'grid_spacing_m': _grid_spacing_m
        })

    # =============================================================================
    # Create Map Layers
    # =============================================================================
    # Build visualization layers:
    # 1. Y-axis reference line (red/orange dots) - shown only when show_actual is checked
    # 2. X-axis reference line (green dots) - shown only when show_actual is checked
    # 3. Raw GNSS data points used for axis derivation (orange/cyan dots) - shown only when show_actual is checked
    # 4. Offset lines (gray dots connecting actual to expected) - shown only when show_actual is checked
    # 5. Estimated grid positions (gray circles) - always shown
    # 6. Actual measured positions (blue/red dots) - shown only when show_actual is checked
    # 7. Arena outline (50x50m square, gray line) - always shown

    _layers = []
    _all_lats = []
    _all_lons = []
    
    # Check checkbox state for showing actual measurements
    _show_actual = show_measurements_checkbox.value

    for _ga in grid_analysis:
        _field_key = _ga['field']
        _field_color = [55, 126, 184] if _field_key == 'A' else [228, 26, 28]
        _y_line = _ga['y_line']
        _x_line = _ga['x_line']

        # Only show calibration/actual data layers when checkbox is checked
        if _show_actual:
            # 1. Y-axis reference line (red/orange)
            if _y_line is not None:
                _y_lats = np.linspace(_y_line['lat1'], _y_line['lat2'], 20)
                _y_lons = np.linspace(_y_line['lon1'], _y_line['lon2'], 20)
                _y_colors = np.column_stack([
                    np.full(len(_y_lats), 255, dtype=np.uint8),  # Red
                    np.full(len(_y_lats), 80, dtype=np.uint8),   # Green
                    np.full(len(_y_lats), 80, dtype=np.uint8),   # Blue
                    np.full(len(_y_lats), 200, dtype=np.uint8)   # Alpha
                ])

                _y_gdf = gpd.GeoDataFrame(
                    [{'type': 'Y-axis'} for _ in range(len(_y_lats))],
                    geometry=gpd.points_from_xy(_y_lons, _y_lats),
                    crs="EPSG:4326"
                )

                _y_layer = ScatterplotLayer.from_geopandas(
                    _y_gdf,
                    get_fill_color=_y_colors,
                    get_radius=4,
                    radius_min_pixels=3,
                    radius_max_pixels=6,
                    filled=True,
                    pickable=True
                )
                _layers.append(_y_layer)
                _all_lats.extend(_y_lats)
                _all_lons.extend(_y_lons)

            # 2. X-axis reference line (green)
            if _x_line is not None:
                _x_lats = np.linspace(_x_line['lat1'], _x_line['lat2'], 20)
                _x_lons = np.linspace(_x_line['lon1'], _x_line['lon2'], 20)
                _x_colors = np.column_stack([
                    np.full(len(_x_lats), 80, dtype=np.uint8),   # Red
                    np.full(len(_x_lats), 200, dtype=np.uint8),  # Green
                    np.full(len(_x_lats), 80, dtype=np.uint8),   # Blue
                    np.full(len(_x_lats), 200, dtype=np.uint8)   # Alpha
                ])

                _x_gdf = gpd.GeoDataFrame(
                    [{'type': 'X-axis'} for _ in range(len(_x_lats))],
                    geometry=gpd.points_from_xy(_x_lons, _x_lats),
                    crs="EPSG:4326"
                )

                _x_layer = ScatterplotLayer.from_geopandas(
                    _x_gdf,
                    get_fill_color=_x_colors,
                    get_radius=4,
                    radius_min_pixels=3,
                    radius_max_pixels=6,
                    filled=True,
                    pickable=True
                )
                _layers.append(_x_layer)
                _all_lats.extend(_x_lats)
                _all_lons.extend(_x_lons)

            # 3. Actual GNSS data points used to derive axes (small circles)
            # These show the raw GNSS data that was used to establish X and Y axis directions
            if _field_key in _axis_data_points:
                for _axis_name in ['y', 'x']:
                    if _axis_name in _axis_data_points[_field_key]:
                        _axis_data = _axis_data_points[_field_key][_axis_name]
                        _raw_axis_lats = _axis_data['lats']
                        _raw_axis_lons = _axis_data['lons']
                        _axis_gnss_id = _axis_data['gnss_id']
                        
                        if len(_raw_axis_lats) > 0:
                            # Color: orange for Y-axis, cyan for X-axis
                            if _axis_name == 'y':
                                _axis_raw_color = [255, 165, 0]  # Orange
                            else:
                                _axis_raw_color = [0, 255, 255]  # Cyan
                            
                            _axis_raw_colors = np.column_stack([
                                np.full(len(_raw_axis_lats), _axis_raw_color[0], dtype=np.uint8),
                                np.full(len(_raw_axis_lats), _axis_raw_color[1], dtype=np.uint8),
                                np.full(len(_raw_axis_lats), _axis_raw_color[2], dtype=np.uint8),
                                np.full(len(_raw_axis_lats), 150, dtype=np.uint8)  # Semi-transparent
                            ])
                            
                            _axis_raw_tooltip = {
                                'type': [f'{_axis_name.upper()}-axis raw GNSS'] * len(_raw_axis_lats),
                                'gnss_id': [_axis_gnss_id] * len(_raw_axis_lats),
                                'point_num': list(range(1, len(_raw_axis_lats) + 1)),
                            }
                            
                            _axis_raw_gdf = gpd.GeoDataFrame(
                                _axis_raw_tooltip,
                                geometry=gpd.points_from_xy(_raw_axis_lons, _raw_axis_lats),
                                crs="EPSG:4326"
                            )
                            
                            _axis_raw_layer = ScatterplotLayer.from_geopandas(
                                _axis_raw_gdf,
                                get_fill_color=_axis_raw_colors,
                                get_radius=3,
                                radius_min_pixels=2,
                                radius_max_pixels=5,
                                filled=True,
                                pickable=True
                            )
                            _layers.append(_axis_raw_layer)
                            _all_lats.extend(_raw_axis_lats)
                            _all_lons.extend(_raw_axis_lons)

            # 4. Offset lines (gray dots)
            _offset_lats = []
            _offset_lons = []
            _offset_colors_list = []

            for _o in _ga['offsets']:
                _offset_lats.extend([_o['lat'], _o['expected_lat']])
                _offset_lons.extend([_o['lon'], _o['expected_lon']])
                _offset_colors_list.extend([[128, 128, 128, 100], [128, 128, 128, 100]])

            if len(_offset_lats) > 0:
                _offset_tooltip = {
                    'type': ['Offset'] * len(_offset_lats),
                    'label': [_o['label'] for _o in _ga['offsets'] for _ in range(2)],
                }

                _offset_gdf = gpd.GeoDataFrame(
                    _offset_tooltip,
                    geometry=gpd.points_from_xy(_offset_lons, _offset_lats),
                    crs="EPSG:4326"
                )

                _offset_colors = np.array(_offset_colors_list, dtype=np.uint8)
                _offset_layer = ScatterplotLayer.from_geopandas(
                    _offset_gdf,
                    get_fill_color=_offset_colors,
                    get_radius=2,
                    radius_min_pixels=1,
                    radius_max_pixels=2,
                    pickable=True
                )
                _layers.append(_offset_layer)
                _all_lats.extend(_offset_lats)
                _all_lons.extend(_offset_lons)

        # 3. Estimated grid positions (gray circles) - always shown
        _exp_lats = [_o['expected_lat'] for _o in _ga['offsets']]
        _exp_lons = [_o['expected_lon'] for _o in _ga['offsets']]
        _exp_labels = [_o['label'] for _o in _ga['offsets']]

        if len(_exp_lats) > 0:
            _exp_colors = np.column_stack([
                np.full(len(_exp_lats), 150, dtype=np.uint8),
                np.full(len(_exp_lats), 150, dtype=np.uint8),
                np.full(len(_exp_lats), 150, dtype=np.uint8),
                np.full(len(_exp_lats), 200, dtype=np.uint8)
            ])

            _exp_gdf = gpd.GeoDataFrame(
                [{'label': _l, 'type': 'Expected', 'grid': f"({label_to_grid[_l][0]},{label_to_grid[_l][1]})"} for _l in _exp_labels],
                geometry=gpd.points_from_xy(_exp_lons, _exp_lats),
                crs="EPSG:4326"
            )

            _exp_layer = ScatterplotLayer.from_geopandas(
                _exp_gdf,
                get_fill_color=_exp_colors,
                get_radius=5,
                radius_min_pixels=4,
                radius_max_pixels=8,
                filled=True,
                stroked=False,
                pickable=True
            )
            _layers.append(_exp_layer)
            _all_lats.extend(_exp_lats)
            _all_lons.extend(_exp_lons)

        # 4. Actual positions (colored dots) - REDUCED SIZE
        # Only shown if toggle button is in "show" state
        if _show_actual:
            _act_lats = [_o['lat'] for _o in _ga['offsets']]
            _act_lons = [_o['lon'] for _o in _ga['offsets']]
            _act_labels = [_o['label'] for _o in _ga['offsets']]
            _act_offsets = [f"{_o['offset_m']:.1f}m" for _o in _ga['offsets']]

            if len(_act_lats) > 0:
                _act_colors = np.column_stack([
                    np.full(len(_act_lats), _field_color[0], dtype=np.uint8),
                    np.full(len(_act_lats), _field_color[1], dtype=np.uint8),
                    np.full(len(_act_lats), _field_color[2], dtype=np.uint8),
                    np.full(len(_act_lats), 220, dtype=np.uint8)
                ])

                _act_gdf = gpd.GeoDataFrame(
                    [{'label': _l, 'type': 'Actual', 'field': _field_key, 'offset': _o} for _l, _o in zip(_act_labels, _act_offsets)],
                    geometry=gpd.points_from_xy(_act_lons, _act_lats),
                    crs="EPSG:4326"
                )

                _act_layer = ScatterplotLayer.from_geopandas(
                    _act_gdf,
                    get_fill_color=_act_colors,
                    get_radius=4,  # REDUCED from 7 to 4
                    radius_min_pixels=4,  # REDUCED from 6 to 4
                    radius_max_pixels=10,  # REDUCED from 14 to 10
                    filled=True,
                    stroked=False,
                    pickable=True,
                    auto_highlight=True
                )
                _layers.append(_act_layer)
                _all_lats.extend(_act_lats)
                _all_lons.extend(_act_lons)

        # 5. Arena outline (50x50m square based on the rigid grid fit)
        # The arena corners in grid units: (0,0), (5,0), (5,5), (0,5)
        # Transform these using the same rigid transformation
        _R = _ga['R']
        _translation = _ga['translation']
        _mean_lat = _ga['mean_lat']
        _mean_lon = _ga['mean_lon']
        _lat_m = _ga['lat_m']
        _lon_m = _ga['lon_m']
        _grid_spacing_m = _ga['grid_spacing_m']
        
        # Arena corners in grid units (0-5 range = 50m with 10m spacing)
        _arena_corners_grid = [
            (0, 0),    # Bottom-left
            (5, 0),    # Bottom-right
            (5, 5),    # Top-right
            (0, 5),    # Top-left
            (0, 0),    # Back to start to close the square
        ]
        
        _arena_lats = []
        _arena_lons = []
        
        for _gx, _gy in _arena_corners_grid:
            # Convert grid units to meters
            _corner_m = np.array([_gx * _grid_spacing_m, _gy * _grid_spacing_m])
            # Apply rigid transformation: rotate then translate
            _rotated = _R @ _corner_m
            _transformed = _rotated + _translation
            # Convert back to lat/lon
            _arena_lon = _mean_lon + _transformed[0] / _lon_m
            _arena_lat = _mean_lat + _transformed[1] / _lat_m
            _arena_lats.append(_arena_lat)
            _arena_lons.append(_arena_lon)
        
        # Create points along the arena outline
        _arena_outline_lats = []
        _arena_outline_lons = []
        _n_points_per_edge = 20  # Number of points per edge for smooth line
        
        for i in range(len(_arena_corners_grid) - 1):
            _lat1, _lon1 = _arena_lats[i], _arena_lons[i]
            _lat2, _lon2 = _arena_lats[i + 1], _arena_lons[i + 1]
            
            for j in range(_n_points_per_edge):
                _t = j / _n_points_per_edge
                _interp_lat = _lat1 + _t * (_lat2 - _lat1)
                _interp_lon = _lon1 + _t * (_lon2 - _lon1)
                _arena_outline_lats.append(_interp_lat)
                _arena_outline_lons.append(_interp_lon)
        
        # Add the last point
        _arena_outline_lats.append(_arena_lats[-1])
        _arena_outline_lons.append(_arena_lons[-1])
        
        if len(_arena_outline_lats) > 0:
            _arena_colors = np.column_stack([
                np.full(len(_arena_outline_lats), 100, dtype=np.uint8),  # Red
                np.full(len(_arena_outline_lats), 100, dtype=np.uint8),  # Green
                np.full(len(_arena_outline_lats), 100, dtype=np.uint8),  # Blue
                np.full(len(_arena_outline_lats), 180, dtype=np.uint8)   # Alpha
            ])
            
            _arena_gdf = gpd.GeoDataFrame(
                [{'type': 'Arena outline (50x50m)'} for _ in range(len(_arena_outline_lats))],
                geometry=gpd.points_from_xy(_arena_outline_lons, _arena_outline_lats),
                crs="EPSG:4326"
            )
            
            _arena_layer = ScatterplotLayer.from_geopandas(
                _arena_gdf,
                get_fill_color=_arena_colors,
                get_radius=2,
                radius_min_pixels=1,
                radius_max_pixels=3,
                filled=True,
                pickable=True
            )
            _layers.append(_arena_layer)
            _all_lats.extend(_arena_outline_lats)
            _all_lons.extend(_arena_outline_lons)

    # =============================================================================
    # Map Creation and View Configuration
    # =============================================================================
    # The map view state (including bearing/rotation) must be passed to the Map
    # constructor for it to take effect in marimo's reactive execution model.
    # Calling set_view_state() after construction does not work.

    # Calculate center and zoom
    if len(_all_lats) > 0:
        _center_lat = (min(_all_lats) + max(_all_lats)) / 2
        _center_lon = (min(_all_lons) + max(_all_lons)) / 2

        _lat_span = max(_all_lats) - min(_all_lats)
        _lon_span = max(_all_lons) - min(_all_lons)
        _max_span = max(_lat_span, _lon_span)

        if _max_span > 0.1:
            _zoom = 14
        elif _max_span > 0.01:
            _zoom = 16
        else:
            _zoom = 18

        # Get bearing from first field
        if len(grid_analysis) > 0:
            _avg_bearing = grid_analysis[0]['bearing']
        else:
            _avg_bearing = 0

        print(f"\nGrid map: {len(_layers)} layers, center=({_center_lat:.6f}, {_center_lon:.6f}), zoom={_zoom}, bearing={_avg_bearing:.1f}°")
    else:
        # Default view if no data
        _center_lat, _center_lon = 43.6, 1.4  # Approximate Toulouse area
        _zoom = 16
        _avg_bearing = 0
        print("\nNo grid data to display")

    # Set bearing to aligned grid bearing immediately
    if len(_all_lats) > 0:
        _view_bearing = float(_avg_bearing)
        print(f"🔄 Grid aligned: bearing={_view_bearing:.1f}°")
    else:
        _view_bearing = 0.0

    # Create map with view_state including bearing
    # Note: Bearing rotation requires a Maplibre-based basemap
    _basemap = MaplibreBasemap(style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json")

    if len(_all_lats) > 0:
        # Create view state as a plain dictionary with bearing for grid alignment
        # Use get_view_state/set_view_state to persist bearing across checkbox toggles
        _current_state = get_view_state()
        _stored_bearing = _current_state.get("bearing")
        
        # If bearing not yet stored, calculate and store it
        if _stored_bearing is None:
            _stored_bearing = float(_view_bearing)
            set_view_state({
                "longitude": float(_center_lon),
                "latitude": float(_center_lat),
                "zoom": float(_zoom),
                "bearing": _stored_bearing,
                "pitch": 0.0
            })
        
        _view_state = {
            "longitude": float(_center_lon),
            "latitude": float(_center_lat),
            "zoom": float(_zoom),
            "bearing": _stored_bearing,
            "pitch": 0.0
        }
        print(f"  Creating map with bearing={_view_state['bearing']:.1f}°")
        _grid_map = Map(
            layers=_layers,
            basemap=_basemap,
            height=800,
            picking_radius=5,
            show_tooltip=True,
            view_state=_view_state
        )
    else:
        _grid_map = Map(
            layers=_layers,
            basemap=_basemap,
            height=800,
            picking_radius=5,
            show_tooltip=True
        )

    _grid_map
    return (grid_analysis,)


@app.cell
def _(mo, pd, grid_analysis):
    # Output a table summarizing the offsets from the fitted grid
    _offset_rows = []
    for _ga in grid_analysis:
        _field = _ga['field']
        for _o in _ga['offsets']:
            _offset_rows.append({
                'Field': _field,
                'Label': _o['label'],
                'Grid X': _o['grid_x'],
                'Grid Y': _o['grid_y'],
                'Measured Lat': f"{_o['lat']:.7f}",
                'Measured Lon': f"{_o['lon']:.7f}",
                'Fitted Lat': f"{_o['expected_lat']:.7f}",
                'Fitted Lon': f"{_o['expected_lon']:.7f}",
                'Offset (m)': f"{_o['offset_m']:.2f}"
            })
    
    _offset_df = pd.DataFrame(_offset_rows)
    
    # Calculate statistics
    _all_offsets = [_o['offset_m'] for _ga in grid_analysis for _o in _ga['offsets']]
    _avg_offset = sum(_all_offsets) / len(_all_offsets)
    _max_offset = max(_all_offsets)
    _rms_offset = (sum(o**2 for o in _all_offsets) / len(_all_offsets))**0.5
    
    mo.md(f"""
    ## Grid Fit Offset Summary
    
    **Statistics:**
    - Average offset: {_avg_offset:.2f}m
    - Max offset: {_max_offset:.2f}m  
    - RMS offset: {_rms_offset:.2f}m
    
    **Detailed offsets by site:**
    
    {mo.as_html(_offset_df)}
    """)
    return


@app.cell
def _(grid_analysis, pd, DATA_DIR, np):
    # Export fitted reward site locations to CSV including arena corners

    # Prepare data for CSV export
    _csv_rows = []
    
    # Define corner labels and their grid positions
    # E1: bottom-left (0,0), E2: bottom-right (5,0), E3: top-right (5,5), E4: top-left (0,5)
    _corner_labels = ['E1', 'E2', 'E3', 'E4']
    _corner_grid = [(0, 0), (5, 0), (5, 5), (0, 5)]
    
    for _ga in grid_analysis:
        _field = _ga['field']
        
        # Add regular reward sites
        for _o in _ga['offsets']:
            _csv_rows.append({
                'field': _field,
                'label': _o['label'],
                'grid_x': _o['grid_x'],
                'grid_y': _o['grid_y'],
                'latitude': _o['expected_lat'],
                'longitude': _o['expected_lon']
            })
        
        # Add arena corners using the same rigid transformation
        _R = _ga['R']
        _translation = _ga['translation']
        _mean_lat = _ga['mean_lat']
        _mean_lon = _ga['mean_lon']
        _lat_m = _ga['lat_m']
        _lon_m = _ga['lon_m']
        _grid_spacing_m = _ga['grid_spacing_m']
        
        for _label, (_gx, _gy) in zip(_corner_labels, _corner_grid):
            # Convert grid units to meters
            _corner_m = np.array([_gx * _grid_spacing_m, _gy * _grid_spacing_m])
            # Apply rigid transformation: rotate then translate
            _rotated = _R @ _corner_m
            _transformed = _rotated + _translation
            # Convert back to lat/lon
            _corner_lon = _mean_lon + _transformed[0] / _lon_m
            _corner_lat = _mean_lat + _transformed[1] / _lat_m
            
            _csv_rows.append({
                'field': _field,
                'label': _label,
                'grid_x': _gx,
                'grid_y': _gy,
                'latitude': _corner_lat,
                'longitude': _corner_lon
            })

    # Create DataFrame and export to CSV
    # Using 9 decimal places for ~1.1mm precision (7 decimals = ~1.1cm)
    _fitted_df = pd.DataFrame(_csv_rows)
    _csv_path = DATA_DIR / "fitted_reward_sites.csv"
    _fitted_df.to_csv(_csv_path, index=False, float_format='%.9f')

    print(f"✓ Exported {len(_csv_rows)} locations ({len(_csv_rows) - 8} reward sites + 8 arena corners) to: {_csv_path}")
    print(f"  Precision: 9 decimal places (~1.1mm)")
    print(f"\nCSV contents:")
    print(_fitted_df.to_string(index=False))
    return


if __name__ == "__main__":
    app.run()
