# GPS Data Analysis

Interactive [marimo](https://marimo.io/) notebooks for visualising and analysing sheep GPS trial data.

## Scripts

### Data preparation & viewing

| Script | Description |
|--------|-------------|
| `scripts/trial_data_cleaning_and_viewing.py` | Parses the raw Excel trial sheets, cleans them, and exports a combined CSV |
| `scripts/trial_data_viewer.py` | Filterable table viewer for the combined trial CSV |
| `scripts/reward_site_identification.py` | Fits reward-site and arena-corner coordinates from field-calibration GNSS data |
| `scripts/gnss_data_viewer.py` | Map-based GPS track viewer with reward-site overlay and per-trial selection |
| `scripts/occupancy_heatmap.py` | 2-D occupancy heatmap with per-configuration orientation transforms and aggregation |

### Analysis

All analysis scripts share a common utility module (`scripts/analysis_utils.py`) that handles data loading, arena coordinate projection, and field auto-detection.

| Script | Description |
|--------|-------------|
| `scripts/reward_site_proximity.py` | Tracks how many sheep are within a configurable radius of each reward site over time, revealing discovery events as spikes in the per-site time series |
| `scripts/path_length_analysis.py` | Computes cumulative path length per sheep; detects reward-site visit events; defines trial completion as when the N-th unique site is first found; aggregates path-to-completion and completion time by assay |
| `scripts/flocking_dynamics.py` | Pairwise inter-animal distances, nearest-neighbour distance, and per-sheep spread from the group centroid over time; aggregate cohesion plots by assay |
| `scripts/leader_follower.py` | Identifies frontal-position leaders (sheep farthest ahead in direction of travel) and pioneer visitors (first to reach each site); reports normalised leadership entropy as a consistency metric |
| `scripts/orientation_check.py` | Diagnostic: side-by-side occupancy heatmaps with and without per-configuration orientation transforms for each config (A/B/C/D); reward-site overlay confirms correct alignment |
| `scripts/spatial_information.py` | Sliding-window spatial entropy (how spread-out the group is), cumulative unique-cell coverage, and revisit rate over time; aggregate by assay |
| `scripts/site_discovery_effects.py` | Smooth probability-of-site-presence time series per site; compares group speed and spread in windows before vs after each first-discovery event |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install uv

**Linux / macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installing, restart your terminal so `uv` is on your PATH.

## Setup

From the `analysis/` directory, install dependencies:

```bash
uv sync
```

This creates a virtual environment and installs everything automatically from `pyproject.toml`.

## Running a notebook

```bash
uv run marimo run scripts/<script_name>.py
```

For example:

```bash
uv run marimo run scripts/gnss_data_viewer.py
```

This opens a read-only view of the notebook in your browser.

### Editing notebooks

```bash
uv run marimo edit scripts/<script_name>.py
```

This opens the notebook editor in your browser.

## Project structure

```
analysis/
├── data/                                  # Not tracked by git (too large)
│   ├── experimental/
│   │   ├── Sheep_Experimental_Data.xlsx   # Source trial metadata
│   │   └── Sheep_Trial_Data.csv           # Cleaned combined trials
│   ├── gnss/
│   │   ├── 2-02-26/ … 26-02-26/          # Per-day GNSS device logs
│   │   ├── gnss_data_field/               # Field-calibration GNSS data
│   │   └── field_merged/                  # Merged field data
│   └── fitted_reward_sites.csv            # Fitted reward-site coordinates
├── scripts/                               # Marimo notebooks
├── reward_sites.md                        # Reward site numbering reference
└── README.md
```
