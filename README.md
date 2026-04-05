# GPS Data Analysis

Interactive [marimo](https://marimo.io/) notebooks for visualising and analysing sheep GPS trial data.

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

Install dependencies into a local virtual environment:

```bash
uv sync
```

## Running the notebooks

All scripts are [marimo](https://marimo.io/) reactive notebooks.  Run them from the repo root.

### Interactive mode (recommended)

Opens the notebook in your browser with live UI controls (dropdowns, sliders, etc.):

```bash
uv run marimo run scripts/reward_site_proximity.py
```

Replace the filename with whichever script you want.  Use this mode for exploration.

### Edit mode

Opens the notebook editor so you can modify cells:

```bash
uv run marimo edit scripts/reward_site_proximity.py
```

### Quick reference

```bash
# Data viewing
uv run marimo run scripts/trial_data_viewer.py
uv run marimo run scripts/gnss_data_viewer.py
uv run marimo run scripts/occupancy_heatmap.py

# Analysis
uv run marimo run scripts/reward_site_proximity.py
uv run marimo run scripts/path_length_analysis.py
uv run marimo run scripts/flocking_dynamics.py
uv run marimo run scripts/leader_follower.py
uv run marimo run scripts/orientation_check.py
uv run marimo run scripts/spatial_information.py
uv run marimo run scripts/site_discovery_effects.py
```

> **Note:** `scripts/analysis_utils.py` is a plain Python utility module imported by the analysis scripts — it is not a notebook and should not be run directly.

## Scripts

### Data preparation & viewing

| Script | Description |
|--------|-------------|
| `trial_data_cleaning_and_viewing.py` | Parses the raw Excel trial sheets, cleans them, and exports a combined CSV |
| `trial_data_viewer.py` | Filterable table viewer for the combined trial CSV |
| `reward_site_identification.py` | Fits reward-site and arena-corner coordinates from field-calibration GNSS data |
| `gnss_data_viewer.py` | Map-based GPS track viewer with reward-site overlay and per-trial selection |
| `occupancy_heatmap.py` | 2-D occupancy heatmap with per-configuration orientation transforms and aggregation |

### Analysis

All analysis scripts share a common utility module (`analysis_utils.py`) that handles GNSS data loading, arena coordinate projection, and automatic field correction.

| Script | Description |
|--------|-------------|
| `reward_site_proximity.py` | Tracks how many sheep are within a configurable radius of each reward site over time, revealing discovery events as spikes in the per-site time series |
| `path_length_analysis.py` | Cumulative path length per sheep; detects reward-site visit events; defines trial completion as when the N-th unique site is first found; aggregates path-to-completion and completion time by assay |
| `flocking_dynamics.py` | Pairwise inter-animal distances, nearest-neighbour distance, and per-sheep spread from the group centroid over time; aggregate cohesion plots by assay |
| `leader_follower.py` | Identifies frontal-position leaders (farthest ahead in direction of travel) and pioneer visitors (first to reach each site); reports normalised leadership entropy as a consistency metric |
| `orientation_check.py` | Diagnostic: side-by-side occupancy heatmaps with and without per-configuration orientation transforms for configs A/B/C/D; reward-site overlay confirms correct alignment |
| `spatial_information.py` | Sliding-window spatial entropy (how spread-out the group is), cumulative unique-cell coverage, and revisit rate over time; aggregate by assay |
| `site_discovery_effects.py` | Smooth probability-of-site-presence time series per site; compares group speed and spread before vs after each first-discovery event |

## Project structure

```
.
├── data/                                  # Not tracked by git (too large)
│   ├── experimental/
│   │   ├── Sheep_Experimental_Data.xlsx   # Source trial metadata
│   │   └── Sheep_Trial_Data.csv           # Cleaned combined trials
│   ├── gnss/
│   │   ├── 2-02-26/ … 26-02-26/          # Per-day GNSS device logs
│   │   ├── gnss_data_field/               # Field-calibration GNSS data
│   │   └── field_merged/                  # Merged field data
│   └── fitted_reward_sites.csv            # Fitted reward-site coordinates
├── scripts/
│   ├── analysis_utils.py                  # Shared utilities (not a notebook)
│   └── *.py                               # Marimo notebooks
├── reward_sites.md                        # Reward site numbering reference
└── README.md
```
