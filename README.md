# GPS Data Analysis

Interactive [marimo](https://marimo.io/) notebooks for visualising and analysing sheep GPS trial data.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/trial_data_cleaning_and_viewing.py` | Parses the raw Excel trial sheets, cleans them, and exports a combined CSV |
| `scripts/trial_data_viewer.py` | Filterable table viewer for the combined trial CSV |
| `scripts/reward_site_identification.py` | Fits reward-site and arena-corner coordinates from field-calibration GNSS data |
| `scripts/gnss_data_viewer.py` | Map-based GPS track viewer with reward-site overlay and per-trial selection |

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
