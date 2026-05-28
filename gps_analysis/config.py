"""Constants and configuration for the gps_analysis package."""
from pathlib import Path

_R = 6_371_000.0  # Earth radius (m)

DATA_DIR = Path(__file__).parent.parent / "data"

CONFIG_TRANSFORMS = {
    "A":        (0,   "none"),
    "B":        (90,  "none"),
    "C":        (0,   "mirror y"),
    "D":        (270, "mirror x"),
    "CTRL_FAR": (0,   "mirror y"),
}

# Reward site grid positions (grid_x, grid_y) for each label.
# Origin is bottom-left; y increases north, x increases east.
# Units: 10 m per grid unit, arena spans ~0–5.
SITE_GRID = {
    'B1': (2, 4), 'A3': (3, 4),
    'C1': (1, 3), 'D2': (2, 3), 'C2': (3, 3), 'D3': (4, 3),
    'A1': (1, 2), 'B2': (2, 2), 'A2': (3, 2), 'B3': (4, 2),
    'D1': (2, 1), 'C3': (3, 1),
}
SITE_LABELS = sorted(SITE_GRID.keys())

# Per (field, config) GPS offset correction (metres), derived from sheep
# cluster centres at known reward sites (experienced trials, assay >= 2).
# Second-iteration values: original one-shot offsets + measured residuals.
# Reduces mean offset from ~0.7m (uncorrected) to ~0.1-0.2m per site.
GPS_OFFSET = {
    ("A", "A"): (0.81, -0.25),
    ("A", "B"): (0.00, 1.09),
    ("A", "C"): (1.03, 0.39),
    ("A", "D"): (-0.15, 1.29),
    ("A", "CTRL_FAR"): (0.44, 0.05),
    ("B", "A"): (1.58, -0.93),
    ("B", "B"): (0.72, 0.35),
    ("B", "C"): (1.69, 1.01),
    ("B", "D"): (-1.10, 1.02),
    ("B", "CTRL_BARN"): (0.54, -0.59),
}
