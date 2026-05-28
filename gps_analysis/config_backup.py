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

# Per (field, config) GPS offset correction (metres), measured ONE-SHOT from
# uncorrected sheep cluster centres at known reward sites (Phase 2, assay >= 2).
# Applying this once reduces mean offset from ~0.7m to ~0.3m per site.
# Further iteration is unstable (detection radius feedback loop).
GPS_OFFSET = {
    ("A", "A"): (0.63, -0.14),
    ("A", "B"): (-0.03, 0.68),
    ("A", "C"): (0.69, 0.25),
    ("A", "D"): (-0.14, 0.88),
    ("B", "A"): (1.07, -0.67),
    ("B", "B"): (0.55, 0.14),
    ("B", "C"): (1.11, 0.58),
    ("B", "D"): (-0.73, 0.55),
}
