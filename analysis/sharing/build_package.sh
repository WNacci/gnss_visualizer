#!/usr/bin/env bash
# Build the sheep_gnss_2026.tar.zst package for external sharing.
#
# Usage: bash analysis/sharing/build_package.sh
# Output: /tmp/sheep_gnss_2026.tar.zst
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SHARING="$REPO/analysis/sharing"
STAGE_PARENT=$(mktemp -d)
STAGE="$STAGE_PARENT/sheep_gnss_2026"
mkdir -p "$STAGE"

echo "==> Staging in $STAGE"

# 1) Phase 2 only (date >= 2026-02-17) trial CSV.
echo "==> Filtering trial CSV to Phase 2"
python3 - "$REPO" "$STAGE" <<'PY'
import sys
import pandas as pd

repo, stage = sys.argv[1], sys.argv[2]
df = pd.read_csv(
    f"{repo}/data/experimental/Sheep_Trial_Data.csv",
    dtype={"Sheep ID": str},
)
df["_d"] = pd.to_datetime(df["date"], errors="coerce")
n_in = len(df)
df = df[df["_d"] >= "2026-02-17"].drop(columns=["_d"])
df.to_csv(f"{stage}/trial_metadata.csv", index=False)
print(f"    {len(df)} / {n_in} rows kept (Phase 2)")
PY

# 2) Reward site calibration.
echo "==> Copying fitted_reward_sites.csv"
cp "$REPO/data/fitted_reward_sites.csv" "$STAGE/"

# 3) Phase 2 raw GNSS days.
echo "==> Copying Phase 2 GNSS days"
mkdir -p "$STAGE/gnss"
n_copied=0
for day in 17 18 19 20 21 22 23 24 25 26; do
    src="$REPO/data/gnss/${day}-02-26"
    if [ -d "$src" ]; then
        cp -r "$src" "$STAGE/gnss/"
        n_copied=$((n_copied + 1))
    fi
done
echo "    $n_copied days copied"

# 4) Loader, example, README.
echo "==> Copying loader, example, README"
cp "$SHARING/sheep_gnss_loader.py" "$STAGE/"
cp "$SHARING/example_usage.py"     "$STAGE/"
cp "$SHARING/README.md"            "$STAGE/"

# 5) Compress.
OUTPUT="${OUTPUT:-/tmp/sheep_gnss_2026.tar.zst}"
echo "==> Compressing to $OUTPUT"
if command -v zstd >/dev/null 2>&1; then
    tar --use-compress-program="zstd -19 -T0" \
        -cf "$OUTPUT" \
        -C "$STAGE_PARENT" sheep_gnss_2026
else
    echo "    zstd not found; falling back to gzip"
    OUTPUT="${OUTPUT%.tar.zst}.tar.gz"
    tar -czf "$OUTPUT" -C "$STAGE_PARENT" sheep_gnss_2026
fi

ls -lh "$OUTPUT"
echo "==> Done: $OUTPUT"
echo "==> Stage left at $STAGE for inspection (remove when ready)"
