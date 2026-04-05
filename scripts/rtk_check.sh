#!/usr/bin/env bash
# rtk_check.sh — Report RTK value distribution per day (per unit) and overall.
# Usage: ./rtk_check.sh [-c output.csv] [gnss_data_dir]
#   -c output.csv   Optionally write results to a CSV file.
#   gnss_data_dir   Defaults to ../data/gnss relative to this script.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
CSV_FILE=""

while getopts "c:" opt; do
    case $opt in
        c) CSV_FILE="$OPTARG" ;;
        *) echo "Usage: $0 [-c output.csv] [gnss_data_dir]" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

DATA_DIR="${1:-$SCRIPT_DIR/../data/gnss}"

# Helper: print a table separator
sep() { printf '%s\n' "------------------------------------------------------------------------"; }

# Helper: format a count + percentage
fmt() {
    local n=$1 total=$2
    if (( total > 0 )); then
        printf "%d (%s%%)" "$n" "$(awk "BEGIN {printf \"%.1f\", $n/$total*100}")"
    else
        printf "0 (0.0%%)"
    fi
}

# Helper: print a table header
header() {
    printf "  %-14s %10s %14s %14s %14s %14s\n" "Unit" "Total" "RTK=0" "RTK=40" "RTK=80" "Other"
    sep
}

# CSV header
if [[ -n "$CSV_FILE" ]]; then
    printf "day,unit,total,rtk_0,rtk_0_pct,rtk_40,rtk_40_pct,rtk_80,rtk_80_pct,other,other_pct\n" > "$CSV_FILE"
fi

csv_row() {
    local day=$1 unit=$2 total=$3 n0=$4 n40=$5 n80=$6 nother=$7
    if [[ -z "$CSV_FILE" ]]; then return; fi
    local pct0=0 pct40=0 pct80=0 pctother=0
    if (( total > 0 )); then
        pct0=$(awk "BEGIN {printf \"%.2f\", $n0/$total*100}")
        pct40=$(awk "BEGIN {printf \"%.2f\", $n40/$total*100}")
        pct80=$(awk "BEGIN {printf \"%.2f\", $n80/$total*100}")
        pctother=$(awk "BEGIN {printf \"%.2f\", $nother/$total*100}")
    fi
    printf "%s,%s,%d,%d,%s,%d,%s,%d,%s,%d,%s\n" \
        "$day" "$unit" "$total" "$n0" "$pct0" "$n40" "$pct40" "$n80" "$pct80" "$nother" "$pctother" >> "$CSV_FILE"
}

grand_total=0
grand_0=0
grand_40=0
grand_80=0
grand_other=0

for day_dir in "$DATA_DIR"/[0-9]*-02-26; do
    day=$(basename "$day_dir")

    day_total=0
    day_0=0
    day_40=0
    day_80=0
    day_other=0

    printf "\n=== %s ===\n" "$day"
    header

    for unit_dir in "$day_dir"/[Gg][Nn][Ss][Ss]-*; do
        unit=$(basename "$unit_dir")
        read -r total n0 n40 n80 nother <<< "$(
            awk -F: '
            {
                rtk = $3; total++
                if (rtk == 0) n0++
                else if (rtk == 40) n40++
                else if (rtk == 80) n80++
                else nother++
            }
            END { printf "%d %d %d %d %d", total+0, n0+0, n40+0, n80+0, nother+0 }
            ' "$unit_dir"/LOGS*.TXT 2>/dev/null || echo "0 0 0 0 0"
        )"

        printf "  %-14s %10d %14s %14s %14s %14s\n" \
            "$unit" "$total" "$(fmt "$n0" "$total")" "$(fmt "$n40" "$total")" \
            "$(fmt "$n80" "$total")" "$(fmt "$nother" "$total")"

        csv_row "$day" "$unit" "$total" "$n0" "$n40" "$n80" "$nother"

        day_total=$((day_total + total))
        day_0=$((day_0 + n0))
        day_40=$((day_40 + n40))
        day_80=$((day_80 + n80))
        day_other=$((day_other + nother))
    done

    sep
    printf "  %-14s %10d %14s %14s %14s %14s\n" \
        "Day Total" "$day_total" "$(fmt "$day_0" "$day_total")" "$(fmt "$day_40" "$day_total")" \
        "$(fmt "$day_80" "$day_total")" "$(fmt "$day_other" "$day_total")"

    csv_row "$day" "TOTAL" "$day_total" "$day_0" "$day_40" "$day_80" "$day_other"

    grand_total=$((grand_total + day_total))
    grand_0=$((grand_0 + day_0))
    grand_40=$((grand_40 + day_40))
    grand_80=$((grand_80 + day_80))
    grand_other=$((grand_other + day_other))
done

printf "\n\n=== OVERALL SUMMARY ===\n"
header
printf "  %-14s %10d %14s %14s %14s %14s\n" \
    "ALL DAYS" "$grand_total" "$(fmt "$grand_0" "$grand_total")" "$(fmt "$grand_40" "$grand_total")" \
    "$(fmt "$grand_80" "$grand_total")" "$(fmt "$grand_other" "$grand_total")"

if [[ -n "$CSV_FILE" ]]; then
    printf "\nCSV written to: %s\n" "$CSV_FILE"
fi
