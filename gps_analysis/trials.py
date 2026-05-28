"""Trial metadata parsing."""
import pandas as pd

from gps_analysis.config import DATA_DIR


def build_trials(df=None):
    """Build list of trial dicts from the experimental CSV.

    Each dict contains:
      name, date, field, config, start_time (str HH:MM:SS),
      duration_min, devices (list[int]), device_to_sheep (dict int→str),
      assay, notes, group_num, group_size.
    """
    if df is None:
        df = pd.read_csv(
            DATA_DIR / "experimental" / "Sheep_Trial_Data.csv",
            dtype={"Sheep ID": str},
        )
    trials = []
    with_start = df[df["start_time"].notna()].copy()
    for (date, start_time, group_num, group_size), group in with_start.groupby(
        ["date", "start_time", "Group #", "Group Size"], dropna=False
    ):
        devices = sorted(
            set(
                [int(v) for v in group["GNSS_SN1"].dropna()]
                + [int(v) for v in group["GNSS_SN2"].dropna()]
            )
        )
        d2s = {}
        for _, row in group.iterrows():
            sid = row["Sheep ID"] if pd.notna(row["Sheep ID"]) else "Unknown"
            if pd.notna(row["GNSS_SN1"]):
                d2s[int(row["GNSS_SN1"])] = sid
            if pd.notna(row["GNSS_SN2"]):
                d2s[int(row["GNSS_SN2"])] = sid

        field = str(group["field"].iloc[0]) if pd.notna(group["field"].iloc[0]) else "Unknown"
        config = (
            str(group["configuration"].iloc[0])
            if pd.notna(group["configuration"].iloc[0])
            else "Unknown"
        )
        gnum = int(group_num) if pd.notna(group_num) else 0
        gsize = int(group_size) if pd.notna(group_size) else 0
        av = group["assay"].iloc[0]
        assay = None
        if pd.notna(av):
            try:
                assay = int(float(av))
            except (ValueError, TypeError):
                assay = str(av)
        notes_list = group["note"].dropna().unique()
        notes = f"Group {gnum}, Size {gsize}"
        if len(notes_list) > 0:
            notes += f" - {'; '.join(notes_list)}"

        trials.append(
            {
                "name": f"{date} - Grp{gnum:02d} {str(start_time)[:5]} Field {field}, {config}, {gsize} sheep",
                "date": str(date),
                "field": field,
                "config": config,
                "start_time": str(start_time),
                "duration_min": 35,
                "devices": devices,
                "device_to_sheep": d2s,
                "assay": assay,
                "notes": notes,
                "group_num": gnum,
                "group_size": gsize,
            }
        )
    return trials
