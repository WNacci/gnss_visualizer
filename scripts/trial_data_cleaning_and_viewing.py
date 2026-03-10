import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    from pathlib import Path
    return Path, mo, pd


@app.cell
def _(Path, pd):
    xlsx_path = Path(__file__).parent.parent / "data" / "experimental" / "Sheep_Experimental_Data.xlsx"
    xlsx = pd.ExcelFile(xlsx_path)
    sheet_names = xlsx.sheet_names
    sheet_names
    return sheet_names, xlsx, xlsx_path


@app.cell
def _(mo, pd, sheet_names, xlsx):
    # Parse all sheets, adding a 'date' column extracted from the sheet name
    all_sheets = {}
    combined_rows = []
    for name in sheet_names:
        df = pd.read_excel(xlsx, sheet_name=name)
        # Drop unnamed columns (artifacts from extra Excel columns)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        # Extract date from sheet name (e.g. "Experimental_Notes_02_02_26" -> "2026-02-02")
        parts = name.replace("Experimental_Notes_", "").split("_")
        date_str = f"20{parts[2]}-{parts[1]}-{parts[0]}"
        df["Sheep ID"] = df["Sheep ID"].astype(int).astype(str).str.zfill(5)
        # Merge second_start_time into note column, then drop it
        if "second_start_time" in df.columns:
            mask = df["second_start_time"].notna()
            prefix = "second_start_time=" + df.loc[mask, "second_start_time"].astype(str)
            existing = df.loc[mask, "note"].fillna("")
            df.loc[mask, "note"] = (existing.str.strip() + "; " + prefix).str.lstrip("; ")
            df = df.drop(columns=["second_start_time"])
        df.insert(0, "date", date_str)
        all_sheets[name] = df
        combined_rows.append(df)

    combined_df = pd.concat(combined_rows, ignore_index=True)
    mo.md(f"**Total rows across all sheets:** {len(combined_df)}")
    return all_sheets, combined_df, combined_rows


@app.cell
def _(Path, combined_df, mo):
    output_path = Path(__file__).parent.parent / "data" / "experimental" / "Sheep_Trial_Data.csv"
    combined_df.to_csv(output_path, index=False)
    mo.md(f"**Exported to:** `{output_path}`")
    return (output_path,)


@app.cell
def _(combined_df, mo):
    mo.ui.table(combined_df, label="All Experimental Data (Combined)")
    return


@app.cell
def _(all_sheets, mo):
    sheet_selector = mo.ui.dropdown(
        options=list(all_sheets.keys()),
        value=list(all_sheets.keys())[0],
        label="Select sheet",
    )
    sheet_selector
    return (sheet_selector,)


@app.cell
def _(all_sheets, mo, sheet_selector):
    selected_df = all_sheets[sheet_selector.value]
    mo.ui.table(selected_df, label=f"Sheet: {sheet_selector.value}")
    return (selected_df,)


if __name__ == "__main__":
    app.run()
