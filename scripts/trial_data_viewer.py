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
    csv_path = Path(__file__).parent.parent / "data" / "experimental" / "Sheep_Trial_Data.csv"
    df = pd.read_csv(csv_path, dtype={"Sheep ID": str})
    df
    return csv_path, df


@app.cell
def _(df, mo):
    dates = sorted(df["date"].unique())
    date_selector = mo.ui.dropdown(
        options=["All"] + dates,
        value="All",
        label="Filter by date",
    )
    configurations = sorted(df["configuration"].dropna().unique())
    config_selector = mo.ui.dropdown(
        options=["All"] + configurations,
        value="All",
        label="Filter by configuration",
    )
    mo.hstack([date_selector, config_selector])
    return config_selector, configurations, date_selector, dates


@app.cell
def _(config_selector, date_selector, df, mo):
    filtered = df.copy()
    if date_selector.value != "All":
        filtered = filtered[filtered["date"] == date_selector.value]
    if config_selector.value != "All":
        filtered = filtered[filtered["configuration"] == config_selector.value]
    mo.md(f"**Showing {len(filtered)} of {len(df)} rows**")
    return (filtered,)


@app.cell
def _(filtered, mo):
    mo.ui.table(filtered, label="Trial Data")
    return


if __name__ == "__main__":
    app.run()
