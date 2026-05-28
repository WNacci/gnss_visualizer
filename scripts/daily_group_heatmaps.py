"""Daily Group Heatmaps / Trajectories

One figure per test day (Feb 17 – Feb 26), showing all 14 groups as subplots.
Layout: 2 rows × 7 columns.  Per-configuration orientation transforms are
applied so baited reward sites align to the same canonical positions across
all configurations.

Controls:
  - Display mode: Heatmap or Trajectory (per-sheep GNSS streams)
  - Bins, duration limit, log scale, colourmap (heatmap mode)
  - Gradient power and contrast (trajectory mode — light-early → dark-late)
"""
import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import io
    import base64
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    import marimo as mo
    from gps_analysis import (
        build_trials, build_tracks_cache, load_trial_tracks,
        build_gps_cache, build_arena_transforms, load_trial_device_tracks,
        DATA_DIR,
    )

    TRIALS = build_trials()
    print(f"Loaded {len(TRIALS)} trials — building tracks cache (runs once)…")
    TRACKS_CACHE = build_tracks_cache()

    # GPS cache and arena transforms are needed to build per-device tracks.
    # build_gps_cache loads instantly from disk after the first run.
    GPS_CACHE = build_gps_cache(TRIALS)
    ARENA_TRANSFORMS = build_arena_transforms()

    # Per-device tracks cache (trajectory mode): each GPS device is kept as a
    # separate entry — sheep that carried two devices appear as two independent
    # tracks sharing the same sheep_id (and therefore the same colour).
    _TEST_DATES_SET = {
        "2026-02-17", "2026-02-18", "2026-02-19",
        "2026-02-21", "2026-02-22", "2026-02-23",
        "2026-02-24", "2026-02-25", "2026-02-26",
    }
    # Key by (date, group_num) — a naturally unique identifier — rather than
    # trial["name"], which was previously non-unique and caused duplicate data.
    DEVICE_TRACKS_CACHE = {}
    for _t in TRIALS:
        if _t["date"] in _TEST_DATES_SET:
            DEVICE_TRACKS_CACHE[(_t["date"], _t["group_num"])] = load_trial_device_tracks(
                _t, gnss_cache=GPS_CACHE, apply_orient=True,
                arena_transforms=ARENA_TRANSFORMS,
            )
    print(f"Device tracks cache: {len(DEVICE_TRACKS_CACHE)} trials")

    # Reference reward sites: config-A baited positions used for the overlay
    # after orientation transforms normalise all configs to the same layout.
    _rdf = pd.read_csv(DATA_DIR / "fitted_reward_sites.csv")
    _ref = _rdf[
        (_rdf["field"] == "A") & (_rdf["label"].str.match(r"^A\d+$"))
    ].copy()
    _ref["_sort"] = _ref["grid_x"] + _ref["grid_y"]
    REF_SITES = _ref.sort_values("_sort").reset_index(drop=True)

    # Qualitative colour palette (tab10) for per-sheep trajectory colouring.
    # Each sheep in a group gets a distinct base colour; time gradient blends
    # the colour from white (early) → full colour (late).
    _tab10 = mcm.get_cmap("tab10")
    SHEEP_PALETTE = [np.array(_tab10(i)[:3]) for i in range(10)]

    return (
        io, base64, np, pd, plt, mo,
        TRIALS, TRACKS_CACHE, load_trial_tracks,
        DEVICE_TRACKS_CACHE, REF_SITES, SHEEP_PALETTE,
    )


@app.cell(hide_code=True)
def _(mo):
    """UI controls: display mode and visualisation parameters."""

    display_mode = mo.ui.radio(
        options=["Heatmap", "Trajectory"],
        value="Heatmap",
        label="Display mode",
    )
    bins_slider = mo.ui.slider(
        start=20, stop=500, step=5, value=150,
        label="Bins (heatmap)",
    )
    duration_slider = mo.ui.slider(
        start=1, stop=35, step=1, value=35,
        label="Max duration (min)",
    )
    log_scale_checkbox = mo.ui.checkbox(label="Log scale (heatmap)", value=False)
    cmap_dropdown = mo.ui.dropdown(
        options=["Blues", "hot_r", "viridis", "plasma", "YlOrRd"],
        value="Blues",
        label="Colormap (heatmap)",
    )
    gradient_power_slider = mo.ui.slider(
        start=0.1, stop=5.0, step=0.1, value=1.0,
        label="Gradient power — transition steepness (trajectory)",
    )
    gradient_contrast_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.85,
        label="Gradient contrast — washed-out↔full-colour range (trajectory)",
    )
    gradient_dir_radio = mo.ui.radio(
        options=[
            "Light start → full colour end  (early = washed out, late = saturated)",
            "Full colour start → light end  (early = saturated, late = washed out)",
        ],
        value="Light start → full colour end  (early = washed out, late = saturated)",
        label="Gradient direction (trajectory)",
    )
    _all_dates = [
        "2026-02-17", "2026-02-18", "2026-02-19",
        "2026-02-21", "2026-02-22", "2026-02-23",
        "2026-02-24", "2026-02-25", "2026-02-26",
    ]
    date_dropdown = mo.ui.dropdown(
        options={"All dates": None, **{d: d for d in _all_dates}},
        value="All dates",
        label="Date (trajectory: single date recommended)",
    )

    mo.md(f"""
# Daily Group Heatmaps

**Display mode:** {display_mode}

---
### Parameters
{mo.hstack([bins_slider, duration_slider])}
{mo.hstack([log_scale_checkbox, cmap_dropdown])}
{mo.hstack([gradient_power_slider, gradient_contrast_slider])}
{gradient_dir_radio}
{date_dropdown}
    """)
    return (
        display_mode, bins_slider, duration_slider, log_scale_checkbox,
        cmap_dropdown, gradient_power_slider, gradient_contrast_slider,
        gradient_dir_radio, date_dropdown,
    )


@app.cell(hide_code=True)
def _(
    TRIALS, TRACKS_CACHE, load_trial_tracks,
    DEVICE_TRACKS_CACHE, REF_SITES, SHEEP_PALETTE,
    io, base64, np, plt, mo,
    display_mode, bins_slider, duration_slider, log_scale_checkbox,
    cmap_dropdown, gradient_power_slider, gradient_contrast_slider,
    gradient_dir_radio, date_dropdown,
):
    """Build one figure per selected test day, 14 groups per figure (2×7 grid)."""

    _ALL_DATES = [
        "2026-02-17", "2026-02-18", "2026-02-19",
        "2026-02-21", "2026-02-22", "2026-02-23",
        "2026-02-24", "2026-02-25", "2026-02-26",
    ]
    _selected = date_dropdown.value
    TEST_DATES = [_selected] if _selected is not None else _ALL_DATES
    NCOLS = 7
    NROWS = 2

    _MODE     = display_mode.value
    _BINS     = bins_slider.value
    _DUR      = duration_slider.value
    _LOG      = log_scale_checkbox.value
    _CMAP     = cmap_dropdown.value
    _GPOWER    = gradient_power_slider.value
    _GCONTRAST = gradient_contrast_slider.value
    _GINVERT   = gradient_dir_radio.value.startswith("Full colour")

    # Fast lookup: (date, group_num) → trial dict
    _trial_lookup = {}
    for _t in TRIALS:
        if _t["date"] in TEST_DATES:
            _trial_lookup[(_t["date"], _t["group_num"])] = _t

    # ------------------------------------------------------------------
    # Helper: 2-D histogram for heatmap mode
    # ------------------------------------------------------------------
    def _compute_heatmap(tracks, dur_limit, bins):
        gx_parts, gy_parts = [], []
        for trk in tracks.values():
            mask = trk["t"] <= dur_limit
            gx_parts.append(trk["gx"][mask])
            gy_parts.append(trk["gy"][mask])
        if not gx_parts:
            return None, None, None
        gx = np.concatenate(gx_parts)
        gy = np.concatenate(gy_parts)
        H, xe, ye = np.histogram2d(gx, gy, bins=bins, range=[[0, 5], [0, 5]])
        return H.T.astype(float), xe, ye  # .T → (bins_y, bins_x) for imshow origin='lower'

    # ------------------------------------------------------------------
    # Helper: render heatmap into an axes
    # ------------------------------------------------------------------
    def _render_heatmap(ax, H, xe, ye, vmax, cmap, log_scale):
        display_H  = np.log1p(H)    if log_scale else H
        disp_vmax  = np.log1p(vmax) if log_scale else vmax
        if display_H.max() > 0:
            return ax.imshow(
                display_H,
                extent=[xe[0], xe[-1], ye[0], ye[-1]],
                origin="lower",
                cmap=cmap,
                aspect="equal",
                interpolation="nearest",
                vmin=0,
                vmax=max(disp_vmax, 1e-9),
            )
        return None

    # ------------------------------------------------------------------
    # Helper: render per-sheep trajectories into an axes
    #
    # Each sheep gets a distinct base colour from SHEEP_PALETTE.  Points
    # are blended from white (early in trial) → full base colour (late),
    # producing a light-to-dark time gradient.
    #
    # gradient_power   : exponent applied to normalised time before
    #                    blending — values >1 keep early points pale
    #                    longer, values <1 make the colour build quickly.
    # gradient_contrast: range of the blend.  0 = no gradient (uniform
    #                    full colour); 1 = full white→colour sweep.
    # ------------------------------------------------------------------
    def _render_trajectory(ax, device_tracks, dur_limit, gpower, gcontrast, invert, palette):
        """Plot per-device GNSS traces with a time gradient.

        Gradient direction:
          invert=False  →  washed-out/white at start, full colour at end
          invert=True   →  full colour at start, washed-out/white at end

        Each unique sheep gets a distinct colour from ``palette``.  Devices
        belonging to the same sheep share that colour — they are plotted as
        independent traces, not averaged.

        Returns sheep_to_idx dict for building the subplot legend.
        """
        unique_sheep = sorted(set(trk["sheep_id"] for trk in device_tracks.values()))
        sheep_to_idx = {sid: i for i, sid in enumerate(unique_sheep)}

        for dev_num in sorted(device_tracks.keys()):
            trk  = device_tracks[dev_num]
            mask = trk["t"] <= dur_limit
            gx   = trk["gx"][mask]
            gy   = trk["gy"][mask]
            t    = trk["t"][mask]
            if len(gx) == 0:
                continue

            base_rgb = palette[sheep_to_idx[trk["sheep_id"]] % len(palette)]

            # Normalise time → [0, 1]
            t_norm = np.clip(t / max(float(dur_limit), 1e-9), 0.0, 1.0)

            # Apply power curve then optionally invert direction
            t_grad = t_norm ** gpower          # 0 at start → 1 at end
            if invert:
                t_grad = 1.0 - t_grad          # 1 at start → 0 at end

            # Blend multiplier: low → washed out (white), high → full colour
            mult = (1.0 - gcontrast) + gcontrast * t_grad

            # Blend base colour with white per point
            rgba = np.ones((len(gx), 4))
            for ch in range(3):
                rgba[:, ch] = base_rgb[ch] * mult + (1.0 - mult)

            # Plot chronologically: without invert, full-colour (later) points
            # sit on top; with invert, full-colour (earlier) points sit on top.
            order = np.argsort(t) if invert else np.argsort(-t)
            ax.scatter(
                gx[order], gy[order],
                c=rgba[order],
                s=2,
                linewidths=0,
                rasterized=True,
            )

        return sheep_to_idx

    def _fig_to_image(fig, dpi):
        """Render figure to PNG bytes and a base64 data-URL embedded in mo.Html.

        Using a data-URL (rather than mo.image(bytes)) avoids a known h11/
        starlette bug where marimo makes a secondary HTTP request for the image
        bytes and h11 raises "Too little data for declared Content-Length" when
        the response body is unexpectedly empty.

        Returns (mo.Html, bytes) so the caller can collect raw PNGs for download.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        png_bytes = buf.getvalue()
        data = base64.b64encode(png_bytes).decode("ascii")
        html = mo.Html(
            f'<img src="data:image/png;base64,{data}" '
            f'style="width:100%;display:block;margin-bottom:8px"/>'
        )
        return html, png_bytes

    # ------------------------------------------------------------------
    # Main loop: one figure per date
    # ------------------------------------------------------------------
    # Trajectory scatter points don't compress well as PNG, so use lower DPI.
    # Heatmap imshow compresses well and can use higher DPI for sharpness.
    _DPI = 72 if _MODE == "Trajectory" else 150
    _png_files = {}   # filename → bytes, collected for ZIP download

    for _date in TEST_DATES:
        _day_label = _date[5:]   # "MM-DD"

        _group_data = {}
        _all_H_display = []   # for shared vmax in heatmap mode
        _day_assay = None

        for _g in range(1, 15):
            _trial = _trial_lookup.get((_date, _g))
            if _trial is None:
                _group_data[_g] = None
                continue
            if _day_assay is None:
                _day_assay = _trial["assay"]

            if _MODE == "Heatmap":
                _tracks = load_trial_tracks(
                    _trial, tracks_cache=TRACKS_CACHE, apply_orient=True,
                )
                _H, _xe, _ye = _compute_heatmap(_tracks, _DUR, _BINS)
                _group_data[_g] = (_trial, None, _H, _xe, _ye)
                if _H is not None:
                    _all_H_display.append(np.log1p(_H) if _LOG else _H)
            else:
                _dev_tracks = DEVICE_TRACKS_CACHE.get((_trial["date"], _trial["group_num"]), {})
                _group_data[_g] = (_trial, _dev_tracks, None, None, None)

        # Shared colour scale (heatmap only)
        _vmax = max((_h.max() for _h in _all_H_display), default=1.0)
        _vmax = _vmax if _vmax > 0 else 1.0

        _assay_str = f"Assay {_day_assay}" if _day_assay is not None else ""

        # --- Draw figure ---
        _subplot_size = 5.5 if _MODE == "Trajectory" else 5.5
        _fig, _axes = plt.subplots(
            NROWS, NCOLS,
            figsize=(_subplot_size * NCOLS, _subplot_size * NROWS),
            squeeze=False,
            constrained_layout=True,
            dpi=_DPI,
        )
        _fig.patch.set_alpha(0)
        _last_im = None

        for _g in range(1, 15):
            _row = (_g - 1) // NCOLS
            _col = (_g - 1) % NCOLS
            _ax  = _axes[_row][_col]

            _data = _group_data.get(_g)
            if _data is None:
                _ax.text(
                    0.5, 0.5, f"Group {_g}\nNo data",
                    ha="center", va="center", transform=_ax.transAxes, fontsize=9,
                )
                _ax.set_facecolor("#e8e8e8")
                _ax.set_xticks([])
                _ax.set_yticks([])
                continue

            _trial, _dev_tracks, _H, _xe, _ye = _data
            _config = _trial["config"]
            _assay  = _trial["assay"]

            if _MODE == "Heatmap":
                if _H is not None:
                    _im = _render_heatmap(_ax, _H, _xe, _ye, _vmax, _CMAP, _LOG)
                    if _im is not None:
                        _last_im = _im
                    else:
                        _ax.text(
                            0.5, 0.5, "0 pts in arena",
                            ha="center", va="center",
                            transform=_ax.transAxes, fontsize=8,
                        )
                else:
                    _ax.text(
                        0.5, 0.5, "0 pts in arena",
                        ha="center", va="center",
                        transform=_ax.transAxes, fontsize=8,
                    )
            else:  # Trajectory
                if _dev_tracks:
                    _sheep_to_idx = _render_trajectory(
                        _ax, _dev_tracks, _DUR, _GPOWER, _GCONTRAST, _GINVERT, SHEEP_PALETTE,
                    )
                    # Small per-subplot legend: sheep ID → trace colour
                    _legend_handles = [
                        plt.Line2D(
                            [0], [0], marker="o", color="w",
                            markerfacecolor=SHEEP_PALETTE[_i % len(SHEEP_PALETTE)],
                            markersize=5, label=_sid,
                        )
                        for _sid, _i in sorted(_sheep_to_idx.items(), key=lambda x: x[1])
                    ]
                    _ax.legend(
                        handles=_legend_handles,
                        fontsize=6,
                        loc="lower left",
                        framealpha=0.85,
                        edgecolor="none",
                        handletextpad=0.3,
                        borderpad=0.4,
                        labelspacing=0.15,
                    )
                else:
                    _ax.text(
                        0.5, 0.5, "No tracks",
                        ha="center", va="center",
                        transform=_ax.transAxes, fontsize=8,
                    )

            # Reward-site overlay — skip assay 0 and control configurations.
            _is_ctrl  = _config not in ("A", "B", "C", "D")
            _show_sites = (str(_assay) != "0") and not _is_ctrl
            if _show_sites:
                for _, _r in REF_SITES.iterrows():
                    _ax.scatter(
                        _r["grid_x"], _r["grid_y"],
                        s=60, zorder=5, marker="o",
                        facecolors="none", edgecolors="#FFE066", linewidths=1.5,
                    )

            # Faint arena grid lines
            for _v in range(1, 5):
                _ax.axvline(_v, color="#cccccc", alpha=0.7, linewidth=0.4)
                _ax.axhline(_v, color="#cccccc", alpha=0.7, linewidth=0.4)

            _ax.set_xlim(-0.05, 5.05)
            _ax.set_ylim(-0.05, 5.05)
            _ax.set_facecolor("white")
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.set_title(f"Grp {_g}  |  {_config}", fontsize=8.5, pad=3)

        # Shared colourbar for heatmap mode
        if _MODE == "Heatmap" and _last_im is not None:
            _cbar_label = "log(1+count)" if _LOG else "count"
            _fig.colorbar(
                _last_im,
                ax=_axes.ravel().tolist(),
                label=_cbar_label,
                shrink=0.55,
                pad=0.01,
            )

        _fig.suptitle(
            f"{_MODE} — {_day_label}   ({_assay_str})",
            fontsize=14, fontweight="bold", y=1.01,
        )

        _html, _png = _fig_to_image(_fig, dpi=_DPI)
        mo.output.append(_html)
        _png_files[f"{_day_label}_{_MODE.lower()}.png"] = _png
        print(f"Rendered {_day_label}  ({_assay_str})  [{_MODE}]")

    # Build ZIP of all rendered figures and offer a download button
    import zipfile
    _zip_buf = io.BytesIO()
    with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
        for _fname, _data in _png_files.items():
            _zf.writestr(_fname, _data)
    mo.output.append(
        mo.download(
            _zip_buf.getvalue(),
            filename=f"daily_group_{_MODE.lower()}s.zip",
            mimetype="application/zip",
        )
    )
    return


if __name__ == "__main__":
    app.run()
