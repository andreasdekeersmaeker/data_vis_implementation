import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from scipy.spatial import ConvexHull
    import warnings
    warnings.filterwarnings("ignore")
    return ConvexHull, KMeans, SimpleImputer, StandardScaler, alt, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌿 Oumalik Vegetation Plot Explorer

    An interactive ordination clock — plots positioned by species composition, coloured by environmental similarity.

    **How to read this:**
    - **Angle** on the circle = dominant species category in that plot
    - **Distance from centre** = how favourable / stable the environmental conditions are (further = more favourable)
    - **Colour** = cluster of similar soil + environmental conditions
    - **Blob** = convex hull wrapping each cluster
    """)
    return


@app.cell
def _(sp_raw):
    # Stap 1: Maak een masker voor de waarde in de eerste kolom
    masker = sp_raw.iloc[:, 2] == 'Campylium hispidulum'

    # Stap 2: Pas het masker toe en negeer de eerste twee rijen
    # (Merk op dat .loc de indexlabels gebruikt, .iloc de posities)
    resultaat = sp_raw.iloc[2:][masker[2:]]
    print(resultaat)
    return


@app.cell
def _(np, pd):
    # ── Load & clean data ──────────────────────────────────────────────────────

    soil_raw = pd.read_csv("oumalik_soil_data.csv")
    env_raw  = pd.read_csv("oumalik_environmental_data.csv")

    sp_raw = pd.read_csv(
        "oumalik_species_data.csv", encoding="cp1252"
    )

    # ── Parse species matrix (rows=species, cols=plots) ───────────────────────
    author_nums = sp_raw.iloc[0, 3:].values.astype(float).astype(int)  # plot IDs
    species_names = sp_raw.iloc[2:, 2].values.astype(str)
    cover_matrix  = sp_raw.iloc[2:, 3:].values.astype(float)            # 286 × 87

    # Replace NaN with 0
    cover_matrix = np.nan_to_num(cover_matrix, nan=0.0)
    return author_nums, cover_matrix, env_raw, soil_raw, sp_raw, species_names


@app.cell
def _(author_nums, cover_matrix, pd, species_names):
    # DataFrame: rows = plots, cols = species
    sp_df = pd.DataFrame(
        cover_matrix.T,
        index=author_nums,
        columns=species_names,
    )
    return (sp_df,)


@app.cell
def _(np, species_names):
    # ── Define 8 major categories placed around the clock ─────────────────────
    CATEGORIES = {
        "Salix (Willows)":      lambda n: n.startswith("Salix"),
        "Carex (Sedges)":       lambda n: n.startswith("Carex"),
        "Eriophorum":           lambda n: n.startswith("Eriophorum"),
        "Dryas":                lambda n: n.startswith("Dryas"),
        "Cladonia (Lichens)":   lambda n: n.startswith("Cladonia") or n.startswith("Alectoria") or n.startswith("Peltigera") or n.startswith("Ochrolechia"),
        "Mosses":               lambda n: any(n.startswith(g) for g in ["Sphagnum","Bryum","Pohlia","Aulacomnium","Hypnum","Polytrichum","Campylium","Dicranum","Encalypta"]),
        "Betula & Shrubs":      lambda n: n.startswith("Betula") or n.startswith("Vaccinium") or n.startswith("Ledum") or n.startswith("Cassiope") or n.startswith("Arctous") or n.startswith("Rubus"),
        "Grasses & Forbs":      lambda n: any(n.startswith(g) for g in ["Poa","Arctagrostis","Puccinellia","Hierochloe","Equisetum","Saxifraga","Pedicularis","Bistorta","Anemone","Astragalus","Draba","Stellaria","Luzula","Juncus","Micranthes","Artemisia","Arnica"]),
    }

    cat_names  = list(CATEGORIES.keys())
    cat_angles = {
        name: 2 * np.pi * i / len(cat_names)
        for i, name in enumerate(cat_names)
    }

    # Assign each species to a category (first match wins; else "Other")
    def assign_category(sname):
        for cat, fn in CATEGORIES.items():
            if fn(sname):
                return cat
        return None  # exclude from angle calculation

    species_cat = {s: assign_category(s) for s in species_names}
    return cat_angles, cat_names, species_cat


@app.cell
def _(cat_names, pd, sp_df, species_cat, species_names):
    # Sum cover per plot per category
    cat_cover = pd.DataFrame(index=sp_df.index, columns=cat_names, dtype=float)
    for cat in cat_names:
        cols = [s for s in species_names if species_cat.get(s) == cat]
        cat_cover[cat] = sp_df[cols].sum(axis=1) if cols else 0.0
    return (cat_cover,)


@app.cell
def _(cat_angles, cat_cover, cat_names, env_raw, np, pd, soil_raw, sp_df):
    # Compute angle for each plot using circular mean weighted by category cover
    def circular_mean(covers, angles):
        """Weighted circular mean of angles."""
        total = covers.sum()
        if total == 0:
            return 0.0
        sin_sum = sum(c * np.sin(angles[i]) for i, c in enumerate(covers))
        cos_sum = sum(c * np.cos(angles[i]) for i, c in enumerate(covers))
        return np.arctan2(sin_sum, cos_sum) % (2 * np.pi)

    angle_values = np.array(list(cat_angles.values()))

    plot_angles = {}
    for pid in sp_df.index:
        covers = cat_cover.loc[pid, cat_names].values.astype(float)
        plot_angles[pid] = circular_mean(covers, angle_values)

    # ── Merge soil + env ───────────────────────────────────────────────────────
    merged = pd.merge(
        soil_raw, env_raw, on="plot_number", suffixes=("_soil", "_env")
    )
    merged = merged.set_index("plot_number")
    return merged, plot_angles


@app.cell
def _(mo):
    mo.md("""
    ---
    ## ⚙️ Controls
    """)
    return


@app.cell
def _(mo):
    # ── Number of clusters ─────────────────────────────────────────────────────
    n_clusters_slider = mo.ui.slider(
        start=2, stop=8, step=1, value=4,
        label="Number of colour clusters (k)"
    )
    n_clusters_slider
    return (n_clusters_slider,)


@app.cell
def _(mo):
    # ── Environmental variables for radius (favourability) ────────────────────
    ENV_VAR_OPTIONS = {
        "Soil moisture":           "soil_moisture ",
        "Organic matter":          "organic_matter",
        "pH":                      "pH",
        "Available water":         "available_water",
        "Carbonates":              "carbonates ",
        "Field capacity":          "field_capacity",
        "Cation exchange cap.":    "cation_ex_capacity",
        "Microrelief height":      "microrelief_ht ",
        "Cover bare soil+rock":    "_bare",      # computed
        "Cover water+litter":      "_wet",       # computed
        "Canopy+shrub+herb+moss":  "_veg_ht",    # computed
        "Aspect + slope":          "_aspect_slope",  # computed
        "Summer air temp":         "summer_air_temp ",
        "Duration snow":           "duration_snow",
        "Wind regime":             "wind_regime ",
        "Disturbance intensity":   "disturbance_intensity",
    }

    env_multiselect = mo.ui.multiselect(
        options=list(ENV_VAR_OPTIONS.keys()),
        value=[
            "Soil moisture", "Organic matter", "pH",
            "Summer air temp", "Cover bare soil+rock", "Cover water+litter"
        ],
        label="Environmental variables for radius (favourability score)"
    )
    env_multiselect
    return ENV_VAR_OPTIONS, env_multiselect


@app.cell
def _(mo):
    # ── Plot selection for comparison ─────────────────────────────────────────
    mo.md("""
    ### 🔍 Select plots to compare (hold Ctrl/Cmd for multiple)
    """)
    return


@app.cell
def _(merged, mo):
    plot_ids_sorted = sorted(merged.index.tolist())
    plot_multiselect = mo.ui.multiselect(
        options=[str(p) for p in plot_ids_sorted],
        value=[],
        label="Plots to compare"
    )
    plot_multiselect
    return (plot_multiselect,)


@app.cell
def _(mo):
    # ── Community filter for comparison bars ──────────────────────────────────
    mo.md("""
    ### 🧪 Species composition comparison: select variable subset
    """)
    return


@app.cell
def _(cat_names, mo):
    cat_filter = mo.ui.multiselect(
        options=cat_names,
        value=cat_names,
        label="Show these species categories in the rolled-out bar chart"
    )
    cat_filter
    return (cat_filter,)


@app.cell
def _(
    ENV_VAR_OPTIONS,
    KMeans,
    SimpleImputer,
    StandardScaler,
    env_multiselect,
    merged,
    n_clusters_slider,
    np,
    pd,
    plot_angles,
):
    # ── Build feature matrix for clustering + radius ───────────────────────────

    df = merged.copy()

    # Computed composite columns
    df["_bare"]         = df["cover_bare_soil "].fillna(0) + df["cover_bare_rock"].fillna(0)
    df["_wet"]          = df["cover_water "].fillna(0)     + df["cover_litter_layer"].fillna(0)
    df["_veg_ht"]       = (
        df["canopy_ht"].fillna(0) +
        df["shrub_layer_ht"].fillna(0) +
        df["herb_layer_ht"].fillna(0)
    )
    # moss_layer_ht is string in env, handle gracefully
    if "moss_layer_ht" in df.columns:
        df["_veg_ht"] += pd.to_numeric(df["moss_layer_ht"], errors="coerce").fillna(0)
    df["_aspect_slope"] = df["aspect"].fillna(0) * np.cos(np.radians(df["slope"].fillna(0)))

    # Selected env variable column names
    selected_labels = env_multiselect.value if env_multiselect.value else list(ENV_VAR_OPTIONS.keys())
    selected_cols   = [ENV_VAR_OPTIONS[label] for label in selected_labels]

    # All soil + env columns for clustering (exclude identifiers)
    CLUSTER_COLS = [
        "sand", "silt", "clay", "organic_matter", "pH", "carbonates ",
        "soil_moisture ", "field_capacity", "wilting_point", "available_water",
        "cation_ex_capacity", "microrelief_ht ", "cover_graminoids",
        "cover_forbs", "cover_mosses_liverworts", "cover_lichen_layer",
        "thaw_depth ", "site_moisture ", "summer_air_temp ",
        "duration_snow", "wind_regime ", "disturbance_intensity",
        "_bare", "_wet", "_veg_ht", "_aspect_slope",
    ]
    cluster_cols_present = [c for c in CLUSTER_COLS if c in df.columns]

    X_cluster = df[cluster_cols_present].values.astype(float)

    imputer = SimpleImputer(strategy="median")
    X_cluster_imp = imputer.fit_transform(X_cluster)

    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster_imp)

    k = n_clusters_slider.value
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_cluster_scaled)

    # ── Radius: favourability from selected variables ──────────────────────────
    radius_cols_present = [c for c in selected_cols if c in df.columns]
    if not radius_cols_present:
        radius_cols_present = ["soil_moisture "]

    X_rad = df[radius_cols_present].values.astype(float)
    X_rad_imp = SimpleImputer(strategy="median").fit_transform(X_rad)
    X_rad_scaled = StandardScaler().fit_transform(X_rad_imp)

    # Favourability = negative mean of "stress" indicators + positive "richness"
    # Simply: PCA-free composite = mean of z-scores (higher = more favourable)
    favourability = X_rad_scaled.mean(axis=1)
    # Normalise to [0.15, 0.95] for radius
    f_min, f_max = favourability.min(), favourability.max()
    if f_max > f_min:
        radius_norm = 0.15 + 0.80 * (favourability - f_min) / (f_max - f_min)
    else:
        radius_norm = np.full(len(favourability), 0.55)

    # ── Assemble plot-level DataFrame ─────────────────────────────────────────
    CLUSTER_COLORS = [
        "#e6534a", "#4a90d9", "#6cc47b", "#f5a623",
        "#9b59b6", "#1abc9c", "#e67e22", "#2c3e50"
    ]

    plot_df = pd.DataFrame({
        "plot":     df.index.tolist(),
        "angle":    [plot_angles.get(p, 0.0) for p in df.index],
        "radius":   radius_norm,
        "cluster":  cluster_labels,
        "colour":   [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in cluster_labels],
        "community": df["plant_community_name_soil"].values
            if "plant_community_name_soil" in df.columns
            else df.get("plant_community_name", "").values,
    })

    # Cartesian coordinates (SVG: y flipped)
    CIRC_R = 300   # px radius of main circle
    CX, CY = 370, 370  # centre

    plot_df["x"] = CX + plot_df["radius"] * CIRC_R * np.cos(plot_df["angle"] - np.pi / 2)
    plot_df["y"] = CY - plot_df["radius"] * CIRC_R * np.sin(plot_df["angle"] - np.pi / 2)
    return (
        CIRC_R,
        CLUSTER_COLORS,
        CX,
        CY,
        cluster_labels,
        df,
        favourability,
        k,
        plot_df,
    )


@app.cell
def _(
    CIRC_R,
    CLUSTER_COLORS,
    CX,
    CY,
    ConvexHull,
    cat_angles,
    cat_names,
    k,
    np,
    plot_df,
    plot_multiselect,
):
    # ── Build SVG for the clock ordination ────────────────────────────────────

    selected_plots = [int(p) for p in plot_multiselect.value] if plot_multiselect.value else []

    def make_clock_svg(pdf, selected):
        lines = []

        W, H = 740, 740

        lines.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
                     f'style="width:100%;max-width:{W}px;font-family:sans-serif">')

        # Background
        lines.append(f'<rect width="{W}" height="{H}" fill="#f8f9f0" rx="12"/>')

        # Title
        lines.append(f'<text x="{CX}" y="28" text-anchor="middle" '
                     f'font-size="15" font-weight="bold" fill="#2c3e2d">'
                     f'Oumalik Plot Ordination Clock</text>')

        # Radial guide circles
        for frac, label in [(0.25, "stressed"), (0.5, ""), (0.75, ""), (1.0, "favourable")]:
            r = frac * CIRC_R
            lines.append(f'<circle cx="{CX}" cy="{CY}" r="{r}" '
                         f'fill="none" stroke="#ccc" stroke-width="0.8" stroke-dasharray="4,4"/>')
            if label:
                lx = CX + r * 0.70
                ly = CY - r * 0.70
                lines.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9" '
                             f'fill="#aaa" text-anchor="middle">{label}</text>')

        # Outer circle
        lines.append(f'<circle cx="{CX}" cy="{CY}" r="{CIRC_R}" '
                     f'fill="none" stroke="#888" stroke-width="1.5"/>')

        # Category spokes + labels
        for cat in cat_names:
            ang = cat_angles[cat] - np.pi / 2
            x_end = CX + (CIRC_R + 8) * np.cos(ang)
            y_end = CY - (CIRC_R + 8) * np.sin(ang)  # SVG y-flip already in sign
            # wait — SVG y down, so:
            x_end2 = CX + (CIRC_R + 55) * np.cos(ang)
            y_end2 = CY + (CIRC_R + 55) * np.sin(ang)

            sx = CX + 20 * np.cos(ang)
            sy = CY + 20 * np.sin(ang)
            ex = CX + CIRC_R * np.cos(ang)
            ey = CY + CIRC_R * np.sin(ang)

            lines.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                         f'stroke="#bbb" stroke-width="1" stroke-dasharray="3,3"/>')

            lx = CX + (CIRC_R + 40) * np.cos(ang)
            ly = CY + (CIRC_R + 40) * np.sin(ang)
            anchor = "middle"
            if np.cos(ang) > 0.3:
                anchor = "start"
            elif np.cos(ang) < -0.3:
                anchor = "end"

            short = cat.split("(")[0].strip()
            lines.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
                         f'font-size="10" font-weight="600" fill="#3a5a3b">{short}</text>')

        # ── Convex hull blobs per cluster ─────────────────────────────────────
        for c in range(k):
            cdf = pdf[pdf["cluster"] == c]
            if len(cdf) >= 3:
                pts = cdf[["x", "y"]].values
                try:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    path_d = "M " + " L ".join(
                        f"{p[0]:.1f},{p[1]:.1f}" for p in hull_pts
                    ) + " Z"
                    col = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
                    lines.append(f'<path d="{path_d}" fill="{col}" '
                                 f'fill-opacity="0.10" stroke="{col}" '
                                 f'stroke-width="1.5" stroke-dasharray="5,3" '
                                 f'stroke-opacity="0.5"/>')
                except Exception:
                    pass

        # ── Plot dots ─────────────────────────────────────────────────────────
        for _, row in pdf.iterrows():
            pid  = int(row["plot"])
            x, y = row["x"], row["y"]
            col  = row["colour"]
            is_sel = pid in selected

            # Shadow / highlight ring
            if is_sel:
                lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" '
                             f'fill="none" stroke="#222" stroke-width="2.5"/>')

            # Dot
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" '
                         f'fill="{col}" stroke="white" stroke-width="1.2" '
                         f'opacity="0.88">'
                         f'<title>Plot {pid}\n{row["community"]}\nCluster {row["cluster"]}</title>'
                         f'</circle>')

            # Label for selected plots
            if is_sel:
                lines.append(f'<text x="{x:.1f}" y="{y-12:.1f}" text-anchor="middle" '
                             f'font-size="9" font-weight="bold" fill="#111">{pid}</text>')

        # ── Centre cross ──────────────────────────────────────────────────────
        lines.append(f'<line x1="{CX-6}" y1="{CY}" x2="{CX+6}" y2="{CY}" '
                     f'stroke="#555" stroke-width="1"/>')
        lines.append(f'<line x1="{CX}" y1="{CY-6}" x2="{CX}" y2="{CY+6}" '
                     f'stroke="#555" stroke-width="1"/>')

        # ── Legend ────────────────────────────────────────────────────────────
        lx0, ly0 = 12, 550
        lines.append(f'<text x="{lx0}" y="{ly0-12}" font-size="10" font-weight="bold" fill="#333">Clusters</text>')
        for c in range(k):
            cy_l = ly0 + c * 18
            col = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            lines.append(f'<circle cx="{lx0+6}" cy="{cy_l+1}" r="5" fill="{col}"/>')
            lines.append(f'<text x="{lx0+16}" y="{cy_l+5}" font-size="9" fill="#333">Cluster {c+1}</text>')

        lines.append("</svg>")
        return "\n".join(lines)


    clock_svg = make_clock_svg(plot_df, selected_plots)
    return clock_svg, selected_plots


@app.cell
def _(clock_svg, mo):
    mo.md(f"""
    ## 🕐 Ordination Clock

    {mo.Html(clock_svg)}

    > **Tip:** Select plots in the control above to highlight and compare them.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 📊 Comparison Panel
    """)
    return


@app.cell
def _(
    alt,
    cluster_labels,
    df,
    favourability,
    mo,
    np,
    pd,
    plot_df,
    selected_plots,
):
    # ── Comparison: relative differences between selected plots ───────────────

    COMPARE_VARS = {
        "Soil moisture":        "soil_moisture ",
        "Organic matter":       "organic_matter",
        "pH":                   "pH",
        "Sand (%)":             "sand",
        "Clay (%)":             "clay",
        "Carbonates":           "carbonates ",
        "Available water":      "available_water",
        "Cation exch. cap.":    "cation_ex_capacity",
        "Summer air temp":      "summer_air_temp ",
        "Duration snow":        "duration_snow",
        "Wind regime":          "wind_regime ",
        "Disturbance intens.":  "disturbance_intensity",
        "Cover bare soil+rock": "_bare",
        "Cover water+litter":   "_wet",
        "Veg height composite": "_veg_ht",
        "Favourability score":  "_fav",
    }

    if len(selected_plots) >= 2:
        df2 = df.copy()
        df2["_fav"] = favourability

        rows = []
        for var_label, col in COMPARE_VARS.items():
            if col not in df2.columns:
                continue
            vals = {}
            for pid1 in selected_plots:
                if pid1 in df2.index:
                    vals[pid1] = df2.loc[pid1, col]
            if not vals:
                continue
            baseline = np.nanmean(list(vals.values()))
            for pid1, v in vals.items():
                rel = ((v - baseline) / (abs(baseline) + 1e-9)) * 100
                rows.append({
                    "Variable": var_label,
                    "Plot": str(pid1),
                    "Value": round(v, 3),
                    "Relative (%)": round(rel, 1),
                })

        comp_df = pd.DataFrame(rows)

        chart = (
            alt.Chart(comp_df)
            .mark_bar()
            .encode(
                x=alt.X("Relative (%):Q", title="Relative difference from mean (%)"),
                y=alt.Y("Variable:N", sort=None),
                color=alt.Color("Plot:N", scale=alt.Scale(scheme="tableau10")),
                tooltip=["Variable", "Plot", "Value", "Relative (%)"],
                row=alt.Row("Plot:N", header=alt.Header(labelFontSize=12)),
            )
            .properties(width=520, height=180)
            .resolve_scale(y="shared")
        )

        summary_rows = []
        for pid2 in selected_plots:
            if pid2 in df2.index:
                cl = cluster_labels[list(df.index).index(pid2)] if pid2 in df.index else "?"
                fav = round(df2.loc[pid2, "_fav"], 3) if pid2 in df2.index else "?"
                comm = plot_df.loc[plot_df["plot"] == pid2, "community"].values
                comm_str = comm[0] if len(comm) else "?"
                summary_rows.append({"Plot": pid2, "Cluster": cl + 1, "Fav. score": fav, "Community": comm_str})

        summary_df = pd.DataFrame(summary_rows)

        comparison_output = mo.vstack([
            mo.md("### Selected plot summary"),
            mo.plain_text(summary_df.to_string(index=False)),
            mo.md("### Relative differences per variable"),
            mo.Html(chart.to_html()),
        ])
    else:
        comparison_output = mo.callout(
            mo.md("Select **2 or more plots** in the control above to compare their environmental conditions."),
            kind="info"
        )

    comparison_output
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 📜 Rolled-Out Species Composition
    """)
    return


@app.cell
def _(
    CLUSTER_COLORS,
    cat_filter,
    cat_names,
    cluster_labels,
    mo,
    plot_df,
    selected_plots,
    sp_df,
    species_cat,
    species_names,
):
    def _():
        # ── Rolled-out species bar chart ──────────────────────────────────────────

        active_cats = cat_filter.value if cat_filter.value else cat_names

        # Filter species to selected categories only
        active_species = [
            s for s in species_names
            if species_cat.get(s) in active_cats
        ]

        # Filter species that appear in at least one selected plot (or all if none selected)
        show_plots = selected_plots if selected_plots else list(sp_df.index[:6])
        show_plots = [p for p in show_plots if p in sp_df.index]

        if not show_plots:
            rolled_out = mo.callout(
                mo.md("No matching plots. Select plots above or check your filter."),
                kind="warn"
            )
        else:
            # Subset matrix: plots × active_species
            sub = sp_df.loc[show_plots, [s for s in active_species if s in sp_df.columns]]

            # Drop species with all-zero across selected plots
            sub = sub.loc[:, (sub > 0).any(axis=0)]

            if sub.empty:
                rolled_out = mo.callout(mo.md("No species with non-zero cover for these plots."), kind="warn")
            else:
                # Assign a colour per plot from cluster
                def plot_color(pid):
                    idx = list(plot_df["plot"]).index(pid) if pid in list(plot_df["plot"]) else 0
                    cl = cluster_labels[idx] if idx < len(cluster_labels) else 0
                    return CLUSTER_COLORS[cl % len(CLUSTER_COLORS)]

                # Build SVG
                bar_h    = 16
                gap      = 4
                label_w  = 200
                bar_max  = 400
                plot_spacing = 26
                n_sp     = len(sub.columns)
                total_h  = n_sp * (bar_h + gap) + 80
                n_plots  = len(show_plots)

                svgl = []
                W_r = label_w + bar_max + 20 + n_plots * 30
                H_r = total_h + 60
                svgl.append(f'<svg viewBox="0 0 {W_r} {H_r}" xmlns="http://www.w3.org/2000/svg" '
                            f'style="width:100%;font-family:sans-serif">')
                svgl.append(f'<rect width="{W_r}" height="{H_r}" fill="#f8f9f0" rx="8"/>')

                # Title
                svgl.append(f'<text x="{W_r//2}" y="22" text-anchor="middle" '
                            f'font-size="13" font-weight="bold" fill="#2c3e2d">'
                            f'Species composition — selected plots</text>')

                # Column headers (plot IDs)
                for j, pid in enumerate(show_plots):
                    hx = label_w + bar_max + 10 + j * 28 + 14
                    col = plot_color(pid)
                    svgl.append(f'<rect x="{hx-12}" y="30" width="24" height="14" '
                                f'rx="3" fill="{col}" fill-opacity="0.8"/>')
                    svgl.append(f'<text x="{hx}" y="41" text-anchor="middle" '
                                f'font-size="8" fill="white" font-weight="bold">{pid}</text>')

                # Species rows
                for i, sp_name in enumerate(sub.columns):
                    y_row = 60 + i * (bar_h + gap)

                    # Category colour background
                    cat = species_cat.get(sp_name)
                    cat_idx = cat_names.index(cat) if cat in cat_names else -1
                    bg_col = "#e8f0e9" if i % 2 == 0 else "#f3f7f3"
                    svgl.append(f'<rect x="0" y="{y_row-1}" width="{W_r}" height="{bar_h+2}" '
                                f'fill="{bg_col}"/>')

                    # Species label
                    short_name = sp_name[:32]
                    svgl.append(f'<text x="{label_w-4}" y="{y_row+11}" text-anchor="end" '
                                f'font-size="8.5" fill="#333" font-style="italic">{short_name}</text>')

                    # Max cover across ALL plots for scale
                    max_val = sub[sp_name].max()
                    if max_val == 0:
                        continue

                    # Bar for mean over selected plots
                    mean_val = sub[sp_name].mean()
                    bw = (mean_val / max_val) * bar_max * 0.8
                    svgl.append(f'<rect x="{label_w}" y="{y_row+2}" width="{bw:.1f}" height="{bar_h-4}" '
                                f'rx="2" fill="#aac8a7" fill-opacity="0.6"/>')

                    # Dot per selected plot
                    for j, pid in enumerate(show_plots):
                        val = sub.loc[pid, sp_name]
                        if val > 0:
                            dot_x = label_w + (val / max_val) * bar_max * 0.8
                            col = plot_color(pid)
                            dot_y = y_row + bar_h // 2
                            svgl.append(f'<circle cx="{dot_x:.1f}" cy="{dot_y}" r="4" '
                                        f'fill="{col}" fill-opacity="0.85" stroke="white" stroke-width="0.5">'
                                        f'<title>{sp_name}: {val} (plot {pid})</title></circle>')

                    # Max label
                    svgl.append(f'<text x="{label_w + bar_max*0.8+4}" y="{y_row+11}" '
                                f'font-size="7.5" fill="#666">{max_val:.0f}</text>')

                svgl.append("</svg>")
                rolled_out = mo.Html("\n".join(svgl))
        return rolled_out


    _()
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🧾 Data Summary
    """)
    return


@app.cell
def _(cluster_labels, k, mo, np, pd, plot_df):
    def _():
        # ── Summary table ─────────────────────────────────────────────────────────
        summary_rows = []
        for c in range(k):
            mask = np.array(cluster_labels) == c
            n = mask.sum()
            communities = plot_df.loc[mask, "community"].value_counts().index[:3].tolist()
            summary_rows.append({
                "Cluster": c + 1,
                "Plots (n)": int(n),
                "Top communities": " | ".join(communities),
            })

        summary_tbl = pd.DataFrame(summary_rows)
        return mo.vstack([
            mo.md("### Cluster overview"),
            mo.plain_text(summary_tbl.to_string(index=False)),
            mo.md(f"""
        **Dataset:** 87 vegetation plots, 286 species, 31 plant communities

        **Species categories (clock positions):** Salix · Carex · Eriophorum · Dryas · Cladonia/Lichens · Mosses · Betula & Shrubs · Grasses & Forbs

        **Clustering:** KMeans on soil texture, chemistry, and environmental variables (scaled, median-imputed)

        **Radius:** composite z-score of selected environmental variables → favourability index (outer edge = more favourable / stable)

        **Angle:** weighted circular mean of species category cover values per plot
            """)
        ])


    _()
    return


if __name__ == "__main__":
    app.run()
