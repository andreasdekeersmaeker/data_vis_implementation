import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from scipy.spatial import ConvexHull
    import warnings
    warnings.filterwarnings("ignore")
    return ConvexHull, KMeans, SimpleImputer, StandardScaler, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 🌿 Oumalik Vegetation Plot Explorer

    **How to read this:**
    - **Angle** on the circle = dominant species category of that plot
    - **Distance from centre** = favourability / stability of selected environmental conditions (outer = more favourable)
    - **Colour** = cluster of similar soil + environmental conditions
    - **Coloured blob** = convex hull wrapping each cluster

    **👆 Click any dot on the clock to select / deselect it for comparison.**
    """)
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

    sp_df = pd.DataFrame(cover_matrix.T, index=author_nums, columns=species_names)

    # ── 8 categories placed around the clock (~99.7% of total cover) ──────────
    CATEGORIES = {
        "Salix\n(Willows)": lambda n: n.startswith("Salix"),
        "Carex\n(Sedges)":  lambda n: n.startswith("Carex"),
        "Eriophorum\n(Cotton grass)": lambda n: n.startswith("Eriophorum"),
        "Dryas": lambda n: n.startswith("Dryas"),
        "Lichens": lambda n: any(n.startswith(g) for g in [
            "Cladonia","Peltigera","Alectoria","Ochrolechia","Flavocetraria",
            "Dactylina","Thamnolia","Stereocaulon","Cetraria","Bryocaulon",
            "Gowardia","Masonhalea","Nephroma","Parmelia","Pertusaria",
            "Physconia","Polyblastia","Rinodina","Solorina","Sphaerophorus",
            "Lecanora","Lobaria","Hypogymnia",
        ]),
        "Mosses &\nLiverworts": lambda n: any(n.startswith(g) for g in [
            "Sphagnum","Bryum","Pohlia","Aulacomnium","Hypnum","Polytrichum",
            "Campylium","Dicranum","Encalypta","Brachythecium","Drepanocladus",
            "Meesia","Paludella","Scorpidium","Calliergon","Warnstorfia","Mnium",
            "Rhytidium","Hylocomium","Tomentypnum","Ditrichum","Hamatocaulis",
            "Ceratodon","Limprichtia","Oncophorus","Sanionia","Cinclidium",
            "Cirriphyllum","Fissidens","Isopterygiopsis","Myurella","Philonotis",
            "Plagiomnium","Rhizomnium","Racomitrium","Hygrohypnum","Amblystegium",
            "Barbula","Cynodontium","Dicranella","Distichium","Psilopilum",
            "Tetraplodon","Timmia","Trematodon","Leptobryum","Loeskypnum",
            "Ptilidium","Barbilophozia","Lophozia","Blepharostoma","Cephalozia",
            "Cephaloziella","Calypogeia","Aneura","Gymnocolea","Jungermannia",
            "Lejeunea","Lophoziopsis","Orthocaulis","Scapania","Tritomaria",
            "Sphenolobus","Marchantia","Unknown moss","Unknown liverworts",
        ]),
        "Betula &\nShrubs": lambda n: any(n.startswith(g) for g in [
            "Betula","Vaccinium","Ledum","Cassiope","Arctous","Rubus",
            "Andromeda","Empetrum","Rhododendron","Pyrola","Orthilia",
        ]),
        "Grasses\n& Forbs": lambda n: any(n.startswith(g) for g in [
            "Poa","Arctagrostis","Puccinellia","Hierochloe","Hierochlo",
            "Equisetum","Saxifraga","Pedicularis","Bistorta","Anemone",
            "Astragalus","Draba","Stellaria","Luzula","Juncus","Micranthes",
            "Artemisia","Arnica","Festuca","Arctophila","Calamagrostis",
            "Kobresia","Trichophorum","Eleocharis","Hippuris","Triglochin",
            "Petasites","Cardamine","Saussurea","Polemonium","Caltha",
            "Chrysosplenium","Descurainia","Gentianella","Parnassia","Parrya",
            "Tephroseris","Tofieldia","Valeriana","Lupinus","Chamerion",
            "Wilhelmsia","Comarum","Ranunculus","Potentilla","Epilobium",
            "Cerastium","Oxyria","Rumex","Silene","Minuartia",
        ]),
    }

    cat_names  = list(CATEGORIES.keys())
    cat_angles = {n: 2 * np.pi * i / len(cat_names) for i, n in enumerate(cat_names)}

    species_cat = {
        s: next((c for c, fn in CATEGORIES.items() if fn(s)), None)
        for s in species_names
    }

    # Cover per plot per category
    cat_cover = pd.DataFrame(index=sp_df.index, columns=cat_names, dtype=float)
    for _cat in cat_names:
        _cols = [s for s in species_names if species_cat.get(s) == _cat]
        cat_cover[_cat] = sp_df[_cols].sum(axis=1) if _cols else 0.0

    # Circular-mean angle weighted by category cover
    _angle_vals = np.array(list(cat_angles.values()))

    def _circ_mean(covers, angles):
        total = covers.sum()
        if total == 0:
            return 0.0
        sin_s = sum(c * np.sin(angles[i]) for i, c in enumerate(covers))
        cos_s = sum(c * np.cos(angles[i]) for i, c in enumerate(covers))
        return np.arctan2(sin_s, cos_s) % (2 * np.pi)

    plot_angles = {
        pid: _circ_mean(cat_cover.loc[pid, cat_names].values.astype(float), _angle_vals)
        for pid in sp_df.index
    }

    merged = pd.merge(
        soil_raw, env_raw, on="plot_number", suffixes=("_soil", "_env")
    ).set_index("plot_number")
    return cat_angles, cat_names, merged, plot_angles


@app.cell
def _(mo):
    mo.md("""
    ---
    ## ⚙️ Controls
    """)
    return


@app.cell
def _(mo):
    n_clusters_slider = mo.ui.slider(
        start=2, stop=8, step=1, value=4,
        label="Number of colour clusters (k)"
    )
    n_clusters_slider
    return (n_clusters_slider,)


@app.cell
def _(mo):
    ENV_VAR_OPTIONS = {
        "Soil moisture":         "soil_moisture ",
        "Organic matter":        "organic_matter",
        "pH":                    "pH",
        "Available water":       "available_water",
        "Field capacity":        "field_capacity",
        "Carbonates":            "carbonates ",
        "Cation exchange cap.":  "cation_ex_capacity",
        "Summer air temp":       "summer_air_temp ",
        "Duration snow":         "duration_snow",
        "Wind regime":           "wind_regime ",
        "Disturbance intensity": "disturbance_intensity",
        "Microrelief height":    "microrelief_ht ",
        "Cover bare soil+rock":  "_bare",
        "Cover water+litter":    "_wet",
        "Vegetation height":     "_veg_ht",
        "Aspect x cos(slope)":   "_aspect_slope",
    }

    env_multiselect = mo.ui.multiselect(
        options=list(ENV_VAR_OPTIONS.keys()),
        value=[
            "Soil moisture","Organic matter","pH",
            "Summer air temp","Cover bare soil+rock","Cover water+litter",
        ],
        label="Environmental variables driving the radius (favourability score)"
    )
    env_multiselect
    return ENV_VAR_OPTIONS, env_multiselect


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
    # ── Build composite data frame + cluster + radius ─────────────────────────
    df = merged.copy()

    df["_bare"]         = df["cover_bare_soil "].fillna(0) + df["cover_bare_rock"].fillna(0)
    df["_wet"]          = df["cover_water "].fillna(0) + df["cover_litter_layer"].fillna(0)
    df["_veg_ht"]       = (
        df["canopy_ht"].fillna(0) + df["shrub_layer_ht"].fillna(0)
        + df["herb_layer_ht"].fillna(0)
        + pd.to_numeric(df.get("moss_layer_ht", 0), errors="coerce").fillna(0)
    )
    df["_aspect_slope"] = df["aspect"].fillna(0) * np.cos(np.radians(df["slope"].fillna(0)))

    CLUSTER_COLS = [
        "sand","silt","clay","organic_matter","pH","carbonates ",
        "soil_moisture ","field_capacity","wilting_point","available_water",
        "cation_ex_capacity","microrelief_ht ","cover_graminoids","cover_forbs",
        "cover_mosses_liverworts","cover_lichen_layer","thaw_depth ","site_moisture ",
        "summer_air_temp ","duration_snow","wind_regime ","disturbance_intensity",
        "_bare","_wet","_veg_ht","_aspect_slope",
    ]
    _cols = [c for c in CLUSTER_COLS if c in df.columns]
    _X    = SimpleImputer(strategy="median").fit_transform(df[_cols].values.astype(float))
    _Xsc  = StandardScaler().fit_transform(_X)

    k = n_clusters_slider.value
    cluster_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(_Xsc)

    # Favourability radius
    _sel_labels = env_multiselect.value or list(ENV_VAR_OPTIONS.keys())
    _sel_cols   = [ENV_VAR_OPTIONS[l] for l in _sel_labels if ENV_VAR_OPTIONS[l] in df.columns]
    if not _sel_cols:
        _sel_cols = ["soil_moisture "]

    _Xr  = SimpleImputer(strategy="median").fit_transform(df[_sel_cols].values.astype(float))
    _Xrs = StandardScaler().fit_transform(_Xr)
    fav  = _Xrs.mean(axis=1)
    _fn, _fx = fav.min(), fav.max()
    radius_norm = 0.15 + 0.80 * (fav - _fn) / (_fx - _fn + 1e-9)

    CLUSTER_COLORS = [
        "#e6534a","#4a90d9","#6cc47b","#f5a623",
        "#9b59b6","#1abc9c","#e67e22","#2c3e50",
    ]

    CIRC_R, CX, CY = 285, 355, 365

    plot_df = pd.DataFrame({
        "plot":      df.index.tolist(),
        "angle":     [plot_angles.get(p, 0.0) for p in df.index],
        "radius":    radius_norm,
        "cluster":   cluster_labels,
        "colour":    [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in cluster_labels],
        "fav":       fav,
        "community": (
            df["plant_community_name_soil"].values
            if "plant_community_name_soil" in df.columns
            else df.get("plant_community_name", pd.Series([""] * len(df))).values
        ),
    })

    plot_df["x"] = CX + plot_df["radius"] * CIRC_R * np.cos(plot_df["angle"] - np.pi / 2)
    plot_df["y"] = CY + plot_df["radius"] * CIRC_R * np.sin(plot_df["angle"] - np.pi / 2)
    return CIRC_R, CLUSTER_COLORS, CX, CY, cluster_labels, df, fav, k, plot_df


@app.cell
def _(mo):
    # Reactive text field that JavaScript writes plot IDs into.
    # It IS shown so Leena can also type / paste IDs directly.
    selected_state = mo.ui.text(
        value="",
        label="Selected plots (click dots on clock, or type IDs separated by commas)",
        full_width=True,
    )
    selected_state
    return (selected_state,)


@app.cell
def _(selected_state):
    def _parse(s):
        ids = []
        for tok in s.replace(" ", "").split(","):
            try:
                ids.append(int(tok))
            except ValueError:
                pass
        return list(dict.fromkeys(ids))

    selected_plots = _parse(selected_state.value)
    return (selected_plots,)


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
    mo,
    np,
    plot_df,
    selected_plots,
):
    def _make_clock(pdf, selected):
        W, H = 710, 730
        sel_set = set(selected)
        lines = []

        lines.append(
            f'<svg id="oumalik-clock" viewBox="0 0 {W} {H}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;max-width:{W}px;font-family:sans-serif;cursor:pointer">'
        )
        lines.append(f'<rect width="{W}" height="{H}" fill="#f7f8f2" rx="14"/>')
        lines.append(
            f'<text x="{W//2}" y="28" text-anchor="middle" font-size="14" '
            f'font-weight="700" fill="#2c3e2d" letter-spacing="0.4">'
            f'Oumalik Plot Ordination Clock</text>'
        )
        lines.append(
            f'<text x="{W//2}" y="46" text-anchor="middle" font-size="9.5" fill="#999">'
            f'Click dots to select / deselect for comparison</text>'
        )

        # Guide rings with labels
        for frac, lbl in [(0.33, "stressed"), (0.67, ""), (1.0, "favourable")]:
            r = frac * CIRC_R
            lines.append(
                f'<circle cx="{CX}" cy="{CY}" r="{r:.1f}" fill="none" '
                f'stroke="#ddd" stroke-width="1" stroke-dasharray="4,4"/>'
            )
            if lbl:
                ang_lbl = np.radians(-38)
                lx = CX + r * np.cos(ang_lbl)
                ly = CY + r * np.sin(ang_lbl)
                lines.append(
                    f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="7.5" '
                    f'fill="#ccc" text-anchor="middle">{lbl}</text>'
                )

        lines.append(
            f'<circle cx="{CX}" cy="{CY}" r="{CIRC_R}" '
            f'fill="none" stroke="#aaa" stroke-width="1.5"/>'
        )

        # Spokes + labels
        for cat in cat_names:
            ang = cat_angles[cat] - np.pi / 2
            sx = CX + 18 * np.cos(ang);  sy = CY + 18 * np.sin(ang)
            ex = CX + CIRC_R * np.cos(ang); ey = CY + CIRC_R * np.sin(ang)
            lines.append(
                f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
                f'stroke="#ccc" stroke-width="1" stroke-dasharray="3,3"/>'
            )
            lx = CX + (CIRC_R + 44) * np.cos(ang)
            ly = CY + (CIRC_R + 44) * np.sin(ang)
            anchor = "middle"
            if np.cos(ang) > 0.28:   anchor = "start"
            elif np.cos(ang) < -0.28: anchor = "end"
            # Multi-line label: split on \n
            parts = cat.split("\n")
            for pi2, part in enumerate(parts):
                dy = ly + pi2 * 11 - (len(parts) - 1) * 5
                lines.append(
                    f'<text x="{lx:.1f}" y="{dy:.1f}" text-anchor="{anchor}" '
                    f'font-size="9.5" font-weight="600" fill="#3a5a3b">{part}</text>'
                )

        # Cluster blobs
        for c in range(k):
            cdf = pdf[pdf["cluster"] == c]
            if len(cdf) >= 3:
                pts = cdf[["x","y"]].values
                try:
                    hull = ConvexHull(pts)
                    hp = pts[hull.vertices]
                    centroid = hp.mean(axis=0)
                    inflated = centroid + 1.14 * (hp - centroid)
                    path_d = "M " + " L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in inflated) + " Z"
                    col = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
                    lines.append(
                        f'<path d="{path_d}" fill="{col}" fill-opacity="0.08" '
                        f'stroke="{col}" stroke-width="1.8" stroke-dasharray="6,3" '
                        f'stroke-opacity="0.4" stroke-linejoin="round"/>'
                    )
                except Exception:
                    pass

        # Plot dots
        for _, row in pdf.iterrows():
            pid = int(row["plot"])
            x, y = float(row["x"]), float(row["y"])
            col = row["colour"]
            is_sel = pid in sel_set

            if is_sel:
                lines.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" '
                    f'fill="none" stroke="#111" stroke-width="2.8" stroke-opacity="0.8"/>'
                )
                lines.append(
                    f'<text x="{x:.1f}" y="{y-15:.1f}" text-anchor="middle" '
                    f'font-size="9" font-weight="700" fill="#111">{pid}</text>'
                )

            lines.append(
                f'<circle class="plot-dot" data-pid="{pid}" '
                f'cx="{x:.1f}" cy="{y:.1f}" r="7" '
                f'fill="{col}" stroke="white" stroke-width="1.5" opacity="0.88">'
                f'<title>Plot {pid} | {row["community"]} | '
                f'Cluster {int(row["cluster"])+1} | Fav {float(row["fav"]):.2f}</title>'
                f'</circle>'
            )

        # Centre cross
        lines.append(f'<line x1="{CX-7}" y1="{CY}" x2="{CX+7}" y2="{CY}" stroke="#888" stroke-width="1.2"/>')
        lines.append(f'<line x1="{CX}" y1="{CY-7}" x2="{CX}" y2="{CY+7}" stroke="#888" stroke-width="1.2"/>')

        # Legend
        lx0 = 14
        ly0 = H - 14 - k * 20
        lines.append(f'<text x="{lx0}" y="{ly0-14}" font-size="9" font-weight="700" fill="#444">Clusters</text>')
        for c in range(k):
            col = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            ly_c = ly0 + c * 20
            lines.append(f'<circle cx="{lx0+6}" cy="{ly_c}" r="5.5" fill="{col}"/>')
            lines.append(f'<text x="{lx0+18}" y="{ly_c+4}" font-size="8.5" fill="#333">Cluster {c+1}</text>')

        # ── JavaScript click handler ───────────────────────────────────────────
        # Finds the marimo text <input> by its label text and toggles the clicked
        # plot ID in the comma-separated value, then fires input + change events.
        lines.append(r"""
    <script>
    (function() {
      function findMarimoInput(labelSubstr) {
    // Marimo wraps inputs in various ways; search all visible text inputs
    var all = document.querySelectorAll('input[type="text"], input:not([type])');
    for (var i = 0; i < all.length; i++) {
      var el = all[i];
      // Check nearby label text (parent chain up to 5 levels)
      var node = el;
      for (var depth = 0; depth < 6; depth++) {
        if (!node) break;
        if (node.textContent && node.textContent.includes(labelSubstr)) return el;
        node = node.parentElement;
      }
    }
    return null;
      }

      function nativeInputValue(input, newVal) {
    // Works with React-controlled inputs (marimo uses React internally)
    var nativeSetter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value'
    ).set;
    nativeSetter.call(input, newVal);
    input.dispatchEvent(new Event('input',  { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
      }

      function init() {
    var svg = document.getElementById('oumalik-clock');
    if (!svg) { setTimeout(init, 300); return; }

    svg.addEventListener('click', function(e) {
      var dot = e.target.closest('.plot-dot');
      if (!dot) return;
      var pid = dot.getAttribute('data-pid');
      if (!pid) return;

      var input = findMarimoInput('Selected plots');
      if (!input) {
        console.warn('Oumalik: could not find marimo text input');
        return;
      }

      var current = input.value
        ? input.value.split(',').map(function(s){ return s.trim(); }).filter(Boolean)
        : [];
      var idx = current.indexOf(pid);
      if (idx === -1) { current.push(pid); }
      else            { current.splice(idx, 1); }

      nativeInputValue(input, current.join(','));
    });
      }

      // Retry until the SVG and inputs are mounted
      init();
      document.addEventListener('DOMContentLoaded', init);
    })();
    </script>
    """)
        lines.append("</svg>")
        return "\n".join(lines)

    clock_svg    = _make_clock(plot_df, selected_plots)
    clock_widget = mo.Html(clock_svg)
    return (clock_widget,)


@app.cell
def _(clock_widget, mo):
    mo.md(f"""
    ## 🕐 Ordination Clock\n\n{clock_widget}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🔍 Environmental Comparison
    """)
    return


@app.cell
def _(mo):
    COMP_VAR_OPTIONS = {
        "Soil moisture":         "soil_moisture ",
        "Organic matter":        "organic_matter",
        "pH":                    "pH",
        "Sand (%)":              "sand",
        "Silt (%)":              "silt",
        "Clay (%)":              "clay",
        "Available water":       "available_water",
        "Field capacity":        "field_capacity",
        "Wilting point":         "wilting_point",
        "Carbonates":            "carbonates ",
        "Cation exch. cap.":     "cation_ex_capacity",
        "NH4":                   "NH4",
        "NO3":                   "NO3",
        "Nitrogen (N)":          "N",
        "Phosphorus (P)":        "P",
        "Potassium (K)":         "K",
        "Summer air temp":       "summer_air_temp ",
        "Duration snow":         "duration_snow",
        "Wind regime":           "wind_regime ",
        "Disturbance intensity": "disturbance_intensity",
        "Microrelief height":    "microrelief_ht ",
        "Thaw depth":            "thaw_depth ",
        "Site moisture":         "site_moisture ",
        "Cover bare soil+rock":  "_bare",
        "Cover water+litter":    "_wet",
        "Vegetation height":     "_veg_ht",
        "Aspect x cos(slope)":   "_aspect_slope",
        "Favourability score":   "_fav",
    }

    comp_var_select = mo.ui.multiselect(
        options=list(COMP_VAR_OPTIONS.keys()),
        value=[
            "Soil moisture","Organic matter","pH","Available water",
            "Summer air temp","Duration snow","Wind regime",
            "Disturbance intensity","Thaw depth","Favourability score",
        ],
        label="Variables to show in comparison"
    )
    comp_var_select
    return COMP_VAR_OPTIONS, comp_var_select


@app.cell
def _(
    CLUSTER_COLORS,
    COMP_VAR_OPTIONS,
    cluster_labels,
    comp_var_select,
    df,
    fav,
    mo,
    np,
    pd,
    plot_df,
    selected_plots,
):
    def _build_comparison(sel_plots, df_in, fav_arr, cl_labels, colors,
                          var_opts, chosen_vars):
        if len(sel_plots) < 2:
            return mo.callout(
                mo.md(
                    "**Click 2 or more dots on the clock to compare their environments.**\n\n"
                    "Or type comma-separated plot IDs directly into the text field above."
                ),
                kind="info",
            )

        df2 = df_in.copy()
        df2["_fav"] = fav_arr

        chosen_labels = chosen_vars if chosen_vars else list(var_opts.keys())

        rows = []
        for label in chosen_labels:
            col = var_opts[label]
            if col not in df2.columns:
                continue
            vals = {}
            for pid in sel_plots:
                if pid in df2.index:
                    vals[pid] = float(df2.loc[pid, col])
            if not vals:
                continue
            mean_v = float(np.nanmean(list(vals.values())))
            for pid, v in vals.items():
                rel = ((v - mean_v) / (abs(mean_v) + 1e-9)) * 100.0
                rows.append({
                    "Variable":       label,
                    "Plot":           str(pid),
                    "Value":          round(v, 3),
                    "D_from_mean":    round(rel, 1),
                })

        if not rows:
            return mo.md("No data available for these variables / plots.")

        comp_df = pd.DataFrame(rows)

        # ── Diverging bar SVG ─────────────────────────────────────────────────
        n_plots  = len(sel_plots)
        bar_h    = 13
        row_h    = n_plots * (bar_h + 2) + 8
        label_w  = 190
        bar_half = 210
        total_w  = label_w + bar_half * 2 + 30
        var_order = list(dict.fromkeys(comp_df["Variable"]))
        n_vars   = len(var_order)
        total_h  = n_vars * row_h + 70

        def pid_color(pid):
            pl = list(plot_df["plot"])
            if pid in pl:
                ci = cl_labels[pl.index(pid)]
                return colors[ci % len(colors)]
            return "#888"

        svgl = []
        svgl.append(
            f'<svg viewBox="0 0 {total_w} {total_h}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;font-family:sans-serif">'
        )
        svgl.append(f'<rect width="{total_w}" height="{total_h}" fill="#f7f8f2" rx="10"/>')

        title_plots = ", ".join(str(p) for p in sel_plots)
        svgl.append(
            f'<text x="{total_w//2}" y="22" text-anchor="middle" '
            f'font-size="12" font-weight="700" fill="#2c3e2d">'
            f'Environmental comparison — plots {title_plots}</text>'
        )

        mid_x = label_w + bar_half
        svgl.append(
            f'<line x1="{mid_x}" y1="32" x2="{mid_x}" y2="{total_h-8}" '
            f'stroke="#bbb" stroke-width="0.8"/>'
        )
        svgl.append(
            f'<text x="{mid_x - bar_half//2}" y="42" text-anchor="middle" '
            f'font-size="8" fill="#bbb">← below mean of selection</text>'
        )
        svgl.append(
            f'<text x="{mid_x + bar_half//2}" y="42" text-anchor="middle" '
            f'font-size="8" fill="#bbb">above mean of selection →</text>'
        )

        # Plot-ID colour key (top right)
        for pi2, pid in enumerate(sel_plots):
            col = pid_color(pid)
            kx  = total_w - 6
            ky  = 14 + pi2 * 13
            svgl.append(f'<circle cx="{kx-18}" cy="{ky}" r="5" fill="{col}"/>')
            svgl.append(
                f'<text x="{kx}" y="{ky+4}" text-anchor="end" '
                f'font-size="8.5" font-weight="700" fill="{col}">Plot {pid}</text>'
            )

        for vi, var in enumerate(var_order):
            sub = comp_df[comp_df["Variable"] == var]
            y0  = 50 + vi * row_h
            bg  = "#eef2ee" if vi % 2 == 0 else "#f7f8f2"
            svgl.append(f'<rect x="0" y="{y0-1}" width="{total_w}" height="{row_h}" fill="{bg}"/>')

            # Variable label
            svgl.append(
                f'<text x="{label_w-5}" y="{y0 + row_h//2 + 4}" '
                f'text-anchor="end" font-size="9" fill="#333">{var}</text>'
            )

            for pi2, pid in enumerate(sel_plots):
                row_sub = sub[sub["Plot"] == str(pid)]
                if row_sub.empty:
                    continue
                rel = float(row_sub["D_from_mean"].iloc[0])
                val = float(row_sub["Value"].iloc[0])
                col = pid_color(pid)
                bw  = min(abs(rel) / 100.0 * bar_half, bar_half)
                by  = y0 + pi2 * (bar_h + 2)
                bx  = mid_x if rel >= 0 else mid_x - bw

                svgl.append(
                    f'<rect x="{bx:.1f}" y="{by}" width="{bw:.1f}" height="{bar_h}" '
                    f'rx="2" fill="{col}" fill-opacity="0.72">'
                    f'<title>Plot {pid}: {val:.3g} ({rel:+.1f}%)</title></rect>'
                )

                # Value annotation
                if bw > 18:
                    tx  = bx + bw/2
                    svgl.append(
                        f'<text x="{tx:.1f}" y="{by+bar_h-3}" text-anchor="middle" '
                        f'font-size="7" fill="white" font-weight="600">{val:.3g}</text>'
                    )
                else:
                    tx  = (mid_x + bw + 3) if rel >= 0 else (mid_x - bw - 3)
                    anc = "start" if rel >= 0 else "end"
                    svgl.append(
                        f'<text x="{tx:.1f}" y="{by+bar_h-3}" text-anchor="{anc}" '
                        f'font-size="7" fill="{col}">{val:.3g}</text>'
                    )

        svgl.append("</svg>")

        # ── Summary table ──────────────────────────────────────────────────────
        sum_rows = []
        for pid in sel_plots:
            mask = plot_df["plot"] == pid
            if not mask.any():
                continue
            r = plot_df[mask].iloc[0]
            sum_rows.append({
                "Plot":        pid,
                "Cluster":     int(r["cluster"]) + 1,
                "Fav. score":  round(float(r["fav"]), 3),
                "Community":   str(r["community"])[:60],
            })

        return mo.vstack([
            mo.md("### Selected plots"),
            mo.plain_text(pd.DataFrame(sum_rows).to_string(index=False)),
            mo.md("### Environmental differences\n"
                  "*Bars = each plot's value relative to the mean of all selected plots.*"),
            mo.Html("\n".join(svgl)),
        ])

    comparison_panel = _build_comparison(
        selected_plots, df, fav, cluster_labels, CLUSTER_COLORS,
        COMP_VAR_OPTIONS, comp_var_select.value,
    )
    comparison_panel
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 📋 Cluster Overview
    """)
    return


@app.cell
def _(cluster_labels, k, mo, np, pd, plot_df):
    _rows = []
    for _c in range(k):
        _mask  = np.array(cluster_labels) == _c
        _comms = plot_df.loc[_mask, "community"].value_counts().index[:3].tolist()
        _rows.append({
            "Cluster":         _c + 1,
            "n plots":         int(_mask.sum()),
            "Top communities": " | ".join(_comms),
        })
    mo.vstack([
        mo.plain_text(pd.DataFrame(_rows).to_string(index=False)),
        mo.md("""
    **Species categories** (99.7% of total cover captured):
    Salix · Carex · Eriophorum · Dryas · Lichens · Mosses & Liverworts · Betula & Shrubs · Grasses & Forbs

    **Clustering:** KMeans on 26 soil + environmental variables (median-imputed, z-scaled)

    **Radius:** mean z-score of selected env variables → favourability (outer = more favourable / stable)

    **Angle:** cover-weighted circular mean across the 8 species categories
        """),
    ])
    return


if __name__ == "__main__":
    app.run()
