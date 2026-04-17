import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np

    species = pd.read_csv("oumalik_species_data.csv", encoding="latin-1").replace(-9999, np.nan)
    env = pd.read_csv("oumalik_environmental_data.csv", encoding="latin-1").replace(-9999, np.nan)
    soils = pd.read_csv("oumalik_soil_data.csv", encoding="latin-1").replace(-9999, np.nan)
    return env, pd, soils, species


@app.cell
def _(env, mo, soils, species):
    mo.md(f"""
    - Species: **{species.shape[0]} rows × {species.shape[1]} cols**
    - Env: **{env.shape[0]} rows × {env.shape[1]} cols**  
    - Soils: **{soils.shape[0]} rows × {soils.shape[1]} cols**
    """)
    return


@app.cell
def _(env):
    env

    return


@app.cell
def _(species_matrix):
    species_matrix
    return


@app.cell
def _(pd):

    df = pd.read_csv("oumalik_species_data.csv", encoding="latin-1", header=None)

    # 1. Extract the plot numbers (row 1, columns 3+)
    plot_numbers = df.iloc[1, 3:]

    # 2. Extract the actual data (row 2+)
    data = df.iloc[3:, :].copy()

    # 3. Get species names (column 2)
    # data["species"] = data.iloc[:, 2]

    # 4. Keep only abundance columns (column 3+)
    species_matrix = data.iloc[:, 3:].copy()

    # 5. Assign correct column names (plot numbers)
    species_matrix.columns = plot_numbers.values

    # 6. Set species as index
    species_matrix.index = data.iloc[:, 2]

    # 7. Convert to numeric
    species_matrix = species_matrix.apply(pd.to_numeric, errors="coerce")
    return (species_matrix,)


@app.cell
def _(species_matrix):
    corr = species_matrix.T.corr()
    return


@app.cell
def _(species_matrix):
    from sklearn.cluster import KMeans

    clusters = KMeans(n_clusters=33).fit_predict(species_matrix.T)
    return (clusters,)


@app.cell
def _(clusters):
    clusters
    return


@app.cell
def _(clusters, env, pd):
    comparison_norm = pd.crosstab(
        env["plant_community_name"],
        clusters,
        normalize="index"
    )

    comparison = pd.crosstab(
        env["plant_community_name"],
        clusters
    )

    import matplotlib.pyplot as plt

    plt.imshow(comparison_norm, cmap='hot', interpolation='nearest')
    plt.show()

    return (plt,)


@app.cell
def _(clusters, env, pd, plt):
    from sklearn.metrics import adjusted_rand_score

    ari = adjusted_rand_score(
        env["plant_community_name"],
        clusters
    )

    print(ari)

    from sklearn.metrics import normalized_mutual_info_score

    nmi = normalized_mutual_info_score(
        env["plant_community_name"],
        clusters
    )

    print(nmi)


    pd.crosstab(
        env["community"],
        clusters
    ).plot(kind="bar", stacked=True, figsize=(12,5))

    plt.ylabel("Number of plots")
    plt.title("Cluster composition within plant communities")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
