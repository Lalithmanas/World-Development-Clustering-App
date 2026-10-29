import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="World Development Clustering", page_icon="🌍", layout="wide")

BASE = Path(__file__).parent / "models"


@st.cache_resource
def load_artifacts():
    return (
        joblib.load(BASE / "scaler.pkl"),
        joblib.load(BASE / "pca_model.pkl"),
        joblib.load(BASE / "kmeans_model.pkl"),
        joblib.load(BASE / "pca_feature_columns.pkl"),
    )


@st.cache_data
def load_data():
    p = BASE / "clustered_data.csv"
    return pd.read_csv(p) if p.exists() else None


scaler, pca, kmeans, cols = load_artifacts()
data = load_data()
n_pcs = pca.n_components_


def predict(df):
    scaled = scaler.transform(df[cols].astype(float))
    pcs = pca.transform(scaled)
    return kmeans.predict(pcs), pcs


st.title("🌍 World Development Clustering")
st.caption(f"{len(cols)} indicators → PCA ({n_pcs} components) → KMeans ({kmeans.n_clusters} clusters)")

tab1, tab2, tab3, tab4 = st.tabs(["Predict a country", "Upload CSV", "Cluster profiles", "About the model"])

# ---------- Predict ----------
with tab1:
    st.subheader("Enter indicator values (raw, not scaled)")
    means = dict(zip(cols, scaler.mean_))
    grid = st.columns(3)
    vals = {}
    for i, c in enumerate(cols):
        vals[c] = grid[i % 3].number_input(c, value=float(round(means[c], 2)), key=f"in_{c}")

    if st.button("Predict cluster", type="primary"):
        labels, pcs = predict(pd.DataFrame([vals]))
        st.success(f"This country belongs to **Cluster {labels[0]}**")
        dist = kmeans.transform(pcs)[0]
        st.write("Distance to each cluster centre (smaller = closer)")
        st.bar_chart(pd.Series(dist, index=[f"Cluster {k}" for k in range(len(dist))]))
        if data is not None:
            st.write(f"Cluster {labels[0]} average vs your input")
            avg = data[data["Cluster"] == labels[0]][cols].mean()
            st.dataframe(pd.DataFrame({"Your input": pd.Series(vals), "Cluster average": avg}).round(2))

# ---------- Upload ----------
with tab2:
    st.write("Required columns:", ", ".join(cols))
    f = st.file_uploader("Upload CSV", type="csv")
    if f:
        up = pd.read_csv(f)
        missing = [c for c in cols if c not in up.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            up = up.dropna(subset=cols).reset_index(drop=True)
            labels, pcs = predict(up)
            up["Cluster"] = labels
            st.dataframe(up, width='stretch')
            st.bar_chart(up["Cluster"].value_counts().sort_index())
            st.scatter_chart(
                pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "Cluster": labels.astype(str)}),
                x="PC1", y="PC2", color="Cluster",
            )
            st.download_button("Download results", up.to_csv(index=False), "clustered.csv", "text/csv")

# ---------- Profiles ----------
with tab3:
    if data is None:
        st.info("clustered_data.csv not found in models/.")
    else:
        st.write("Average of each indicator per cluster")
        st.dataframe(data.groupby("Cluster")[cols].mean().round(2))
        st.write("Countries per cluster")
        st.bar_chart(data.groupby("Cluster")["Country"].nunique())
        pick = st.selectbox("Look up a country", sorted(data["Country"].unique()))
        st.dataframe(data[data["Country"] == pick].reset_index(drop=True))

# ---------- About ----------
with tab4:
    evr = pca.explained_variance_ratio_
    st.write(f"**Variance explained by {n_pcs} components:** {evr.sum():.1%}")
    st.bar_chart(pd.Series(evr, index=[f"PC{i+1}" for i in range(n_pcs)]))
    st.write("**Indicators used:**", ", ".join(cols))
