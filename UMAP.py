# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import plotly.graph_objects as go

# ✅ データ読み込み
df = pd.read_csv("Merged_TasteDataDB15.csv")

# ✅ 官能軸セット
feature_components = {
    "甘味": ["ブドウ糖", "果糖"],
    "酸味": ["リンゴ酸", "酒石酸"],
    "渋味": ["総ポリフェノール", "グリセリン", "pH"],
}

# ✅ 特徴量リスト
features = [
    "20mm_L*", "20mm_a*", "20mm_b*",
    "Ethyl acetate", "Propanol", "2-Methylbutyl acetate", "Ethyl lactate", "Acetic acid",
    "Furfural", "2-Acetylfuran", "2-Nonanol", "Propanoic acid", "2,3-Butanediol isomer-1",
    "Isobutyric acid", "2,3-Butanediol isomer-2", "gamma-Butyrolactone", "Butyric acid",
    "Furfuryl alcohol", "Isovaleric acid", "2-Methyl butyric acid", "Diethyl succinate",
    "Methionol", "Cyclotene", "2-Phenylethyl acetate", "3-Mercapto-1-hexanol",
    "Capronic acid", "Guaiacol", "1-Undecanol", "Benzyl alcohol", "2-Phenylethanol",
    "Maltol", "2-Acetylpyrrole", "Phenol", "Furaneol", "gamma-Nonalactone",
    "Pantolactone", "Diethyl malate", "Caprylic acid", "p-Cresol", "4-Ethylphenol",
    "3-Ethylphenol", "p-Vinylguaiacol", "2,6-Dimethoxyphenol", "Benzoic acid",
    "エタノール", "ブドウ糖", "果糖", "滴定酸度7.0",
    "揮発性酸", "リンゴ酸", "乳酸", "酒石酸", "エキス", "pH", "グリセリン",
    "グルコン酸", "総ポリフェノール"
]

# ✅ 特徴量データ
X = df[features]
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imputed)

# ✅ PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# ✅ UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_pca)

# ✅ UMAP DataFrame
umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
umap_df["PC1"] = X_pca[:, 0]
umap_df["PC2"] = X_pca[:, 1]
umap_df["JAN"] = df["JAN"].astype(str)
umap_df["Type"] = df["Type"] if "Type" in df.columns else "Unknown"
umap_df["商品名"] = df["商品名"] if "商品名" in df.columns else umap_df["JAN"]

# ✅ Streamlit UI
st.title("UMAP")

# ✅ 等高線 軸選択
selected_feature = st.selectbox("等高線軸を選択", list(feature_components.keys()))

# ✅ 合成Z軸
components = feature_components[selected_feature]
z_combined = df[components].sum(axis=1).values
umap_df["Z"] = z_combined

# ✅ Plotly図
fig = go.Figure()

# --- カラーマップ ---
color_map = {
    "White": "green",
    "Red": "red",
    "Spa": "blue",
    "Rose": "pink"
}

# --- 等高線 ---
fig.add_trace(go.Contour(
    x=umap_df["UMAP1"],
    y=umap_df["UMAP2"],
    z=umap_df["Z"],
    colorscale='YlOrBr',
    opacity=0.3,
    showscale=False,
    contours=dict(coloring='heatmap', showlines=False)
))

# --- 散布図 ---
fig.add_trace(go.Scatter(
    x=umap_df["UMAP1"],
    y=umap_df["UMAP2"],
    mode='markers',
    marker=dict(
        size=4,
        color=umap_df["Type"].map(color_map),
        opacity=0.85,
        line=dict(width=0.3, color='black')
    ),
    text=umap_df["商品名"],
    name="ワイン"
))

# --- レイアウト ---
fig.update_layout(
    showlegend=False,
    title="",
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis_title="UMAP1",
    yaxis_title="UMAP2",
    height=600,
    autosize=True
)

# ✅ グラフ表示
st.plotly_chart(fig, use_container_width=True)
