# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from scipy.spatial.distance import cdist
import plotly.express as px

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

# ✅ 欠損補完
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# ✅ 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ✅ PCA → UMAP
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_umap = reducer.fit_transform(X_pca)

# ✅ UMAP DataFrame
umap_df = pd.DataFrame(embedding_umap, columns=["UMAP1", "UMAP2"])
umap_df["JAN"] = df["JAN"].astype(str)
umap_df["Type"] = df["Type"] if "Type" in df.columns else "Unknown"
umap_df["商品名"] = df["商品名"] if "商品名" in df.columns else umap_df["JAN"]

# ✅ Streamlit UI
st.set_page_config(page_title="TasteMAP UMAP", layout="wide")

st.title("TasteMAP UMAP + 等高線 + 一致度")

# ✅ 等高線 軸 選択
selected_feature = st.selectbox("等高線軸を選択", list(feature_components.keys()))

# ✅ Z軸 合成
components = feature_components[selected_feature]
z_combined = df[components].sum(axis=1).values
umap_df["Z"] = z_combined

# ✅ ワイン選択
product_options = umap_df["商品名"].tolist()
selected_product = st.selectbox("🔍 近いワインを出す基準ワインを選択", product_options)

# ✅ 選択位置
selected_row = umap_df[umap_df["商品名"] == selected_product].iloc[0]
target_xyz = np.array([[selected_row["UMAP1"], selected_row["UMAP2"], selected_row["Z"]]])
all_xyz = umap_df[["UMAP1", "UMAP2", "Z"]].values
distances = cdist(target_xyz, all_xyz).flatten()
umap_df["distance"] = distances
df_top10 = umap_df.sort_values("distance").head(10)

# ✅ plotly 等高線 + scatter overlay
fig = px.density_contour(
    umap_df,
    x="UMAP1", y="UMAP2",
    z="Z",
    color_continuous_scale="YlOrBr",
    contours_coloring="fill",
    nbinsx=50,
    nbinsy=50
)

# ✅ scatter trace 追加
scatter_fig = px.scatter(
    umap_df,
    x="UMAP1", y="UMAP2",
    color="Type",
    hover_data=["商品名", "JAN"],
    size=z_combined,
    size_max=12,
    color_discrete_sequence=px.colors.qualitative.Set2
)

for trace in scatter_fig.data:
    fig.add_trace(trace)

# ✅ 赤ピン
fig.add_scatter(
    x=[selected_row["UMAP1"]],
    y=[selected_row["UMAP2"]],
    mode="markers+text",
    marker=dict(color="red", size=20, line=dict(color="black", width=2)),
    text=[selected_product],
    textposition="top center",
    name="Selected"
)

# ✅ 表示
st.plotly_chart(fig, use_container_width=True)

# ✅ TOP10 出力
st.markdown("📋 **近いワイン TOP10**")
st.dataframe(df_top10[["Type", "商品名", "distance"]].reset_index(drop=True))

