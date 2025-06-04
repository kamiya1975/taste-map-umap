# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from scipy.spatial.distance import cdist
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
st.title("TasteMAP UMAP ＋ 等高線 ＋ 一致度")

# ✅ 等高線 軸選択
selected_feature = st.selectbox("等高線軸を選択", list(feature_components.keys()))

# ✅ 合成Z軸
components = feature_components[selected_feature]
z_combined = df[components].sum(axis=1).values
umap_df["Z"] = z_combined

# ✅ 基準ワイン選択
selected_wine = st.selectbox("🔍 近いワインを出す基準ワインを選択", umap_df["商品名"].tolist())

# ✅ 一致度計算
target_row = umap_df[umap_df["商品名"] == selected_wine].iloc[0]
target_xyz = np.array([[target_row["UMAP1"], target_row["UMAP2"], target_row["Z"]]])
all_xyz = umap_df[["UMAP1", "UMAP2", "Z"]].values
distances = cdist(target_xyz, all_xyz).flatten()
umap_df["distance"] = distances
df_sorted = umap_df.sort_values("distance").head(10)

# ✅ Plotly 図
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

# --- 散布図（小さく） ---
fig.add_trace(go.Scatter(
    x=umap_df["UMAP1"],
    y=umap_df["UMAP2"],
    mode='markers',
    marker=dict(
        size=3,
        color=umap_df["Type"].map(color_map),
        opacity=0.85,
        line=dict(width=0.5, color='black')
    ),
    text=umap_df["商品名"],
    name="ワイン"
))

# --- ピン（基準ワイン） ---
fig.add_trace(go.Scatter(
    x=[target_row["UMAP1"]],
    y=[target_row["UMAP2"]],
    mode='markers+text',
    marker=dict(size=18, color='black', symbol='circle-open'),
    text=[selected_wine],
    textposition='top center',
    name='Selected'
))

# --- レイアウト (4:3固定 + 軸比固定) ---
fig.update_layout(
    autosize=False,
    width=800,
    height=600,  # 4:3
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    title=""
)

# ★ 軸比を完全固定（これが超効く！）
fig.update_yaxes(scaleanchor="x")


# ✅ 表示
st.plotly_chart(fig, use_container_width=False)

# ✅ 一致度 TOP10 表
st.subheader("📋 近いワイン TOP10")
st.dataframe(df_sorted[["Type", "商品名", "distance"]].reset_index(drop=True))
