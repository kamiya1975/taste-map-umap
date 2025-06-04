# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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
umap_df["商品名"] = df["商品名"]
umap_df["Type"] = df["Type"] if "Type" in df.columns else "Unknown"

# ✅ Streamlit App
st.title("TasteMAP UMAP ＋ 等高線 ＋ 一致度")

# ✅ 等高線軸 選択
selected_feature = st.selectbox("等高線軸を選択", list(feature_components.keys()))

# ✅ Z軸 合成（生値の和）
components = feature_components[selected_feature]
z_combined = df[components].sum(axis=1).values
umap_df["Z"] = z_combined

# ✅ 基準ワイン選択
selected_wine = st.selectbox("🔍 近いワインを出す基準ワインを選択", umap_df["商品名"].tolist())

# ✅ 等高線（matplotlib）
fig2, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(
    x=umap_df["UMAP1"], y=umap_df["UMAP2"],
    weights=umap_df["Z"],
    cmap="YlOrBr", fill=True, bw_adjust=0.5, levels=50, alpha=0.6, ax=ax
)
ax.set_title(f"TasteMAP 等高線 ({selected_feature})", fontsize=14)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")

# ✅ 描画
st.pyplot(fig2)

# ✅ Plotly 散布図
fig = px.scatter(
    umap_df,
    x="UMAP1", y="UMAP2",
    color="Type",
    hover_data=["商品名"],
    size=umap_df["Z"],
    size_max=12,
    title=f"TasteMAP UMAP ＋ {selected_feature}：{selected_wine}"
)

# ✅ 選択ワイン → 赤丸
selected_row = umap_df[umap_df["商品名"] == selected_wine].iloc[0]
fig.add_trace(go.Scatter(
    x=[selected_row["UMAP1"]],
    y=[selected_row["UMAP2"]],
    mode="markers+text",
    marker=dict(size=14, color="red", line=dict(width=2, color="black")),
    text=[selected_wine],
    textposition="top center",
    name="Selected"
))

st.plotly_chart(fig, use_container_width=True)

# ✅ 一致度計算
target_xyz = np.array([[selected_row["UMAP1"], selected_row["UMAP2"], selected_row["Z"]]])
all_xyz = umap_df[["UMAP1", "UMAP2", "Z"]].values
distances = cdist(target_xyz, all_xyz).flatten()
umap_df["distance"] = distances

# ✅ TOP10 表示
df_sorted = umap_df.sort_values("distance").head(10).reset_index(drop=True)

st.subheader("📋 近いワイン TOP10")
st.dataframe(df_sorted[["Type", "商品名", "distance"]])
