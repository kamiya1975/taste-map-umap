# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# ✅ 日本語フォント
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

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
umap_df["JAN"] = df["JAN"].astype(str)
umap_df["Type"] = df["Type"] if "Type" in df.columns else "Unknown"
umap_df["商品名"] = df["商品名"] if "商品名" in df.columns else umap_df["JAN"]

# ✅ Streamlit UI
st.title("UMAP + seaborn.kdeplot版 等高線（改良版）")

# ✅ 等高線 軸選択
selected_feature = st.selectbox("等高線軸を選択", list(feature_components.keys()))

# ✅ Z軸 → 1変数 or 平均
components = feature_components[selected_feature]
if len(components) == 1:
    z_var = components[0]
    umap_df["Z"] = df[z_var].astype(float)
else:
    umap_df["Z"] = df[components].mean(axis=1).values

# ✅ Plot（Jupyter版と同じ kdeplot 使用） + 改良反映
fig, ax = plt.subplots(figsize=(10, 8))

# --- 等高線（色合い強調版）
sns.kdeplot(
    x=umap_df["UMAP1"], y=umap_df["UMAP2"],
    weights=umap_df["Z"],
    fill=True, cmap="YlOrBr", levels=50, alpha=0.5, bw_adjust=0.5,
    ax=ax
)

# --- ワインの打点（小さめに調整）
sns.scatterplot(
    data=umap_df, x="UMAP1", y="UMAP2",
    hue="Type", palette="Set2", s=20, edgecolor='k', alpha=0.85,
    ax=ax
)

# --- 基準ワイン（商品名＝blendF）を特別表示
blendF_df = umap_df[umap_df["商品名"] == "blendF"]

ax.scatter(
    blendF_df["UMAP1"], blendF_df["UMAP2"],
    c='red', s=100, edgecolor='black', linewidth=1.5,
    label="基準ワイン (blendF)", zorder=5
)

# --- レイアウト調整 ---
ax.set_title(f"UMAP + 等高線: {selected_feature}", fontsize=14)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(title="Type", loc='upper right')
ax.grid(True)

# ✅ グラフ表示
st.pyplot(fig)
