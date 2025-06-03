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

# ✅ 特徴量データ処理
X = df[features]
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imputed)

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
st.title("TasteMAP UMAP + 等高線 + 一致度")

# ✅ UI選択
selected_feature = st.selectbox("等高線軸を選択", list(feature_components.keys()))
jan_input = st.text_input("一致度用 JANコード (空欄でもOK)", "")

# ✅ Z軸 合成
components = feature_components[selected_feature]
z_combined = df[components].sum(axis=1).values
umap_df["Z"] = z_combined

# ✅ プロット
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(
    x=umap_df["UMAP1"], y=umap_df["UMAP2"],
    hue=umap_df["Type"], palette="Set2",
    s=50, edgecolor='k', alpha=0.85, ax=ax
)

sns.kdeplot(
    x=umap_df["UMAP1"], y=umap_df["UMAP2"],
    weights=z_combined,
    cmap="YlOrBr", fill=True, bw_adjust=0.5, levels=100, alpha=0.5, ax=ax
)

ax.set_title(f"TasteMAP UMAP + 等高線: {selected_feature}", fontsize=16)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.grid(True)

st.pyplot(fig)

# ✅ 一致度計算（任意JAN指定時）
if jan_input != "":
    if jan_input not in umap_df["JAN"].values:
        st.warning(f"⚠️ JAN {jan_input} はデータに存在しません。")
    else:
        target_row = umap_df[umap_df["JAN"] == jan_input].iloc[0]
        target_xyz = np.array([[target_row["UMAP1"], target_row["UMAP2"], target_row["Z"]]])
        all_xyz = umap_df[["UMAP1", "UMAP2", "Z"]].values
        distances = cdist(target_xyz, all_xyz).flatten()
        umap_df["distance"] = distances
        df_sorted = umap_df.sort_values("distance").head(10)

        st.subheader(f"一致度TOP10（基準JAN: {jan_input}）")
        st.dataframe(df_sorted[["商品名", "JAN", "distance", "Z"]].rename(columns={"Z": f"{selected_feature}スコア"}))
