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
st.title("TasteMAP UMAP ＋ 等高線 ＋ 一致度")

# 等高線選択
selected_feature = st.selectbox("等高線軸を選択", list(feature_components.keys()))

# JANコード入力
jan_input = st.text_input("一致度用 JANコード（空欄でもOK）")

# ✅ Z軸 合成
components = feature_components[selected_feature]
z_combined = df[components].sum(axis=1).values
umap_df["Z"] = z_combined

# ✅ Plotly で散布図
fig = px.scatter(
    umap_df,
    x="UMAP1", y="UMAP2",
    color="Type",
    hover_data=["商品名", "JAN"],
    size_max=15
)

# ✅ 等高線（density heatmap overlay）
fig.update_traces(marker=dict(size=8, opacity=0.8))
fig.add_trace(
    px.density_heatmap(
        umap_df,
        x="UMAP1", y="UMAP2",
        z="Z",
        nbinsx=50, nbinsy=50,
        color_continuous_scale="YlOrBr"
    ).data[0]
)

st.plotly_chart(fig, use_container_width=True)

# ✅ 一致度（JAN 入力時）
if jan_input != "":
    if jan_input not in umap_df["JAN"].values:
        st.warning(f"JAN {jan_input} はデータに存在しません。")
    else:
        target_row = umap_df[umap_df["JAN"] == jan_input].iloc[0]
        target_xyz = np.array([[target_row["UMAP1"], target_row["UMAP2"], target_row["Z"]]])
        all_xyz = umap_df[["UMAP1", "UMAP2", "Z"]].values
        distances = cdist(target_xyz, all_xyz).flatten()
        umap_df["distance"] = distances
        df_sorted = umap_df.sort_values("distance").head(10)
        st.subheader(f"一致度 TOP10 （基準JAN: {jan_input}）")
        st.dataframe(df_sorted[["Type", "商品名", "JAN", "distance"]])

# ✅ マップクリック対応
st.subheader("🔍 マップ上をクリックすると、近いワイン10本を表示")

# Plotly のクリックイベントを取得
click = st.plotly_events(fig, click_event=True, select_event=False)

if click:
    clicked_x = click[0]["x"]
    clicked_y = click[0]["y"]
    st.write(f"クリック位置: ({clicked_x:.2f}, {clicked_y:.2f})")

    target_point = np.array([[clicked_x, clicked_y]])
    all_xy = umap_df[["UMAP1", "UMAP2"]].values
    distances = cdist(target_point, all_xy).flatten()
    umap_df["click_distance"] = distances

    df_sorted_click = umap_df.sort_values("click_distance").head(10)
    st.dataframe(df_sorted_click[["Type", "商品名", "JAN", "click_distance"]])

