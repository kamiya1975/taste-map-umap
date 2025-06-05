# ✅ 必要なライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
import os
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# ✅ rcParams を初期化
matplotlib.rcdefaults()

# ✅ フォント fallback をグローバル設定（GitHubでも安全）
matplotlib.rc('font', family='Arial Unicode MS')

# ✅ Streamlit タイトル
st.title("TasteMAPテスト画面")

# ✅ データ読み込み
df = pd.read_csv("Merged_TasteDataDB15.csv")

# ✅ PCA対象の特徴量
features = [
    "20mm_L*", "20mm_a*", "20mm_b*", "Ethyl acetate", "Propanol",
    "2-Methylbutyl acetate", "Ethyl lactate", "Acetic acid", "Furfural",
    "2-Acetylfuran", "2-Nonanol", "Propanoic acid", "2,3-Butanediol isomer-1",
    "Isobutyric acid", "2,3-Butanediol isomer-2", "gamma-Butyrolactone",
    "Butyric acid", "Furfuryl alcohol", "Isovaleric acid", "2-Methyl butyric acid",
    "Diethyl succinate", "Methionol", "Cyclotene", "2-Phenylethyl acetate",
    "3-Mercapto-1-hexanol", "Capronic acid", "Guaiacol", "1-Undecanol",
    "Benzyl alcohol", "2-Phenylethanol", "Maltol", "2-Acetylpyrrole"
]

# ✅ 前処理
df_pca = df.dropna(subset=features)
X = df_pca[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ✅ スライダー（PC1, PC2）
st.subheader("基準のワインを飲んだ印象は？")
slider_pc1 = st.slider("←もう少し軽やかな感じがいいな 　　　　　　　　　　　　 　　　　　　　　　もう少し濃厚なコクがほしいな→", 0, 100, 50)
slider_pc2 = st.slider("←こんなに甘みはいらない 　　　　　　　　　　　　 　　もう少し甘みがほしいな→", 0, 100, 50)

# ✅ スライダー値 → PCA空間に変換（中心50→0）
slider_pc1_val = (slider_pc1 - 50) / 10
slider_pc2_val = (slider_pc2 - 50) / 10
target_point = np.array([[slider_pc1_val, slider_pc2_val]])

# ✅ 距離計算（ユークリッド距離）
distances = cdist(target_point, X_pca)[0]
df_pca["Distance"] = distances
df_top10 = df_pca.nsmallest(10, "Distance")

# ✅ 散布図
fig, ax = plt.subplots(figsize=(8, 8))

color_map = {
    "White": "gold",
    "Red": "red",
    "Rose": "pink",
    "Spa": "blue"
}

for wine_type in df_pca["Type"].unique():
    mask = df_pca["Type"] == wine_type
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=wine_type,
        color=color_map.get(wine_type, "gray"),
        alpha=0.6
    )

# ✅ 基準ワイン位置
ax.scatter(
    slider_pc1_val,
    slider_pc2_val,
    color="black",
    s=200,
    edgecolor="white",
    linewidth=2,
    label="基準ワイン"
)

# ✅ ラベル・凡例
ax.set_xlabel("🟥 PCA軸1（軽やか ←→ 濃厚）")
ax.set_ylabel("🟦 PCA軸2（甘さ控えめ ←→ 甘さ強め）")
ax.legend()
ax.grid(True)

# ✅ 表示
st.pyplot(fig)

# ✅ 近いワイン TOP10 表示
st.subheader("📋 近いワイン TOP10")
st.dataframe(df_top10[["JAN", "ProductName", "Type", "Distance"]].reset_index(drop=True))
