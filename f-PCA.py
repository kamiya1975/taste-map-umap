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

# ✅ rcParams 初期化
matplotlib.rcdefaults()
matplotlib.rc('font', family='Arial Unicode MS')

# ✅ タイトルCSS
title_css = """
<style>
h1 { font-size: 32px !important; margin-bottom: 10px !important; }
</style>
"""
st.markdown(title_css, unsafe_allow_html=True)

# ✅ スライダー赤丸 CSS
slider_thumb_css = """
<style>
div[role="slider"] {
    height: 32px !important;
    width: 32px !important;
    background: red !important;
    border-radius: 50% !important;
    border: none !important;
    cursor: pointer !important;
}
</style>
"""
st.markdown(slider_thumb_css, unsafe_allow_html=True)

# ✅ スライダー数値 非表示
hide_slider_value_css = """
<style>
.stSlider > div > div > div > div > div {
    visibility: hidden;
}
</style>
"""
st.markdown(hide_slider_value_css, unsafe_allow_html=True)

# ✅ データ読み込み
df = pd.read_csv("Merged_TasteDataDB15.csv")

# ✅ 使用する成分
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

# ✅ 欠損除外
df_clean = df.dropna(subset=features + ["Type", "JAN", "商品名"]).reset_index(drop=True)
df_clean["JAN"] = df_clean["JAN"].astype(str)

# ✅ PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

PC1 = X_pca[:, 0]
PC2 = X_pca[:, 1]
PC3 = X_pca[:, 2]

甘味軸 = (PC2 + PC3) / np.sqrt(2)
複合ボディ軸 = (PC1 + 甘味軸) / np.sqrt(2)

df_clean["BodyAxis"] = 複合ボディ軸
df_clean["SweetAxis"] = 甘味軸

# ✅ 色設定＋凡例順
legend_order = ["Spa", "White", "Red", "Rose"]
color_map_fixed = {
    "Spa": "blue", "White": "gold", "Red": "red", "Rose": "pink"
}

# ✅ Entry Wine位置取得
blendF_row = df_clean[df_clean["JAN"] == "blendF"].iloc[0]
blendF_x = blendF_row["BodyAxis"]
blendF_y = blendF_row["SweetAxis"]

# ✅ スライダー
st.subheader("基準のワインを飲んだ印象は？")

# ★ blendF中心 → スライダーは50:50 → ±10範囲にする
x_range = 10
y_range = 10

slider_pc1 = st.slider(
    "← 軽やか　　　　濃厚 →", 0, 100, 50
)
slider_pc2 = st.slider(
    "← 甘さ控えめ　　　　甘さ強め →", 0, 100, 50
)

# スケーリング
target_x = blendF_x + ((slider_pc1 - 50) / 100) * x_range
target_y = blendF_y + ((slider_pc2 - 50) / 100) * y_range

# ✅ blendF 除外
df_search = df_clean[df_clean["JAN"] != "blendF"].copy()

# ✅ 一致度
target_xy = np.array([[target_x, target_y]])
all_xy = df_search[["BodyAxis", "SweetAxis"]].values
distances = cdist(target_xy, all_xy).flatten()
df_search["distance"] = distances
df_sorted = df_search.sort_values("distance").head(10)

# ✅ 散布図
fig, ax = plt.subplots(figsize=(8, 8))

# Type別ワイン打点
for wine_type in legend_order:
    mask = df_clean["Type"] == wine_type
    if mask.sum() > 0:
        ax.scatter(
            df_clean.loc[mask, "BodyAxis"],
            df_clean.loc[mask, "SweetAxis"],
            label=wine_type,
            alpha=0.6,
            color=color_map_fixed.get(wine_type, "gray"),
            s=20
        )

# TOP10 ハイライト
for idx, (i, row) in enumerate(df_sorted.iterrows(), start=1):
    ax.scatter(row["BodyAxis"], row["SweetAxis"],
               color='black', edgecolor='white', s=240, marker='o')
    ax.text(row["BodyAxis"], row["SweetAxis"], str(idx),
            fontsize=9, color='white', ha='center', va='center')

# ✅ Your Impression マーク (緑X)
ax.scatter(target_x, target_y, color='green', s=200, marker='X', label='Your Impression')

# ✅ legend
handles, labels = ax.get_legend_handles_labels()
sorted_handles_labels = [
    (h, l) for l in legend_order for h, lbl in zip(handles, labels) if lbl == l
]
# Add "Your Impression"
sorted_handles_labels.append((
    ax.scatter([], [], color='green', s=200, marker='X'), 'Your Impression'
))

sorted_handles, sorted_labels = zip(*sorted_handles_labels)
ax.legend(sorted_handles, sorted_labels, title="Type")

# ✅ Grid & Axes
ax.set_xlabel("-  Body  +")
ax.set_ylabel("-  Sweet  +")
ax.set_title("TasteMAP")
ax.grid(True)
ax.set_xticks([])
ax.set_yticks([])

# ✅ グラフ
st.pyplot(fig)
