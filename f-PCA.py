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
matplotlib.rc('font', family='Arial Unicode MS')

# ✅ タイトルCSS
title_css = """
<style>
h1 {
    font-size: 32px !important;
    margin-bottom: 10px !important;
}
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
.css-1t42p7b .e1f1d6gn2 {
    display: none;
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
legend_order = ["Spa", "White", "Red", "Rose", "Entry Wine"]
color_map_fixed = {
    "Spa": "blue", "White": "gold", "Red": "red", "Rose": "pink", "Entry Wine": "green"
}

# ✅ Entry Wine の座標取得
blendF_row = df_clean[df_clean["JAN"] == "blendF"].iloc[0]
blendF_x = blendF_row["BodyAxis"]
blendF_y = blendF_row["SweetAxis"]

# ✅ スライダー
st.subheader("基準のワインを飲んだ印象は？")
slider_pc2 = st.slider("← こんなに甘みはいらない　　　　　　もう少し甘みがほしいな →", 0, 100, 50)
slider_pc1 = st.slider("← もう少し軽やかな感じがいいな　　　　もう少し濃厚なコクがほしいな →", 0, 100, 50)

# ✅ 軸スケール
x_min, x_max = df_clean["BodyAxis"].min(), df_clean["BodyAxis"].max()
y_min, y_max = df_clean["SweetAxis"].min(), df_clean["SweetAxis"].max()

# ✅ スライダー → MAP座標（blendF起点）
target_x = blendF_x + ((slider_pc1 - 50) / 50) * (x_max - x_min) / 2
target_y = blendF_y + ((slider_pc2 - 50) / 50) * (y_max - y_min) / 2

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

# Entry Wine 固定マーク
ax.scatter(blendF_x, blendF_y, color='green', s=200, marker='X', label='Entry Wine')

# スライダー位置マーク
ax.scatter(target_x, target_y, color='green', s=200, marker='X')

# 図設定
ax.set_xlabel("-  Body  +")
ax.set_ylabel("-  Sweet  +")
ax.set_title("TasteMAP")

# 凡例
handles, labels = ax.get_legend_handles_labels()
sorted_handles_labels = [
    (h, l) for l in legend_order for h, lbl in zip(handles, labels) if lbl == l
]
if sorted_handles_labels:
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    ax.legend(sorted_handles, sorted_labels, title="Type")

ax.grid(True)
ax.set_xticks([])
ax.set_yticks([])

# グラフ
st.pyplot(fig)

# ✅ TOP10 評価 UI
st.subheader("近いワイン TOP10（評価つき）")

if "user_ratings_dict" not in st.session_state:
    st.session_state.user_ratings_dict = {}

rating_options = ["未評価", "★", "★★", "★★★", "★★★★", "★★★★★"]

for idx, (i, row) in enumerate(df_sorted.iterrows(), start=1):
    jan = str(row["JAN"])
    label_text = f"{idx}. {row['商品名']} ({row['Type']}) {int(row['希望小売価格']):,} 円"

    current_rating = st.session_state.user_ratings_dict.get(jan, 0)
    current_index = current_rating if 0 <= current_rating <= 5 else 0

    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])

    with col1:
        st.markdown(f"**{label_text}**")

    with col2:
        selected_index = st.selectbox(
            " ", options=rating_options,
            index=current_index,
            key=f"rating_{jan}_selectbox"
        )
        new_rating = rating_options.index(selected_index)

    with col3:
        if st.button("反映", key=f"reflect_{jan}"):
            st.session_state.user_ratings_dict[jan] = new_rating
            st.rerun()

    st.markdown("---")
