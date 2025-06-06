# ✅ 必要なライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# ✅ rcParams を初期化
matplotlib.rcdefaults()
matplotlib.rc('font', family='Arial Unicode MS')

# ✅ タイトル CSS
title_css = """
<style>
h1 {
    font-size: 32px !important;
    margin-bottom: 10px !important;
}
</style>
"""
st.markdown(title_css, unsafe_allow_html=True)

# ✅ スライダー赤丸（もっと大きく！）
slider_thumb_css = """
<style>
div[role="slider"] {
    height: 30px !important;
    width: 30px !important;
    background: red !important;
    border-radius: 50% !important;
    border: none !important;
    cursor: pointer !important;
}
</style>
"""
st.markdown(slider_thumb_css, unsafe_allow_html=True)

# ✅ スライダー数値「50」非表示
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

# ✅ 使用成分
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

# ✅ 色設定
color_map = {
    "Spa": "blue", "White": "gold", "Red": "red", "Rose": "pink", "Entry Wine": "green"
}
legend_order = ["Spa", "White", "Red", "Rose", "Entry Wine"]

# ✅ blendF の位置取得
blendF_row = df_clean[df_clean["JAN"] == "blendF"].iloc[0]
blendF_x = blendF_row["BodyAxis"]
blendF_y = blendF_row["SweetAxis"]

# ✅ 軸の min/max
x_min, x_max = df_clean["BodyAxis"].min(), df_clean["BodyAxis"].max()
y_min, y_max = df_clean["SweetAxis"].min(), df_clean["SweetAxis"].max()

# ✅ Entry Wine からの距離
range_left_x  = blendF_x - x_min
range_right_x = x_max - blendF_x
range_down_y  = blendF_y - y_min
range_up_y    = y_max - blendF_y

# ✅ スライダー
st.subheader("基準のワインを飲んだ印象は？")
slider_pc2 = st.slider("← こんなに甘みはいらない　　　　　　もう少し甘みがほしいな →", 0, 100, 50)
slider_pc1 = st.slider("← もう少し軽やかな感じがいいな　　　　もう少し濃厚なコクがほしいな →", 0, 100, 50)

# ✅ スライダー → MAP座標変換
# BodyAxis
if slider_pc1 <= 50:
    target_x = blendF_x - ((50 - slider_pc1) / 50) * range_left_x
else:
    target_x = blendF_x + ((slider_pc1 - 50) / 50) * range_right_x

# SweetAxis
if slider_pc2 <= 50:
    target_y = blendF_y - ((50 - slider_pc2) / 50) * range_down_y
else:
    target_y = blendF_y + ((slider_pc2 - 50) / 50) * range_up_y

# ✅ blendF 除外
df_search = df_clean[df_clean["JAN"] != "blendF"].copy()

# ✅ 一致度計算
target_xy = np.array([[target_x, target_y]])
all_xy = df_search[["BodyAxis", "SweetAxis"]].values
distances = cdist(target_xy, all_xy).flatten()
df_search["distance"] = distances
df_sorted = df_search.sort_values("distance").head(10)

# ✅ 散布図
fig, ax = plt.subplots(figsize=(8, 8))

# ワイン点
for wine_type in legend_order:
    mask = df_clean["Type"] == wine_type
    if mask.sum() > 0:
        ax.scatter(
            df_clean.loc[mask, "BodyAxis"],
            df_clean.loc[mask, "SweetAxis"],
            label=wine_type,
            alpha=0.6,
            color=color_map.get(wine_type, "gray"),
            s=20
        )

# ✅ TOP10 ハイライト
for idx, (i, row) in enumerate(df_sorted.iterrows(), start=1):
    ax.scatter(row["BodyAxis"], row["SweetAxis"],
               color='black', edgecolor='white', s=240, marker='o')
    ax.text(row["BodyAxis"], row["SweetAxis"], str(idx),
            fontsize=9, color='white', ha='center', va='center')

# ✅ ユーザー印象 (緑X)
ax.scatter(target_x, target_y, color='green', s=200, marker='X', label='Your Impression')

# ✅ 凡例
handles, labels = ax.get_legend_handles_labels()
sorted_handles_labels = [
    (h, l) for l in legend_order for h, lbl in zip(handles, labels) if lbl == l
]
# + User Impression
sorted_handles_labels.append((ax.scatter([], [], color='green', s=200, marker='X'), 'Your Impression'))

if sorted_handles_labels:
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    ax.legend(sorted_handles, sorted_labels, title="Type")

# ✅ 軸設定
ax.set_xlabel("-  Body  +")
ax.set_ylabel("-  Sweet  +")
ax.set_title("TasteMAP")
ax.grid(True)
ax.set_xticks([])
ax.set_yticks([])

# ✅ 表示
st.pyplot(fig)

st.subheader("近いワイン TOP10（評価つき）")

# user_ratings_dict の初期化（もしなければ）
if "user_ratings_dict" not in st.session_state:
    st.session_state.user_ratings_dict = {}

# ★評価 options
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
