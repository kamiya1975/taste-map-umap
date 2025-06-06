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

# ✅ タイトルCSS（完全版）
title_css = """
<style>
/* Streamlitのタイトル（emotionクラス対応） */
h1 {
    font-size: 32px !important;
    margin-bottom: 10px !important;
}
</style>
"""
st.markdown(title_css, unsafe_allow_html=True)

# ✅ タイトル
# st.title("TasteMAPテスト画面")

# ✅ スライダー赤丸 完全対応版
slider_thumb_css = """
<style>
/* Streamlitのスライダーは div[role="slider"] を使う */
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

# ✅ スライダー数値非表示CSS
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

# ✅ JAN を str に揃える
df_clean["JAN"] = df_clean["JAN"].astype(str)

# ✅ PCA（3成分 → 複合軸）
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

# ✅ Typeごとの色設定
color_map = {
    "Spa": "blue", "White": "gold", "Red": "red", "Rose": "pink",
    "ロゼ": "pink", "スパークリング": "blue", "白": "gold", "赤": "red"
}

# ✅ スライダー（Body, Sweet） ← 先頭
st.subheader("基準のワインを飲んだ印象は？")
slider_pc2 = st.slider("← こんなに甘みはいらない　　　　　　もう少し甘みがほしいな →", 0, 100, 50)
slider_pc1 = st.slider("← もう少し軽やかな感じがいいな　　　　もう少し濃厚なコクがほしいな →", 0, 100, 50)

# ✅ 軸スケール変換
x_min, x_max = df_clean["BodyAxis"].min(), df_clean["BodyAxis"].max()
y_min, y_max = df_clean["SweetAxis"].min(), df_clean["SweetAxis"].max()

target_x = x_min + (slider_pc1 / 100) * (x_max - x_min)
target_y = y_min + (slider_pc2 / 100) * (y_max - y_min)

# ✅ blendF 除外
df_search = df_clean[df_clean["JAN"] != "blendF"].copy()

# ✅ 一致度計算
target_xy = np.array([[target_x, target_y]])
all_xy = df_search[["BodyAxis", "SweetAxis"]].values
distances = cdist(target_xy, all_xy).flatten()
df_search["distance"] = distances
df_sorted = df_search.sort_values("distance").head(10)

# ✅ 散布図 ← ここをスライダーの次に
fig, ax = plt.subplots(figsize=(8, 8))

# ✅ 凡例に出す順番を固定
legend_order = ["Spa", "White", "Red", "Rose", "Entry Wine"]
color_map_fixed = {
    "Spa": "blue", "White": "gold", "Red": "red", "Rose": "pink", "Entry Wine": "green"
}

# ✅ プロット（ワイン打点サイズ 1/2 → s=20）
for wine_type in legend_order:
    mask = df_clean["Type"] == wine_type
    if mask.sum() > 0:  # 存在する場合のみ
        ax.scatter(
            df_clean.loc[mask, "BodyAxis"],
            df_clean.loc[mask, "SweetAxis"],
            label=wine_type,
            alpha=0.6,
            color=color_map_fixed.get(wine_type, "gray"),
            s=20  # ★ 小さく
        )

# ✅ 一致度TOP10 ハイライト
for idx, (i, row) in enumerate(df_sorted.iterrows(), start=1):
    ax.scatter(row["BodyAxis"], row["SweetAxis"],
               color='black', edgecolor='white', s=240, marker='o')
    ax.text(row["BodyAxis"], row["SweetAxis"], str(idx),
            fontsize=9, color='white', ha='center', va='center')

# ✅ ターゲット位置
ax.scatter(target_x, target_y, color='green', s=200, marker='X', label='point')

# ✅ バブルチャート重ね（サイズ2倍 → *160）
df_ratings_input = pd.DataFrame([
    {"JAN": jan, "rating": rating}
    for jan, rating in st.session_state.user_ratings_dict.items()
    if rating > 0
])

if not df_ratings_input.empty:
    df_plot = df_clean.merge(df_ratings_input, on="JAN", how="inner")
    
    for i, row in df_plot.iterrows():
        ax.scatter(
            row["BodyAxis"], row["SweetAxis"],
            s=row["rating"] * 160,  # ★ 2倍に変更
            color='orange', alpha=0.5, edgecolor='black', linewidth=1.5
        )
    st.info(f"🎈 現在 {len(df_ratings_input)} 件の評価が登録されています")

# ✅ 図設定
ax.set_xlabel("Body")
ax.set_ylabel("Sweet")
ax.set_title("TasteMAP")

# ✅ 凡例は固定順（User Rating は出さない）
handles, labels = ax.get_legend_handles_labels()

# 並び替え
sorted_handles_labels = [
    (h, l) for l in legend_order for h, lbl in zip(handles, labels) if lbl == l
]

# 描画
if sorted_handles_labels:
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    ax.legend(sorted_handles, sorted_labels, title="Type")

ax.grid(True)
ax.set_xticks([])
ax.set_yticks([])

# ✅ グラフ表示
st.pyplot(fig)


import streamlit as st

# ✅ TOP10 表示＋商品ごと反映ボタン版（完全版・1回でOK）
st.subheader("近いワイン TOP10（評価つき）")

# ▶️ 「ユーザー評価 dict」がなければ初期化
if "user_ratings_dict" not in st.session_state:
    st.session_state.user_ratings_dict = {}

for idx, (i, row) in enumerate(df_sorted.iterrows(), start=1):
    jan = str(row["JAN"])
    
    # ▶️ 表示ラベル
    label_text = f"{idx}. {row['商品名']} ({row['Type']}) {int(row['希望小売価格']):,} 円"
    
    # ▶️ 既存の評価
    default_rating = st.session_state.user_ratings_dict.get(jan, 0)
    
    # ▶️ 3カラム → [商品名] [selectbox] [button]
    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])  # 調整可
    
    with col1:
        st.markdown(f"**{label_text}**")
    
    with col2:
        rating = st.selectbox(
            " ",  # ← ラベル無し
            options=[0, 1, 2, 3, 4, 5],
            index=default_rating,
            key=f"rating_{jan}_selectbox"
        )
    
    with col3:
        if st.button("反映", key=f"reflect_{jan}"):
            st.session_state.user_ratings_dict[jan] = rating
            st.rerun()

    # 区切り線
    st.markdown("---")
