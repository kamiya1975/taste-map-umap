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

# ✅ PCA（3成分）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# ✅ 各軸の構成
PC1 = X_pca[:, 0]
PC2 = X_pca[:, 1]
PC3 = X_pca[:, 2]

df_clean["BodyAxis"] = PC1
df_clean["SweetAxis"] = PC2

# ✅ Typeごとの色設定
color_map = {
    "Spa": "blue", "White": "gold", "Red": "red", "Rose": "pink",
    "ロゼ": "pink", "スパークリング": "blue", "白": "gold", "赤": "red"
}

# ✅ スライダー（PC1, PC2）
st.subheader("基準のワインを飲んだ印象は？")
slider_pc2 = st.slider("←　こんなに甘みはいらない 　　　　　　　　　　　　 　　　　　　　　　　　　もう少し甘みがほしいな　→", 0, 100, 50)
slider_pc1 = st.slider("←　もう少し軽やかな感じがいいな 　　　　　　　　　　　　 　　　　　　もう少し濃厚なコクがほしいな　→", 0, 100, 50)

# ✅ PCA軸の min/max を取得
x_min, x_max = df_clean["BodyAxis"].min(), df_clean["BodyAxis"].max()
y_min, y_max = df_clean["SweetAxis"].min(), df_clean["SweetAxis"].max()

# ✅ スライダー値を軸スケールに変換
target_x = x_min + (slider_pc1 / 100) * (x_max - x_min)
target_y = y_min + (slider_pc2 / 100) * (y_max - y_min)

# ✅ 検索対象から "blendF" を除外
df_search = df_clean[df_clean["JAN"] != "blendF"].copy()

# ✅ 一致度計算
target_xy = np.array([[target_x, target_y]])
all_xy = df_search[["BodyAxis", "SweetAxis"]].values
distances = cdist(target_xy, all_xy).flatten()
df_search["distance"] = distances
df_sorted = df_search.sort_values("distance").head(10)

# ✅ 散布図
fig, ax = plt.subplots(figsize=(8, 8))

# Typeごとにプロット
for wine_type in df_clean["Type"].unique():
    mask = df_clean["Type"] == wine_type
    ax.scatter(
        df_clean.loc[mask, "BodyAxis"],
        df_clean.loc[mask, "SweetAxis"],
        label=wine_type,
        alpha=0.6,
        color=color_map.get(wine_type, "gray")
    )

# ✅ 一致度TOP10 ハイライト（順位ラベル表示 改良版）
for idx, (i, row) in enumerate(df_sorted.iterrows(), start=1):
    # 黒丸
    ax.scatter(
        row["BodyAxis"], row["SweetAxis"],
        color='black', edgecolor='white', s=240, marker='o'
    )
    # 順位番号 → 白文字・中央
    ax.text(
        row["BodyAxis"], row["SweetAxis"],  # 中央
        str(idx),
        fontsize=9,
        color='white',
        ha='center', va='center'
    )

# ✅ スライダー位置（ターゲット）マーク
ax.scatter(target_x, target_y, color='green', s=200, marker='X', label='point')

# ✅ 図の設定
ax.set_xlabel("Body")
ax.set_ylabel("Sweet")
ax.set_title("TasteMAP")
ax.legend(title="Type")
ax.grid(True)

# ✅ 軸目盛り（tick）非表示
ax.set_xticks([])
ax.set_yticks([])

# ✅ グラフ表示
st.pyplot(fig)

# ✅ 近いワイン TOP10 表示（静的テーブル版）
st.subheader("近いワイン TOP10")
df_sorted_display = df_sorted[["Type", "商品名", "希望小売価格"]].reset_index(drop=True)
df_sorted_display.index += 1

# ✅ Type列 → マーク付きに変換
type_markers = {
    "Red": "🔴",
    "White": "🟡",
    "Spa": "🔵",
    "Rose": "🟣",
}

# ✅ 希望小売価格 → 整形（例: 1,600 円）
df_sorted_display["希望小売価格"] = df_sorted_display["希望小売価格"].apply(lambda x: f"{int(x):,} 円")

# ✅ 静的表示に変更
st.table(df_sorted_display)


