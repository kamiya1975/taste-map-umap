# ✅ 必要なライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

# ✅ rcParams 初期化
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

# ✅ DeckGL 用データ準備
df_deck = df_clean.copy()
df_deck["x"] = df_deck["BodyAxis"]
df_deck["y"] = df_deck["SweetAxis"]

# ✅ ① 中心座標を 0,0 にシフト
df_deck["x_shift"] = df_deck["x"] - (x_min + x_max) / 2
df_deck["y_shift"] = df_deck["y"] - (y_min + y_max) / 2

# ✅ ② スケーリング係数をかけて Deck に適合させる
scale_factor = 100
df_deck["x_scaled"] = df_deck["x_shift"] * scale_factor
df_deck["y_scaled"] = df_deck["y_shift"] * scale_factor

# ✅ ③ Deck 用 カラー変換 → RGB
type_color_rgb = {
    "Spa": [0, 0, 255, 180],
    "White": [255, 215, 0, 180],
    "Red": [255, 0, 0, 180],
    "Rose": [255, 105, 180, 180],
    "Entry Wine": [0, 255, 0, 180],
}

df_deck["color"] = df_deck["Type"].map(type_color_rgb).apply(lambda x: x if x is not None else [100, 100, 100, 180])

# ✅ ④ Scatterplot Layer（全体）
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_deck,
    get_position=["x_scaled", "y_scaled"],
    get_fill_color="color",
    get_radius=100,
    pickable=True,
    auto_highlight=True
)

# ✅ ViewState（PCA座標用に変更！）
view_state = pdk.ViewState(
    target=[0, 0],  # ← PCAの中心
    zoom=0,         # ← 適正ズーム
    min_zoom=-5,
    max_zoom=20,
    bearing=0,
    pitch=0
)

# ✅ ⑤ Target 緑丸（ユーザー印象）
target_df = pd.DataFrame({
    "x_scaled": [(target_x - (x_min + x_max) / 2) * scale_factor],
    "y_scaled": [(target_y - (y_min + y_max) / 2) * scale_factor],
    "color": [[0, 255, 0, 255]],  # 緑・不透明
    "label": ["Your Impression"]
})

target_layer = pdk.Layer(
    "ScatterplotLayer",
    data=target_df,
    get_position=["x_scaled", "y_scaled"],
    get_fill_color="color",
    get_radius=150,
    pickable=False,
    auto_highlight=False
)

# ✅ ⑥ TOP10 黒丸
top10_df = df_sorted.copy()
top10_df["x_scaled"] = (top10_df["BodyAxis"] - (x_min + x_max) / 2) * scale_factor
top10_df["y_scaled"] = (top10_df["SweetAxis"] - (y_min + y_max) / 2) * scale_factor
top10_df["color"] = [[0, 0, 0, 255]] * len(top10_df)  # 黒・不透明

top10_layer = pdk.Layer(
    "ScatterplotLayer",
    data=top10_df,
    get_position=["x_scaled", "y_scaled"],
    get_fill_color="color",
    get_radius=200,
    pickable=True
)

# ✅ Deck 作成 → map_style=None（背景白）
deck_map = pdk.Deck(
    layers=[scatter_layer, target_layer, top10_layer],
    initial_view_state=view_state,
    map_style=None,
    tooltip={"text": "{商品名} ({Type})"}
)

# ✅ Deck 表示
st.pydeck_chart(deck_map)

# ✅ Legend
st.markdown("### Type Legend")
for t, color in type_color_rgb.items():
    rgba_css = f"rgba({color[0]}, {color[1]}, {color[2]}, {color[3]/255})"
    st.markdown(f'<div style="display:inline-block;width:20px;height:20px;background:{rgba_css};margin-right:10px;"></div> {t}', unsafe_allow_html=True)
