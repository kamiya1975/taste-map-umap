# ✅ 必要なライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

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

# ✅ Plotly 用ライブラリ追加
import plotly.express as px
import plotly.graph_objects as go

# ✅ Plotly 用データ準備
plot_df = df_clean.copy()

# サイズと色列を追加
plot_df["size"] = 20
plot_df["color"] = plot_df["Type"].map(color_map).fillna("gray")

# ✅ TOP10 → 特大黒丸
plot_df.loc[plot_df["JAN"].isin(df_sorted["JAN"]), "size"] = 40
plot_df.loc[plot_df["JAN"].isin(df_sorted["JAN"]), "color"] = "black"

# ✅ ユーザー印象（Your Impression） → 別 DataFrame
impression_df = pd.DataFrame({
    "BodyAxis": [target_x],
    "SweetAxis": [target_y],
    "Type": ["Your Impression"],
    "size": [50],
    "color": ["green"],
    "商品名": ["Your Impression"]
})

# ✅ Base scatter (全体)
fig = px.scatter(
    plot_df,
    x="BodyAxis",
    y="SweetAxis",
    color="Type",
    color_discrete_map=color_map,
    size="size",
    hover_data=["商品名", "JAN", "Type"]
)

# ✅ ユーザー印象 (Xマーク) を追加
fig.add_trace(go.Scatter(
    x=impression_df["BodyAxis"],
    y=impression_df["SweetAxis"],
    mode="markers+text",
    marker=dict(size=50, color="green", symbol="x"),
    text=["Your Impression"],
    textposition="top center",
    name="Your Impression"
))

# ✅ バブルチャート（ユーザー評価） ← ⭐️ ⭐️ ⭐️ ⭐️ ⭐️
if "user_ratings_dict" in st.session_state:
    df_ratings_input = pd.DataFrame([
        {"JAN": jan, "rating": rating}
        for jan, rating in st.session_state.user_ratings_dict.items()
        if rating > 0
    ])

    if not df_ratings_input.empty:
        df_plot_ratings = df_clean.merge(df_ratings_input, on="JAN", how="inner")
        
        fig.add_trace(go.Scatter(
            x=df_plot_ratings["BodyAxis"],
            y=df_plot_ratings["SweetAxis"],
            mode="markers",
            marker=dict(
                size=df_plot_ratings["rating"] * 16,
                color="orange",
                opacity=0.5,
                line=dict(width=1.5, color="black")
            ),
            text=df_plot_ratings["商品名"],
            name="Your Ratings 🎈"
        ))

        st.info(f"🎈 現在 {len(df_ratings_input)} 件の評価が登録されています")

# ✅ レイアウト整備（dragmode=pan + legend横並び + 背景グレー）
fig.update_layout(
    title="TasteMAP (PCA複合軸版 Interactive)",
    xaxis_title="- Body +（PC1 + 甘味軸）",
    yaxis_title="- Sweet +（PC2 + PC3）",
    showlegend=True,
    width=800,
    height=800,
    plot_bgcolor="rgba(245,245,245,1)",
    paper_bgcolor="rgba(245,245,245,1)",
    dragmode="pan",

    # ✅ 凡例（legend）を外に出す（下に横並び）
    legend=dict(
        orientation="h",
        x=0,
        y=-0.15,  # ← y=-0.1〜-0.15 がスマホ/PC両方でバランス良い
        bordercolor="black",
        borderwidth=0.5,
        bgcolor="rgba(255,255,255,0.8)"
    )
)

# ✅ 軸の設定（目盛り復活＋ゼロ線＋グリッド＋ズーム固定）
x_range_margin = (x_max - x_min) * 0.1
y_range_margin = (y_max - y_min) * 0.1

fig.update_xaxes(
    title_text="- Body +（PC1 + 甘味軸）",
    showticklabels=True,
    zeroline=True,
    zerolinewidth=2,
    zerolinecolor='black',
    gridcolor='lightgray',
    range=[x_min - x_range_margin, x_max + x_range_margin]
)

fig.update_yaxes(
    title_text="- Sweet +（PC2 + PC3）",
    showticklabels=True,
    zeroline=True,
    zerolinewidth=2,
    zerolinecolor='black',
    gridcolor='lightgray',
    range=[y_min - y_range_margin, y_max + y_range_margin]
)

# ✅ 最終表示（インタラクティブ！）→ scrollZoom 有効化 + responsive + key 追加
st.plotly_chart(
    fig,
    use_container_width=True,
    config={"scrollZoom": True, "responsive": True, "doubleClick": "reset"},
    key="pca_plot"
)

# ✅ TOP10（評価つき）
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

# ✅ 必要ライブラリ
import pydeck as pdk

# ✅ DeckGL 用データ準備（PCA複合軸）
df_deck = df_clean.copy()
df_deck["x"] = df_deck["BodyAxis"]
df_deck["y"] = df_deck["SweetAxis"]

# ✅ Scatterplot Layer（シンプル版 → 背景真っ白）
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_deck,
    get_position="[x, y]",
    get_fill_color="[0, 128, 255, 160]",  # 青
    get_radius=50,
    pickable=True,
    auto_highlight=True
)

# ✅ Viewport セッティング
x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2
zoom_level = 2

view_state = pdk.ViewState(
    longitude=x_center,
    latitude=y_center,
    zoom=zoom_level,
    min_zoom=1,
    max_zoom=10,
    bearing=0,
    pitch=0
)

# --- DeckGL 部分は今はコメントアウトする！ ---

# deck_map = pdk.Deck(
#     layers=[scatter_layer],
#     initial_view_state=view_state,
#     map_style=None,
#     parameters={"projection": "ORTHOGRAPHIC"},
#     controller=True
# )

# st.pydeck_chart(deck_map)


# ✅ 必要ライブラリ
import pydeck as pdk

# ✅ DeckGL 用データ準備（PCA複合軸）
df_deck = df_clean.copy()
df_deck["x"] = df_deck["BodyAxis"]
df_deck["y"] = df_deck["SweetAxis"]

# ✅ Deck 用 カラー変換 → RGB (0-255)
type_color_rgb = {
    "Spa": [0, 0, 255, 180],         # 青
    "White": [255, 215, 0, 180],     # ゴールド
    "Red": [255, 0, 0, 180],         # 赤
    "Rose": [255, 105, 180, 180],    # ピンク
    "Entry Wine": [0, 255, 0, 180],  # 緑
}

# 各行に RGB カラー列を追加
df_deck["color"] = df_deck["Type"].map(type_color_rgb).apply(lambda x: x if x is not None else [100, 100, 100, 180])

# ✅ Scatterplot Layer（背景真っ白 / XY空間！）
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_deck,
    get_position=["x", "y"],   # XY空間
    get_fill_color="color",    # RGB
    get_radius=50,
    pickable=True,
    auto_highlight=True
)

# ✅ XY空間として ViewState 調整（地図にしない）
# → x/y の中心 & range を Deck 側に設定
view_state = pdk.ViewState(
    longitude=0,  # ダミー → 実際は XY空間
    latitude=0,
    zoom=0,       # zoom=0 → Deck 側は scale に依存
    min_zoom=-5,
    max_zoom=20,
    bearing=0,
    pitch=0,
    target=[(x_min + x_max) / 2, (y_min + y_max) / 2]  # 中心 XY座標
)

# ✅ Deck 作成（背景白にする！）
deck_map = pdk.Deck(
    layers=[scatter_layer],
    initial_view_state=view_state,
    map_style=None,  # 背景真っ白
    tooltip={"text": "{商品名} ({Type})"}
)

# ✅ Deck 表示
st.pydeck_chart(deck_map)


# ✅ 仮の Legend を Streamlit 側に出す
st.markdown("### Type Legend")
for t, color in color_map_rgba.items():
    rgba_css = f"rgba({color[0]}, {color[1]}, {color[2]}, {color[3]/255})"
    st.markdown(f'<div style="display:inline-block;width:20px;height:20px;background:{rgba_css};margin-right:10px;"></div> {t}', unsafe_allow_html=True)

