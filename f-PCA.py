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

# ✅ フォント fallback
matplotlib.rc('font', family='Arial Unicode MS')

# ✅ session_state に評価用 dict 初期化
if "user_ratings_dict" not in st.session_state:
    st.session_state.user_ratings_dict = {}

# ✅ タイトル
st.title("TasteMAP（複合PCA軸版）＋ユーザー評価テスト")

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

# 一致度TOP10 ハイライト
for idx, (i, row) in enumerate(df_sorted.iterrows(), start=1):
    ax.scatter(row["BodyAxis"], row["SweetAxis"],
               color='black', edgecolor='white', s=240, marker='o')
    ax.text(row["BodyAxis"], row["SweetAxis"], str(idx),
            fontsize=9, color='white', ha='center', va='center')

# ターゲット位置
ax.scatter(target_x, target_y, color='green', s=200, marker='X', label='point')

# ✅ バブルチャート重ね
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
            s=row["rating"] * 80,
            color='orange', alpha=0.5, edgecolor='black', linewidth=1.5,
            label='User Rating' if i == 0 else ""
        )
    st.info(f"🎈 現在 {len(df_ratings_input)} 件の評価が登録されています")

# 図設定
ax.set_xlabel("Body")
ax.set_ylabel("Sweet")
ax.set_title("TasteMAP")
ax.legend(title="Type")
ax.grid(True)
ax.set_xticks([])
ax.set_yticks([])

# グラフ表示
st.pyplot(fig)

# ✅ TOP10 表示＋評価フォーム ← 最後に配置
st.subheader("近いワイン TOP10（評価つき）")

with st.form("rating_form"):
    for idx, (i, row) in enumerate(df_sorted.iterrows(), start=1):
        jan = str(row["JAN"])
        label = f"{idx}. {row['商品名']} ({row['Type']}) {int(row['希望小売価格']):,} 円"
        
        default_rating = st.session_state.user_ratings_dict.get(jan, 0)
        
        rating = st.selectbox(
            label,
            options=[0, 1, 2, 3, 4, 5],
            index=default_rating,
            key=f"rating_{jan}"
        )
        
        st.session_state.user_ratings_dict[jan] = rating
    
    submitted = st.form_submit_button("評価を反映する")
