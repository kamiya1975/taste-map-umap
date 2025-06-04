# ✅ 必要なライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# ✅ rcParams を初期化
matplotlib.rcdefaults()

# ✅ フォント設定（Streamlit対応）
import matplotlib.font_manager as fm
font_prop = fm.FontProperties(family='IPAexGothic')  # 必要なら 'Arial' や 'Noto Sans CJK JP'
# print("✅ 使用フォント →", font_prop.get_name()) ← コメントアウト！


# ✅ データ読み込み
csv_path = 'Merged_TasteDataDB15.csv'
df = pd.read_csv(csv_path)


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
甘味軸 = (PC2 + PC3) / np.sqrt(2)
複合ボディ軸 = (PC1 + 甘味軸) / np.sqrt(2)

# ✅ DataFrameに軸追加
df_clean["BodyAxis"] = 複合ボディ軸
df_clean["SweetAxis"] = 甘味軸

# ✅ Typeごとの色設定
color_map = {
    "Spa": "blue", "White": "gold", "Red": "red", "Rose": "pink",
    "ロゼ": "pink", "スパークリング": "blue", "白": "gold", "赤": "red"
}

# ======================================
# ✅ Streamlit アプリ部
# ======================================

st.title("🎈 TasteMAP：PCA合成軸マップ with スライダー一致度")

# ✅ スライダー（PC1, PC2）
st.markdown("#### 🔍 基準ワインの印象調整（スライダー）")
slider_pc1 = st.slider("← PC1（軽やか） / PC1（濃厚） →", 0, 100, 50)
slider_pc2 = st.slider("← PC2（甘さ控えめ） / PC2（甘さ強め） →", 0, 100, 50)

# ✅ スライダー → マップ座標（正規化して -3 ～ +3 スケール）
# PCAのスケールを約±3想定（実データ確認しながら微調整可）
target_x = (slider_pc1 - 50) / 50 * 3
target_y = (slider_pc2 - 50) / 50 * 3

# ✅ 散布図表示
fig, ax = plt.subplots(figsize=(16, 12))
for wine_type in df_clean["Type"].unique():
    mask = df_clean["Type"] == wine_type
    ax.scatter(
        df_clean.loc[mask, "BodyAxis"],
        df_clean.loc[mask, "SweetAxis"],
        label=wine_type,
        alpha=0.6,
        color=color_map.get(wine_type, "gray")
    )

# ✅ JANコードハイライト
for i, row in df_clean.iterrows():
    if str(row["JAN"]) in target_jans:
        ax.scatter(
            row["BodyAxis"], row["SweetAxis"],
            color='black', edgecolor='white', s=100, marker='o'
        )
        ax.text(
            row["BodyAxis"] + 0.1, row["SweetAxis"],
            str(row["商品名"]),
            fontsize=8, color='black', fontproperties=font_prop
        )

# ✅ スライダー位置（ターゲット）マーク
ax.scatter(target_x, target_y, color='green', s=200, marker='X', label='基準スライダー位置')

# ✅ 図の設定
ax.set_xlabel("複合ボディ軸（PC1 & 甘味軸）", fontproperties=font_prop)
ax.set_ylabel("甘味軸（PC2 + PC3）", fontproperties=font_prop)
ax.set_title("散布図②：複合ボディ軸 vs 甘味軸", fontproperties=font_prop)
ax.legend(title="Type", prop=font_prop)
ax.grid(True)

# ✅ 表示
st.pyplot(fig)

# ✅ 一致度計算
target_xy = np.array([[target_x, target_y]])
all_xy = df_clean[["BodyAxis", "SweetAxis"]].values
distances = cdist(target_xy, all_xy).flatten()
df_clean["distance"] = distances
df_sorted = df_clean.sort_values("distance").head(10)

# ✅ 一致度TOP10
st.subheader("📋 近いワイン TOP10")
st.dataframe(df_sorted[["Type", "商品名", "distance"]].reset_index(drop=True))
