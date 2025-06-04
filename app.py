# app.py（散布図＋等高線のみ版）

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.font_manager as fm

# ✅ フォント設定
font_path = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
font_prop = fm.FontProperties(fname=font_path)
matplotlib.rcdefaults()

# ✅ データ読み込み
df = pd.read_csv("Merged_TasteDataDB15.csv")

# ✅ パラメータ
z_variables = ["果糖", "ブドウ糖"]
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
target_jans = [
    "4935919319140", "4935919080316", "4935919058186", "850832004260", "4935919071604",
    "4935919193559", "4935919197175", "4935919052504", "4935919080378", "blendF",
    "4935919213578", "4935919961554", "4935919194624", "4935919080965", "850755000028",
]

# ✅ 特徴量処理
X = df[features]
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imputed)

# ✅ UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_scaled)

# ✅ データ整理
plot_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
plot_df["JAN"] = df["JAN"].astype(str)
plot_df["Type"] = df["Type"]

# ✅ 描画
fig = plt.figure(figsize=(12, 9))  # ← 4:3 = 12:9 固定 🎨

# ▼ 等高線
for z_var in z_variables:
    if z_var in df.columns:
        try:
            plot_df["Z"] = df[z_var].astype(float)
            sns.kdeplot(
                x=plot_df["UMAP1"], y=plot_df["UMAP2"],
                weights=plot_df["Z"],
                fill=True, cmap="YlOrBr", levels=30, alpha=0.25, bw_adjust=0.7,
                label=z_var
            )
        except Exception as e:
            st.warning(f"⚠️ 等高線エラー: {z_var} → {e}")

# ▼ 散布図
sns.scatterplot(
    data=plot_df, x="UMAP1", y="UMAP2",
    hue="Type", palette="Set2", s=50, edgecolor='k', alpha=0.85
)

# ▼ ターゲットJAN ラベル
for i, row in plot_df.iterrows():
    if row["JAN"] in target_jans:
        product_name = df.loc[df["JAN"].astype(str) == row["JAN"], "商品名"].values
        label = product_name[0] if len(product_name) > 0 else row["JAN"]

        plt.scatter(row["UMAP1"], row["UMAP2"], c='red', s=60, edgecolor='k', zorder=5)
        plt.text(row["UMAP1"], row["UMAP2"], label, fontsize=9, color='red', zorder=6, fontproperties=font_prop)

# ▼ 図設定
plt.title(f"UMAP + 等高線: {', '.join(z_variables)}", fontsize=14, fontproperties=font_prop)
plt.xlabel("UMAP1", fontproperties=font_prop)
plt.ylabel("UMAP2", fontproperties=font_prop)
plt.legend(title="Type & 成分", loc='upper right')
plt.grid(True)
plt.tight_layout()

# ✅ Streamlit で表示
st.pyplot(fig)
