# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import plotly.express as px
import plotly.graph_objects as go

# ✅ 日本語フォント
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

# ✅ データ読み込み
df = pd.read_csv("Merged_TasteDataDB15.csv")

# ✅ 官能軸セット
feature_components = {
    "甘味": ["ブドウ糖", "果糖","リンゴ酸", "酒石酸"],
    "酸味": ["リンゴ酸", "酒石酸"],
    "渋味": ["総ポリフェノール", "グリセリン", "pH"],
}

# ✅ 特徴量リスト
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

# ✅ 特徴量データ
X = df[features]
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imputed)

# ✅ PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# ✅ UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_pca)

# ✅ UMAP DataFrame
umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
umap_df["JAN"] = df["JAN"].astype(str)
umap_df["Type"] = df["Type"] if "Type" in df.columns else "Unknown"
umap_df["商品名"] = df["商品名"] if "商品名" in df.columns else umap_df["JAN"]
umap_df["希望小売価格"] = df["希望小売価格"] if "希望小売価格" in df.columns else np.nan

# ✅ Streamlit UI
st.title("UMAP + 任意商品選択 → 近傍Top10版（完全一致版）")

# ✅ 等高線 軸選択
selected_feature = st.selectbox("等高線軸を選択", list(feature_components.keys()))

# ✅ Z軸 → 1変数 or 平均
components = feature_components[selected_feature]
if len(components) == 1:
    z_var = components[0]
    umap_df["Z"] = df[z_var].astype(float)
else:
    umap_df["Z"] = df[components].mean(axis=1).values

# ✅ 背景 等高線（matplotlib）表示
fig_kde, ax = plt.subplots(figsize=(10, 8))
sns.kdeplot(
    x=umap_df["UMAP1"], y=umap_df["UMAP2"],
    weights=umap_df["Z"],
    fill=True, cmap="YlOrBr", levels=50, alpha=0.5, bw_adjust=0.5,
    ax=ax
)
ax.set_title("UMAP")
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.grid(True)

# ✅ → ここで matplotlib の「実際の軸range」を取得✨
x_range = ax.get_xlim()
y_range = ax.get_ylim()

# ✅ matplotlib プロット表示
st.pyplot(fig_kde)

# ✅ 近傍探索関数
def compute_nearest(df, target_x, target_y, top_n=10):
    df = df.copy()
    df["distance"] = np.sqrt((df["UMAP1"] - target_x)**2 + (df["UMAP2"] - target_y)**2)
    df_sorted = df.sort_values("distance").reset_index(drop=True)
    df_sorted["順位"] = df_sorted.index + 1
    return df_sorted.head(top_n)

# ✅ 商品選択UI
selected_product = st.selectbox(
    "基準とするワインを選んでください",
    umap_df["商品名"].unique()
)

# ✅ 選択商品座標
selected_row = umap_df[umap_df["商品名"] == selected_product].iloc[0]
clicked_x = selected_row["UMAP1"]
clicked_y = selected_row["UMAP2"]

st.write(f"選択ワイン: {selected_product} → 座標 ({clicked_x:.2f}, {clicked_y:.2f})")

# ✅ 近傍探索
nearest_df = compute_nearest(umap_df, clicked_x, clicked_y, top_n=10)

# ✅ Plotly scatter作成（Top10表示付き）
fig_plotly = px.scatter(
    umap_df,
    x="UMAP1", y="UMAP2",
    color="Type",
    hover_data=["商品名", "JAN", "希望小売価格"],
    opacity=0.85,
    size=np.full(len(umap_df), 6),
)

# ✅ 選択商品（赤丸）追加
fig_plotly.add_trace(go.Scatter(
    x=[clicked_x],
    y=[clicked_y],
    mode='markers',
    marker=dict(size=15, color='red', line=dict(width=2, color='black')),
    name=selected_product,
    hovertext=selected_product
))

# ✅ Top10順位番号追加
fig_plotly.add_trace(go.Scatter(
    x=nearest_df["UMAP1"],
    y=nearest_df["UMAP2"],
    text=nearest_df["順位"].astype(str),
    mode="text",
    textposition="top center",
    showlegend=False
))

# ✅ range固定＋aspect固定（完全一致）
fig_plotly.update_layout(
    xaxis=dict(range=x_range),
    yaxis=dict(range=y_range, scaleanchor="x", scaleratio=1),
    title="UMAP",
    xaxis_title="UMAP1",
    yaxis_title="UMAP2",
    legend_title="Type",
    autosize=True
)

# ✅ Plotly表示
st.plotly_chart(fig_plotly, use_container_width=True)

# ✅ 下に表表示
st.subheader("近いワイン Top10")
st.dataframe(nearest_df[["順位", "JAN", "希望小売価格", "商品名"]])
