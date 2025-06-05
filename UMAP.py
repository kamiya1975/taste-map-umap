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

# ✅ 順位付きマップ再表示
fig_plotly.add_trace(go.Scatter(
    x=nearest_df["UMAP1"],
    y=nearest_df["UMAP2"],
    text=nearest_df["順位"].astype(str),
    mode="text",
    textposition="top center",
    showlegend=False
))

st.plotly_chart(fig_plotly, use_container_width=True)

# ✅ 下に表表示
st.subheader("近いワイン Top10")
st.dataframe(nearest_df[["順位", "JAN", "希望小売価格", "商品名"]])
