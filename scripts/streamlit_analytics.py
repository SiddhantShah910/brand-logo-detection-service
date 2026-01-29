import streamlit as st
import pandas as pd
import plotly.express as px

FPS = 30 
DATA_PATH = "../outputs/detections.csv"

st.set_page_config(page_title="Brand Logo Analytics", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def add_features(df: pd.DataFrame, fps: int) -> pd.DataFrame:
    df = df.copy()
    df["exposure_sec"] = 1 / fps
    df["box_area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
    df["center_x"] = (df["x1"] + df["x2"]) / 2
    df["center_y"] = (df["y1"] + df["y2"]) / 2
    df["impact_score"] = df["exposure_sec"] * df["confidence"] * df["box_area"]
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    brand_options = sorted(df["brand"].unique())
    selected = st.sidebar.multiselect(
        "Select brands",
        brand_options,
        default=brand_options
    )
    conf_threshold = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    if not selected:
        return df.iloc[0:0]
    return df[(df["brand"].isin(selected)) & (df["confidence"] >= conf_threshold)]


def longest_streak(data: pd.DataFrame, fps: int) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["brand", "longest_streak_sec"])
    results = []
    for brand, bdf in data.sort_values("frame_id").groupby("brand"):
        longest = 0
        current = 0
        last_frame = None
        for frame in bdf["frame_id"]:
            if last_frame is None or frame == last_frame + 1:
                current += 1
            else:
                longest = max(longest, current)
                current = 1
            last_frame = frame
        longest = max(longest, current)
        results.append({"brand": brand, "longest_streak_sec": round(longest / fps, 2)})
    return pd.DataFrame(results)


df = add_features(load_data(DATA_PATH), FPS)
filtered_df = apply_filters(df)

total_exposure = filtered_df["exposure_sec"].sum()
avg_conf = filtered_df["confidence"].mean() if not filtered_df.empty else 0
total_detections = len(filtered_df)
total_impact = filtered_df["impact_score"].sum()

st.title("Brand Logo Analytics Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Exposure (sec)", round(total_exposure, 2))
col2.metric("Avg Confidence", round(avg_conf, 3))
col3.metric("Total Detections", total_detections)
col4.metric("Brand Impact Score", round(total_impact, 1))

brand_summary = (
    filtered_df
    .groupby("brand")
    .agg(
        exposure_sec=("exposure_sec", "sum"),
        avg_confidence=("confidence", "mean"),
        total_area=("box_area", "sum"),
        impact_score=("impact_score", "sum")
    )
    .reset_index()
)

col1, col2 = st.columns(2)
fig_exp = px.bar(
    brand_summary,
    x="brand",
    y="exposure_sec",
    title="Brand Exposure Time (Seconds)",
    labels={"exposure_sec": "Seconds"}
)
col1.plotly_chart(fig_exp, use_container_width=True)

fig_area = px.bar(
    brand_summary,
    x="brand",
    y="total_area",
    title="Brand Screen Dominance (Total Box Area)",
    labels={"total_area": "Pixel Area"}
)
col2.plotly_chart(fig_area, use_container_width=True)

fig_conf = px.box(
    filtered_df,
    x="brand",
    y="confidence",
    title="Confidence Distribution by Brand"
)
st.plotly_chart(fig_conf, use_container_width=True)

fig_time = px.scatter(
    filtered_df,
    x="timestamp",
    y="brand",
    size="box_area",
    color="brand",
    title="Brand Detection Timeline",
    labels={"timestamp": "Time (sec)"}
)
st.plotly_chart(fig_time, use_container_width=True)

fig_heat = px.scatter(
    filtered_df,
    x="center_x",
    y="center_y",
    size="box_area",
    color="brand",
    title="Viewer Attention Zones",
    labels={"center_x": "Screen X", "center_y": "Screen Y"}
)
st.plotly_chart(fig_heat, use_container_width=True)

streak_df = longest_streak(filtered_df, FPS)
fig_streak = px.bar(
    streak_df,
    x="brand",
    y="longest_streak_sec",
    title="Longest Continuous Visibility (Seconds)"
)
st.plotly_chart(fig_streak, use_container_width=True)

total_exposure_all = brand_summary["exposure_sec"].sum()
brand_summary["fair_share_pct"] = (
    (brand_summary["exposure_sec"] / total_exposure_all) * 100
    if total_exposure_all
    else 0
)
fig_share = px.pie(
    brand_summary,
    names="brand",
    values="fair_share_pct",
    title="Brand Fair Share (%)"
)
st.plotly_chart(fig_share, use_container_width=True)

peak_moments = (
    filtered_df
    .sort_values("impact_score", ascending=False)
    .groupby("brand")
    .first()
    .reset_index()[["brand", "timestamp", "impact_score"]]
)

st.subheader("Peak Brand Impact Moments")
st.dataframe(peak_moments, use_container_width=True)

with st.expander("Show raw detection data"):
    st.dataframe(filtered_df, use_container_width=True)
