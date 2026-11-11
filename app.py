"""Streamlit interface for lightweight 6-DoF exploration and clustering."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analysis import (
    FEATURE_COLUMNS,
    condition_tests,
    descriptive_statistics,
    evenly_sample,
    infer_column_mapping,
    participant_features,
    prepare_data,
    spatial_metrics,
)
from i18n import LANGUAGES, translate


ROOT = Path(__file__).resolve().parent
PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True, "responsive": True}
ACCENT = "#62e7ff"
MAGENTA = "#f284ff"
GREEN = "#58f3a4"

st.set_page_config(
    page_title="6-DoF Motion Lab",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
      background:
        radial-gradient(circle at 78% 0%, rgba(51, 94, 160, .20), transparent 32rem),
        radial-gradient(circle at 12% 22%, rgba(21, 123, 138, .12), transparent 26rem),
        #080d17;
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0d1625 0%, #09111d 100%);
      border-right: 1px solid rgba(126, 218, 255, .12);
    }
    [data-testid="stHeader"] { background: rgba(8, 13, 23, .72); }
    h1, h2, h3 { letter-spacing: -.025em; }
    h1 {
      background: linear-gradient(90deg, #f7fbff 5%, #62e7ff 68%, #a9a4ff 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .eyebrow {
      color: #62e7ff; font-size: .72rem; font-weight: 750; letter-spacing: .16em;
      text-transform: uppercase; margin-bottom: -.35rem;
    }
    .subtle { color: #93a4ba; max-width: 58rem; line-height: 1.65; }
    .source-pill {
      display: inline-flex; align-items: center; gap: .4rem; margin-top: .35rem;
      border: 1px solid rgba(98,231,255,.28); background: rgba(32,78,101,.18);
      color: #b9eff8; border-radius: 999px; padding: .28rem .68rem; font-size: .76rem;
    }
    [data-testid="stMetric"] {
      background: linear-gradient(145deg, rgba(20,32,52,.94), rgba(12,22,37,.92));
      border: 1px solid rgba(140, 201, 235, .13); border-radius: 14px;
      padding: .85rem 1rem; box-shadow: 0 12px 30px rgba(0,0,0,.15);
    }
    [data-testid="stMetricLabel"] { color: #93a4ba; }
    [data-testid="stMetricLabel"] {
      min-height: 2.55rem; align-items: flex-start;
    }
    [data-testid="stMetricLabel"] p {
      white-space: normal; line-height: 1.25;
    }
    [data-testid="stMetricValue"] {
      color: #f3f7fc; overflow: visible;
    }
    [data-testid="stMetricValue"] > div {
      font-size: clamp(1.75rem, 2.25vw, 2.5rem);
      line-height: 1.15; letter-spacing: -.025em;
      overflow: visible; text-overflow: clip; white-space: nowrap;
    }
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid #1d2c42; }
    .stTabs [data-baseweb="tab"] {
      background: transparent; border-radius: 9px 9px 0 0; padding: .45rem .8rem;
    }
    .stTabs [aria-selected="true"] { background: rgba(75,179,207,.11); }
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
      border-radius: 14px; overflow: hidden;
    }
    .callout {
      border-left: 3px solid #62e7ff; background: rgba(21,44,65,.55);
      padding: .7rem .9rem; border-radius: 0 9px 9px 0; color: #b9c8d9;
      font-size: .88rem; margin: .5rem 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def polish_figure(figure: go.Figure, height: int = 520) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=52, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,18,31,.72)",
        font=dict(family="Inter, ui-sans-serif, system-ui", color="#c8d4e3"),
        title_font=dict(size=17, color="#edf5ff"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#101b2a", font_color="#f5f8fb"),
    )
    figure.update_xaxes(gridcolor="rgba(132,171,207,.13)", zerolinecolor="#314159")
    figure.update_yaxes(gridcolor="rgba(132,171,207,.13)", zerolinecolor="#314159")
    return figure


@st.cache_data(show_spinner=False)
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    from io import BytesIO

    try:
        frame = pd.read_csv(BytesIO(content), skipinitialspace=True)
        if len(frame.columns) == 1:
            frame = pd.read_csv(BytesIO(content), sep=None, engine="python", skipinitialspace=True)
        return frame
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(content), encoding="gb18030", skipinitialspace=True)


@st.cache_data(show_spinner=False)
def load_builtin(dataset_id: str, point_limit: int) -> tuple[pd.DataFrame, str, int]:
    if dataset_id.startswith("viewgauss"):
        files = sorted((ROOT / "dataset").glob("sequence*.csv"))
        selected = files if dataset_id == "viewgauss_all" else files[:1]
        condition = "ViewGauss"
    else:
        files = sorted((ROOT / "raw").glob("User*.csv"))
        selected = files if dataset_id == "hololens_all" else files[:1]
        condition = "HoloLens"

    frames = []
    for path in selected:
        frame = pd.read_csv(path, skipinitialspace=True).dropna(how="all")
        if condition == "ViewGauss":
            # Repository files concatenate trials: Frame runs upward and then
            # restarts at 1. Preserve each run as an independent sample.
            trial = frame["Frame"].diff().le(0).cumsum().add(1)
            frame["participant"] = [f"{path.stem} · T{value:02d}" for value in trial]
            frame["condition"] = path.stem
            frame = frame.groupby("participant", sort=False, group_keys=False).head(point_limit)
        else:
            frame = frame.iloc[:point_limit].copy()
            frame["participant"] = path.stem
            frame["condition"] = condition
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(condition)
    combined = pd.concat(frames, ignore_index=True)
    trajectory_count = combined["participant"].nunique()
    return combined, condition, trajectory_count


def mapping_select(
    label: str,
    field: str,
    columns: list[str],
    inferred: dict[str, str | None],
    *,
    none_label: str,
    optional: bool = False,
):
    choices = ([none_label] if optional else []) + columns
    suggestion = inferred.get(field)
    if suggestion in choices:
        index = choices.index(suggestion)
    else:
        index = 0
    selected = st.selectbox(label, choices, index=index, key=f"map_{field}")
    return None if selected == none_label else selected


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def fmt(value: float, suffix: str = "") -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:,.3f}{suffix}"


with st.sidebar:
    language_name = st.selectbox(
        "Language / 语言 / Sprache", list(LANGUAGES), index=0, key="language"
    )
    language = LANGUAGES[language_name]
    t = partial(translate, language=language)

    st.markdown(f"### {t('data_parsing')}")
    source_type = st.radio(
        t("data_source"),
        ["builtin", "upload"],
        format_func=lambda value: t("builtin_data" if value == "builtin" else "upload_csv"),
        horizontal=True,
    )

    if source_type == "upload":
        uploaded = st.file_uploader(t("choose_csv"), type=["csv", "txt"])
        if uploaded is None:
            st.info(t("upload_hint"))
            st.stop()
        try:
            raw = read_csv_bytes(uploaded.getvalue())
            source_label = uploaded.name
        except Exception as exc:
            st.error(t("read_error", error=exc))
            st.stop()
    else:
        demo_options = ["viewgauss_one", "hololens_one", "viewgauss_all", "hololens_all"]
        demo = st.selectbox(
            t("demo_sample"), demo_options, format_func=lambda value: t(value)
        )
        point_limit = st.slider(t("points_per_demo"), 500, 4000, 2000, 250)
        try:
            raw, dataset_name, trajectory_count = load_builtin(demo, point_limit)
            source_label = t(
                "builtin_source", dataset=dataset_name, count=trajectory_count
            )
        except FileNotFoundError as exc:
            st.error(t("builtin_missing", dataset=str(exc)))
            st.stop()

    if raw.empty:
        st.error(t("empty_csv"))
        st.stop()

    none_label = t("none")
    with st.expander(t("column_mapping"), expanded=source_type == "upload"):
        inferred = infer_column_mapping(raw.columns)
        columns = [str(column) for column in raw.columns]
        field_labels = {
            "time": t("time_column"),
            "x": t("position_x"),
            "y": t("position_y"),
            "z": t("position_z"),
            "qx": t("quaternion", axis="qx"),
            "qy": t("quaternion", axis="qy"),
            "qz": t("quaternion", axis="qz"),
            "qw": t("quaternion", axis="qw"),
        }
        mapping = {
            field: mapping_select(
                label, field, columns, inferred, none_label=none_label
            )
            for field, label in field_labels.items()
        }
        mapping["participant"] = mapping_select(
            t("participant_optional"), "participant", columns, inferred,
            none_label=none_label, optional=True,
        )
        mapping["condition"] = mapping_select(
            t("condition_optional"), "condition", columns, inferred,
            none_label=none_label, optional=True,
        )

    time_modes = {
        "auto_detect": "auto",
        "seconds": "seconds",
        "milliseconds": "milliseconds",
        "microseconds": "microseconds",
        "nanoseconds": "nanoseconds",
        "dotnet_ticks": "dotnet_ticks",
        "frame_number": "frame",
    }
    selected_time_key = st.selectbox(
        t("timestamp_format"), list(time_modes), format_func=lambda value: t(value)
    )
    timestamp_mode = time_modes[selected_time_key]
    frame_rate = st.number_input(t("frame_rate"), 1.0, 240.0, 60.0, 1.0)

    coordinate_modes = {
        "native_xyz": "native",
        "unity_right": "unity_to_right_handed",
        "y_up_z_up": "y_up_to_z_up",
    }
    selected_convention_key = st.selectbox(
        t("coordinate_convention"),
        list(coordinate_modes),
        format_func=lambda value: t(value),
    )
    selected_convention_label = t(selected_convention_key)
    smoothing = st.slider(t("smoothing"), 1, 21, 5, 2)

    st.divider()
    max_plot_points = st.slider(t("plot_limit"), 500, 6000, 2500, 250)
    max_dbscan_points = st.slider(t("dbscan_limit"), 500, 6000, 3000, 250)

try:
    data, report = prepare_data(
        raw,
        mapping,
        timestamp_mode=timestamp_mode,
        frame_rate=frame_rate,
        coordinate_convention=coordinate_modes[selected_convention_key],
        smoothing_window=smoothing,
    )
except Exception as exc:
    st.error(t("parse_error", error=exc))
    st.stop()

if len(data) < 3:
    st.error(t("too_few_points"))
    st.stop()

features = participant_features(data)
participants = features["participant"].tolist()
feature_display_names = {
    "participant": t("participant"),
    "condition": t("condition"),
    "n_points": t("point_count"),
    **{feature: t(f"feature_{feature}") for feature in FEATURE_COLUMNS},
}

st.markdown(f'<div class="eyebrow">{t("eyebrow")}</div>', unsafe_allow_html=True)
st.title("6-DoF Motion Lab")
st.markdown(
    f"<div class='subtle'>{t('subtitle')}</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<span class='source-pill'>{t('source_summary', source=source_label, points=report.valid_rows, samples=len(participants))}</span>",
    unsafe_allow_html=True,
)

toolbar_a, toolbar_b = st.columns([1, 3])
with toolbar_a:
    selected_participant = st.selectbox(t("current_trajectory"), participants)
with toolbar_b:
    st.caption(
        t(
            "parse_summary",
            time=report.timestamp_mode,
            coordinates=selected_convention_label,
            dropped=report.dropped_rows,
            repaired=report.quaternion_repairs,
        )
    )

sample = data[data["participant"] == selected_participant].copy()
sample_features = features.set_index("participant").loc[selected_participant]

metric_columns = st.columns(5)
metric_columns[0].metric(f"{t('duration')} (s)", fmt(sample_features["duration_s"]))
metric_columns[1].metric(f"{t('distance')} (m)", fmt(sample_features["distance"]))
metric_columns[2].metric(f"{t('mean_speed')} (m/s)", fmt(sample_features["mean_speed"]))
metric_columns[3].metric(
    f"{t('spatial_entropy')} (bit)", fmt(sample_features["spatial_entropy_bits"])
)
metric_columns[4].metric(
    f"{t('path_efficiency')} (%)", fmt(sample_features["path_efficiency"] * 100)
)

trajectory_tab, signal_tab, entropy_tab, statistics_tab, learning_tab = st.tabs(
    [
        t("trajectory_tab"), t("time_series_tab"), t("entropy_tab"),
        t("statistics_tab"), t("learning_tab"),
    ]
)

with trajectory_tab:
    st.markdown(f"### {t('trajectory_3d')}")
    trajectory = evenly_sample(sample, max_plot_points)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=trajectory["x"],
            y=trajectory["y"],
            z=trajectory["z"],
            mode="lines",
            line=dict(color=trajectory["elapsed_s"], colorscale="Turbo", width=5),
            name=t("trajectory"),
            text=[f"t = {value:.2f} s" for value in trajectory["elapsed_s"]],
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>%{text}<extra></extra>",
        )
    )
    endpoints = sample.iloc[[0, -1]]
    figure.add_trace(
        go.Scatter3d(
            x=endpoints["x"], y=endpoints["y"], z=endpoints["z"], mode="markers+text",
            marker=dict(size=7, color=[GREEN, MAGENTA], line=dict(width=1, color="#ffffff")),
            text=[t("start"), t("end")], textposition="top center", name=t("start_end"),
            hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{selected_participant} · {t('colored_by_time')}",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data", bgcolor="rgba(7,13,23,.68)",
        ),
    )
    st.plotly_chart(polish_figure(figure, 590), width="stretch", config=PLOTLY_CONFIG)

    left, right = st.columns([2, 1])
    with left:
        endpoint_table = pd.DataFrame(
            {
                t("position"): [t("start"), t("end")],
                "X": [sample.iloc[0]["x"], sample.iloc[-1]["x"]],
                "Y": [sample.iloc[0]["y"], sample.iloc[-1]["y"]],
                "Z": [sample.iloc[0]["z"], sample.iloc[-1]["z"]],
            }
        )
        st.dataframe(endpoint_table, hide_index=True, width="stretch")
    with right:
        st.download_button(
            t("download_points"), csv_bytes(data), "processed_6dof.csv", "text/csv",
            width="stretch",
        )
        st.download_button(
            t("download_features"), csv_bytes(features), "participant_features.csv", "text/csv",
            width="stretch",
        )

with signal_tab:
    st.markdown(f"### {t('kinematic_series')}")
    st.markdown(
        f"<div class='callout'>{t('kinematic_note')}</div>",
        unsafe_allow_html=True,
    )
    timeline = evenly_sample(sample, max_plot_points)
    signal_figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=(t("speed_unit"), t("acceleration_unit"), t("angular_speed_unit")),
    )
    for row, (column, color) in enumerate(
        [("speed", ACCENT), ("acceleration", MAGENTA), ("angular_speed", GREEN)], start=1
    ):
        signal_figure.add_trace(
            go.Scatter(
                x=timeline["elapsed_s"], y=timeline[column], mode="lines",
                line=dict(color=color, width=1.8), name=column,
                hovertemplate=f"t=%{{x:.3f}} s<br>{t('value')}=%{{y:.4f}}<extra></extra>",
            ), row=row, col=1,
        )
    signal_figure.update_xaxes(title_text=t("time_axis"), row=3, col=1)
    signal_figure.update_layout(
        showlegend=False,
        title=f"{selected_participant} · {t('smoothing_title', window=smoothing)}",
    )
    st.plotly_chart(polish_figure(signal_figure, 720), width="stretch", config=PLOTLY_CONFIG)

with entropy_tab:
    st.markdown(f"### {t('exploration_occupancy')}")
    entropy_bins = st.slider(t("grid_bins"), 4, 32, 12, 2, key="entropy_bins")
    entropy, coverage, occupied, total = spatial_metrics(sample[["x", "y", "z"]], entropy_bins)
    entropy_metrics = st.columns(4)
    entropy_metrics[0].metric("Shannon Entropy", fmt(entropy, " bit"))
    entropy_metrics[1].metric(t("coverage"), fmt(coverage * 100, "%"))
    entropy_metrics[2].metric(t("occupied_cells"), f"{occupied:,}")
    entropy_metrics[3].metric(t("total_cells"), f"{total:,}")

    plane = st.radio(t("heatmap_projection"), ["X–Z", "X–Y", "Y–Z"], horizontal=True)
    plane_columns = {"X–Z": ("x", "z"), "X–Y": ("x", "y"), "Y–Z": ("y", "z")}
    horizontal, vertical = plane_columns[plane]
    heatmap_data, x_edges, y_edges = np.histogram2d(
        sample[horizontal], sample[vertical], bins=max(entropy_bins * 2, 12)
    )
    heatmap = go.Figure(
        go.Heatmap(
            x=(x_edges[:-1] + x_edges[1:]) / 2,
            y=(y_edges[:-1] + y_edges[1:]) / 2,
            z=heatmap_data.T,
            colorscale=[[0, "#08101d"], [.25, "#123d57"], [.6, "#20b7c9"], [1, "#f8e879"]],
            colorbar=dict(title=t("sample_points")),
            hovertemplate=f"{horizontal}=%{{x:.3f}}<br>{vertical}=%{{y:.3f}}<br>{t('points_short')}=%{{z:.0f}}<extra></extra>",
        )
    )
    heatmap.update_layout(
        title=t("occupancy_heatmap", plane=plane), xaxis_title=f"{horizontal.upper()} (m)",
        yaxis_title=f"{vertical.upper()} (m)", yaxis_scaleanchor="x",
    )
    st.plotly_chart(polish_figure(heatmap, 560), width="stretch", config=PLOTLY_CONFIG)
    st.caption(t("entropy_note"))

with statistics_tab:
    st.markdown(f"### {t('descriptive_tests')}")
    summary = descriptive_statistics(sample)
    summary = summary.rename(
        columns={
            "valid_points": t("stat_valid_points"),
            "mean": t("stat_mean"),
            "std": t("stat_std"),
            "min": t("stat_min"),
            "median": t("stat_median"),
            "max": t("stat_max"),
        },
        index={
            "speed": t("speed"),
            "acceleration": t("acceleration"),
            "angular_speed": t("angular_speed"),
        },
    )
    st.dataframe(summary.style.format("{:.4f}"), width="stretch")

    variable = st.selectbox(
        t("distribution_variable"),
        ["speed", "acceleration", "angular_speed"],
        format_func=lambda value: t(value),
        key="distribution",
    )
    variable_label = t(variable)
    distribution = evenly_sample(data.dropna(subset=[variable]), max_plot_points * max(len(participants), 1))
    histogram = px.histogram(
        distribution, x=variable, color="participant", nbins=60, marginal="box",
        color_discrete_sequence=px.colors.qualitative.Set2, opacity=.72,
        title=t("distribution_title", variable=variable_label),
    )
    histogram.update_layout(barmode="overlay")
    st.plotly_chart(polish_figure(histogram, 520), width="stretch", config=PLOTLY_CONFIG)

    st.markdown(f"#### {t('participant_features')}")
    st.dataframe(
        features.rename(columns=feature_display_names), hide_index=True, width="stretch"
    )
    tests = condition_tests(features)
    if tests.empty:
        st.info(t("no_tests"))
    else:
        tests["note"] = t("exploratory_uncorrected")
        tests["feature"] = tests["feature"].map(feature_display_names).fillna(tests["feature"])
        st.markdown(f"#### {t('condition_tests')}")
        tests = tests.rename(
            columns={
                "feature": t("test_feature"), "test": t("test_name"),
                "groups": t("test_groups"), "statistic": t("test_statistic"),
                "p_value": t("test_p_value"), "note": t("test_note"),
            }
        )
        st.dataframe(
            tests.style.format(
                {t("test_statistic"): "{:.4f}", t("test_p_value"): "{:.4g}"}
            ),
            hide_index=True,
        )

with learning_tab:
    point_tab, participant_tab = st.tabs([t("dbscan_tab"), t("participant_model_tab")])

    with point_tab:
        st.markdown(f"#### {t('density_clustering')}")
        st.caption(t("dbscan_note"))
        dbscan_sample = evenly_sample(sample, max_dbscan_points)
        controls = st.columns(3)
        eps = controls[0].number_input(t("eps"), 0.001, 10.0, 0.08, 0.01, format="%.3f")
        min_samples = controls[1].number_input(t("min_samples"), 2, 200, 12, 1)
        controls[2].metric(t("points_used"), f"{len(dbscan_sample):,}")
        labels = DBSCAN(eps=eps, min_samples=int(min_samples)).fit_predict(
            dbscan_sample[["x", "y", "z"]]
        )
        dbscan_sample["cluster"] = np.where(labels == -1, t("noise"), labels.astype(str))
        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = float(np.mean(labels == -1))
        st.caption(t("cluster_summary", clusters=cluster_count, noise=noise_ratio))
        cluster_figure = px.scatter_3d(
            dbscan_sample, x="x", y="y", z="z", color="cluster",
            color_discrete_sequence=px.colors.qualitative.Bold, opacity=.76,
            title=t("dbscan_title", participant=selected_participant),
            labels={"cluster": t("cluster")},
            hover_data={"elapsed_s": ":.2f", "x": ":.3f", "y": ":.3f", "z": ":.3f"},
        )
        cluster_figure.update_traces(marker=dict(size=3))
        cluster_figure.update_layout(scene=dict(aspectmode="data", bgcolor="rgba(7,13,23,.68)"))
        st.plotly_chart(polish_figure(cluster_figure, 590), width="stretch", config=PLOTLY_CONFIG)
        st.download_button(
            t("download_dbscan"), csv_bytes(dbscan_sample), "dbscan_points.csv", "text/csv"
        )

    with participant_tab:
        st.markdown(f"#### {t('behavior_clustering')}")
        st.caption(t("behavior_note"))
        default_features = [
            "distance", "mean_speed", "speed_std", "mean_angular_speed",
            "path_efficiency", "spatial_entropy_bits",
        ]
        selected_features = st.multiselect(
            t("model_features"),
            list(FEATURE_COLUMNS),
            default=default_features,
            format_func=lambda value: t(f"feature_{value}"),
        )
        clean_features = features.dropna(subset=selected_features).copy() if selected_features else pd.DataFrame()
        if len(selected_features) < 2:
            st.info(t("select_two_features"))
        elif len(clean_features) < 3:
            st.info(t("need_three_samples"))
        else:
            scaled = StandardScaler().fit_transform(clean_features[selected_features])
            maximum_clusters = min(6, len(clean_features) - 1)
            n_clusters = st.slider(t("kmeans_clusters"), 2, maximum_clusters, 2)
            clean_features["kmeans_cluster"] = KMeans(
                n_clusters=n_clusters, random_state=42, n_init=20
            ).fit_predict(scaled).astype(str)

            pca = PCA(n_components=2, random_state=42)
            components = pca.fit_transform(scaled)
            clean_features["PC1"] = components[:, 0]
            clean_features["PC2"] = components[:, 1]
            explained = pca.explained_variance_ratio_ * 100

            pca_figure = px.scatter(
                clean_features, x="PC1", y="PC2", color="kmeans_cluster",
                symbol="condition", text="participant", color_discrete_sequence=px.colors.qualitative.Set2,
                title=t("pca_title", variance=explained.sum()),
                hover_data=selected_features,
                labels={
                    "kmeans_cluster": t("cluster"),
                    "condition": t("condition"),
                    **{feature: feature_display_names[feature] for feature in selected_features},
                },
            )
            pca_figure.update_traces(marker=dict(size=12, line=dict(width=1, color="#ffffff")), textposition="top center")
            pca_figure.update_xaxes(title=f"PC1 ({explained[0]:.1f}%)")
            pca_figure.update_yaxes(title=f"PC2 ({explained[1]:.1f}%)")
            st.plotly_chart(polish_figure(pca_figure, 560), width="stretch", config=PLOTLY_CONFIG)

            loadings = pd.DataFrame(
                pca.components_.T,
                index=selected_features,
                columns=["PC1 loading", "PC2 loading"],
            )
            with st.expander(t("pca_details")):
                st.dataframe(
                    loadings.rename(index=feature_display_names).style.format("{:.3f}"),
                    width="stretch",
                )
                st.dataframe(
                    clean_features.rename(columns=feature_display_names),
                    hide_index=True,
                    width="stretch",
                )
            st.download_button(
                t("download_pca"), csv_bytes(clean_features),
                "participant_clusters_pca.csv", "text/csv",
            )

with st.expander(t("quality_methods")):
    st.markdown(
        f"""
        - {t('quality_rows', input=report.input_rows, valid=report.valid_rows, dropped=report.dropped_rows)}
        - {t('quality_quaternion', repairs=report.quaternion_repairs)}
        - {t('quality_time')}
        - {t('quality_coordinates')}
        - {t('quality_sampling')}
        """
    )
