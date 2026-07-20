import os
import subprocess
import sys
import base64
from importlib import reload
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from pyproj import Transformer
from streamlit.runtime.scriptrunner import get_script_run_ctx

import nh_parser
from nh_specs import MPD_SPECS, RIDE_SPECS


nh_parser = reload(nh_parser)

BNG_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
ASSET_DIR = Path(__file__).parent / "assets"
HDS_LOGO_DARK = ASSET_DIR / "HDS logo landscape small white 2022.png"


if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)],
        check=False,
    )
    sys.exit()


def _format_m(value):
    if value is None:
        return "-"
    return f"{float(value):,.1f} m"


def _style_status(df: pd.DataFrame):
    if df.empty or "status" not in df.columns:
        return df
    return df.style.map(
        lambda v: "background-color: #d8f3dc; color: #14532d"
        if v == "PASS"
        else "background-color: #fee2e2; color: #7f1d1d"
        if v == "FAIL"
        else "",
        subset=["status"],
    )


def _status_label(has_data: bool, has_fail: bool) -> str:
    if not has_data:
        return "NO DATA"
    return "FAIL" if has_fail else "PASS"


def _status_delta(status: str) -> str:
    return {
        "PASS": "All assessed sections pass",
        "FAIL": "One or more assessed sections fail",
        "NO DATA": "No assessable results",
    }[status]


def _status_card(label: str, status: str, detail: str):
    colors = {
        "PASS": ("#14532d", "#dcfce7", "#22c55e"),
        "FAIL": ("#7f1d1d", "#fee2e2", "#ef4444"),
        "NO DATA": ("#374151", "#f3f4f6", "#9ca3af"),
    }
    text, bg, border = colors[status]
    st.markdown(
        f"""
        <div style="border:1px solid {border}; background:{bg}; border-radius:8px; padding:14px 16px;">
            <div style="font-size:0.85rem; color:{text}; font-weight:700;">{label}</div>
            <div style="font-size:2rem; line-height:1.2; color:{text}; font-weight:800;">{status}</div>
            <div style="font-size:0.85rem; color:{text};">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _geometry_with_latlon(geometry: pd.DataFrame) -> pd.DataFrame:
    if geometry.empty or not {"x", "y"}.issubset(geometry.columns):
        return pd.DataFrame()
    df = geometry.copy()
    lon, lat = BNG_TO_WGS84.transform(df["x"].to_numpy(), df["y"].to_numpy())
    df["lon"] = lon
    df["lat"] = lat
    return df


def _with_nearest_location(data: pd.DataFrame, geometry_geo: pd.DataFrame) -> pd.DataFrame:
    if data.empty or geometry_geo.empty or "chainage" not in data.columns:
        return data
    loc = geometry_geo[["chainage", "x", "y", "lat", "lon"]].sort_values("chainage")
    base = data.sort_values("chainage")
    return pd.merge_asof(base, loc, on="chainage", direction="nearest")


def _survey_map(geometry_geo: pd.DataFrame, height: int = 360, selected_chainage: float | None = None):
    if geometry_geo.empty:
        st.info("No geometry coordinates were found for mapping.")
        return
    path = geometry_geo[["lon", "lat"]].dropna().values.tolist()
    midpoint = geometry_geo[["lat", "lon"]].mean()
    endpoints = geometry_geo.iloc[[0, -1]].copy()
    endpoints["point"] = ["Start", "End"]
    endpoints["color"] = [[34, 197, 94, 230], [239, 68, 68, 230]]
    layers = [
        pdk.Layer(
            "PathLayer",
            data=[{"path": path, "name": "Survey route"}],
            get_path="path",
            get_width=5,
            get_color=[25, 118, 210],
            width_min_pixels=3,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=endpoints,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=18,
            radius_min_pixels=5,
            pickable=True,
        ),
    ]
    if selected_chainage is not None:
        selected = _nearest_geometry_row(geometry_geo, selected_chainage)
        if selected is not None:
            selected["point"] = "Selected"
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame([selected]),
                    get_position="[lon, lat]",
                    get_fill_color=[239, 68, 68, 240],
                    get_radius=28,
                    radius_min_pixels=8,
                    pickable=True,
                )
            )
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=float(midpoint["lat"]),
                longitude=float(midpoint["lon"]),
                zoom=12,
                pitch=0,
            ),
            layers=layers,
            tooltip={"text": "{point}\nChainage: {chainage} m\nE: {x}\nN: {y}"},
        ),
        use_container_width=True,
        height=height,
    )


def _line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    geometry_geo: pd.DataFrame,
    map_hover: bool,
    exclusions: list[tuple[float, float]] | None = None,
    key: str | None = None,
    selected_chainage: float | None = None,
    marker_key: str | None = None,
):
    chart_df = _with_nearest_location(df[[x, y]].dropna(), geometry_geo) if map_hover else df[[x, y]].dropna()
    hover_cols = ["x", "y", "lat", "lon"] if map_hover and {"x", "y", "lat", "lon"}.issubset(chart_df.columns) else None
    fig = px.line(chart_df, x=x, y=y, title=title, hover_data=hover_cols)
    for start, end in exclusions or []:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(239, 68, 68, 0.18)",
            line_width=0,
            annotation_text="Excluded",
            annotation_position="top left",
        )
    if selected_chainage is not None:
        fig.add_vline(
            x=selected_chainage,
            line_width=3,
            line_color="#facc15",
            annotation_text=f"{selected_chainage:.0f} m",
            annotation_position="top right",
        )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    if map_hover and marker_key:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="points",
        )
        selection = getattr(event, "selection", None) if event else None
        if selection is None and isinstance(event, dict):
            selection = event.get("selection")
        points = selection.get("points", []) if isinstance(selection, dict) else []
        if points and points[0].get("x") is not None:
            st.session_state[marker_key] = float(points[0]["x"])
            return float(points[0]["x"])
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)
    return None


def _chainage_picker(data: pd.DataFrame, label: str, key: str) -> float | None:
    if data.empty or "chainage" not in data.columns:
        return None
    min_chainage = float(np.floor(data["chainage"].min() / 10.0) * 10.0)
    max_chainage = float(np.ceil(data["chainage"].max() / 10.0) * 10.0)
    if min_chainage >= max_chainage:
        return min_chainage
    if key not in st.session_state:
        st.session_state[key] = min_chainage
    st.session_state[key] = min(max(float(st.session_state[key]), min_chainage), max_chainage)
    return st.slider(
        label,
        min_value=min_chainage,
        max_value=max_chainage,
        step=10.0,
        key=key,
    )


def _nearest_geometry_row(geometry_geo: pd.DataFrame, chainage: float):
    if geometry_geo.empty:
        return None
    idx = (geometry_geo["chainage"] - chainage).abs().idxmin()
    row = geometry_geo.loc[idx]
    return row.to_dict()


def _overall_results(survey, ride_spec: dict, mpd_spec: dict, exclusions: list[tuple[float, float]]):
    ride_frames = []
    for metric in ["left_ri", "right_ri"]:
        if metric in survey.ride_10m.columns:
            result = nh_parser.evaluate_ride(survey.ride_10m, metric, ride_spec, exclusions)
            if not result.empty:
                result = result.copy()
                result.insert(0, "metric", metric)
                ride_frames.append(result)
    ride_results = pd.concat(ride_frames, ignore_index=True) if ride_frames else pd.DataFrame()
    mpd_results = nh_parser.evaluate_mpd_with_exclusions(survey.mpd_10m, mpd_spec, exclusions) if not survey.mpd_10m.empty else pd.DataFrame()

    ride_status = _status_label(not ride_results.empty, not ride_results.empty and (ride_results["status"] == "FAIL").any())
    mpd_status = _status_label(not mpd_results.empty, not mpd_results.empty and (mpd_results["status"] == "FAIL").any())
    overall_status = "FAIL" if "FAIL" in (ride_status, mpd_status) else "PASS" if "NO DATA" not in (ride_status, mpd_status) else "NO DATA"
    return ride_results, mpd_results, ride_status, mpd_status, overall_status


st.set_page_config(page_title="NH Ride and MPD Evaluator", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.5rem;
        }
        .hds-top-bar {
            background: #0e1117;
            margin: 0 0 1rem 0;
            padding: 0.35rem 1.55rem 0.95rem 1.55rem;
            overflow: visible;
        }
        .hds-top-bar img {
            width: 188px;
            max-width: 42vw;
            height: auto;
            display: block;
            margin: 0;
            padding: 0;
            object-fit: contain;
            object-position: left center;
            margin-bottom: 0.35rem;
        }
        .hds-top-bar h1 {
            color: #ffffff;
            margin: 0;
            line-height: 1.08;
        }
        .hds-top-bar p {
            color: rgba(255,255,255,0.74);
            margin: 0.45rem 0 0 0;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if HDS_LOGO_DARK.exists():
    logo_data = base64.b64encode(HDS_LOGO_DARK.read_bytes()).decode("ascii")
    logo_src = f"data:image/png;base64,{logo_data}"
else:
    logo_src = ""

st.markdown(
    f"""
    <div class="hds-top-bar">
        {'<img src="' + logo_src + '" alt="HDS logo">' if logo_src else ''}
        <h1>National Highways Ride and MPD Evaluator</h1>
        <p>Load a BCD or Surface Profile RCD file and review structure, coverage, charts and draft specification checks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Specification")
    ride_spec_name = st.selectbox("Ride quality profile", list(RIDE_SPECS))
    mpd_spec_name = st.selectbox("MPD profile", list(MPD_SPECS))
    map_hover = st.toggle(
        "Map coordinates on charts",
        value=True,
        help="Adds nearest survey coordinates to chart hover tooltips and shows a map marker controlled by a chainage slider.",
    )
    st.divider()
    uploaded = st.file_uploader("Load BCD or RCD", type=["bcd", "rcd", "txt"])
    st.caption("RCD files are preferred as they contain exclusions and structure data. BCD files include derived ride/MPD values.")

if not uploaded:
    st.info("Choose one of the example BCD files to see pass/fail sections and MPD line checks.")
    st.stop()

try:
    text = nh_parser.read_uploaded_text(uploaded)
    survey = nh_parser.parse_survey_text(text, uploaded.name)
except Exception as exc:
    st.error(f"Could not parse file: {exc}")
    st.stop()

ride_spec = RIDE_SPECS[ride_spec_name]
mpd_spec = MPD_SPECS[mpd_spec_name]
geometry_geo = _geometry_with_latlon(survey.geometry)
exclusions = nh_parser.exclusion_intervals(survey.events)
summary_ride_results, summary_mpd_results, ride_status, mpd_status, overall_status = _overall_results(
    survey, ride_spec, mpd_spec, exclusions
)

st.subheader(f"{survey.file_type}: {survey.metadata.get('survey') or uploaded.name}")

meta_cols = st.columns(5)
meta_cols[0].metric("Length", _format_m(survey.metadata.get("survey_length_m")))
meta_cols[1].metric("Geometry rows", f"{len(survey.geometry):,}")
meta_cols[2].metric("Raw LP records", f"{survey.metadata.get('longitudinal_profile_records', 0):,}")
meta_cols[3].metric("Ride rows", f"{len(survey.ride_10m):,}")
meta_cols[4].metric("MPD rows", f"{len(survey.mpd_10m):,}")

with st.expander("File metadata", expanded=False):
    st.json(survey.metadata)
    st.caption(f"Parser version: {nh_parser.PARSER_VERSION}")
    if not survey.quality_limits.empty:
        st.write("BCD embedded quality limits")
        st.dataframe(survey.quality_limits, use_container_width=True, hide_index=True)

tab_summary, tab_ride, tab_mpd, tab_structure = st.tabs(
    ["Summary", "Ride Index", "MPD", "File Structure"]
)

with tab_summary:
    s1, s2, s3 = st.columns(3)
    with s1:
        _status_card("Overall", overall_status, _status_delta(overall_status))
    with s2:
        _status_card("UKRI", ride_status, _status_delta(ride_status))
    with s3:
        _status_card("MPD", mpd_status, _status_delta(mpd_status))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ride Requirement**")
        st.write(
            f"{ride_spec['surface_type']} on {ride_spec['traffic']}: "
            f"100% of 10 m values < {ride_spec['all_lt']} and "
            f"80% of 10 m values < {ride_spec['pct80_lt']}."
        )
    with c2:
        st.markdown("**MPD Requirement**")
        st.write(
            f"{mpd_spec['material']}, {mpd_spec['application']}: "
            f"average {mpd_spec['avg_min']} to {mpd_spec['avg_max']} mm, "
            f"standard deviation <= {mpd_spec['std_max']} mm, with at least 50% valid 10 m values."
        )

    if survey.file_type == "RCD" and not survey.mpd_10m.empty:
        st.info(
            "This RCD contains raw profile data. MPD is derived here by averaging the RCD MSD records into "
            "10 m line values before applying the 100 m specification checks. Ride Index is derived from the "
            "raw longitudinal profile and assessed as 10 m UKRI values over 300 m sections."
        )
    elif survey.file_type == "RCD":
        st.warning(
            "This RCD is raw profile data. The app currently validates structure, events, geometry and coverage; "
            "RI/MPD pass/fail checks need a BCD or a completed derived-value algorithm."
        )

    st.markdown("**Survey Location**")
    _survey_map(geometry_geo)
    if exclusions:
        st.caption(f"{len(exclusions)} excluded region(s) removed from UKRI and MPD pass/fail calculations.")

    if not survey.geometry.empty:
        st.markdown("**Longitudinal Geometry**")
        _line_chart(
            survey.geometry,
            "chainage",
            "z",
            "Survey height profile",
            geometry_geo,
            map_hover,
            exclusions,
            "summary_height",
        )

with tab_ride:
    if survey.ride_10m.empty:
        if survey.file_type == "RCD" and survey.metadata.get("longitudinal_profile_records"):
            st.info(
                f"This RCD contains {survey.metadata['longitudinal_profile_records']:,} raw longitudinal "
                "profile records, but no Ride Index rows could be calculated from them."
            )
        else:
            st.info("No derived 10 m ride table was found in this file.")
    else:
        ride_columns = [c for c in survey.ride_10m.columns if c != "chainage"]
        side_options = [c for c in ["left_ri", "right_ri"] if c in ride_columns]
        other_options = [c for c in ride_columns if c not in {"left_ri", "right_ri"}]
        if side_options:
            side_metric = st.radio(
                "UKRI side",
                side_options,
                horizontal=True,
                format_func=lambda value: "Left UKRI" if value == "left_ri" else "Right UKRI",
            )
        else:
            side_metric = None
        detail_metric = st.selectbox("Other ride metric", other_options, index=0) if other_options else None
        use_detail_metric = st.toggle("Show dropdown metric", value=False, disabled=detail_metric is None)
        metric = detail_metric if use_detail_metric and detail_metric else side_metric or detail_metric
        ride_results = nh_parser.evaluate_ride(survey.ride_10m, metric, ride_spec, exclusions)
        pass_count = int((ride_results["status"] == "PASS").sum()) if not ride_results.empty else 0
        fail_count = int((ride_results["status"] == "FAIL").sum()) if not ride_results.empty else 0
        r1, r2, r3 = st.columns(3)
        r1.metric("Assessment lengths", len(ride_results))
        r2.metric("Pass", pass_count)
        r3.metric("Fail", fail_count)

        chart_col, map_col = st.columns([2, 1]) if map_hover else (None, None)
        if map_hover:
            ride_marker_key = f"ride_marker_{metric}"
            with chart_col:
                _line_chart(
                    survey.ride_10m,
                    "chainage",
                    metric,
                    f"{metric} by chainage",
                    geometry_geo,
                    map_hover,
                    exclusions,
                    f"ride_{metric}",
                    st.session_state.get(ride_marker_key),
                    ride_marker_key,
                )
            with map_col:
                selected_chainage = _chainage_picker(survey.ride_10m, "Map marker chainage", ride_marker_key)
                _survey_map(geometry_geo, height=360, selected_chainage=selected_chainage)
        else:
            _line_chart(survey.ride_10m, "chainage", metric, f"{metric} by chainage", geometry_geo, map_hover, exclusions, f"ride_{metric}")
        st.dataframe(_style_status(ride_results), use_container_width=True, hide_index=True)
        st.download_button(
            "Download ride results CSV",
            data=nh_parser.dataframe_to_csv(ride_results),
            file_name="ride_results.csv",
            mime="text/csv",
        )

with tab_mpd:
    if survey.mpd_10m.empty:
        st.info("No derived MPD rows were found in this file.")
    else:
        if "source" in survey.mpd_10m.columns:
            st.caption("RCD MPD rows are derived from MSD records by 10 m averaging.")
        line_options = sorted(survey.mpd_10m["line"].unique())
        selected_lines = st.multiselect("Measurement lines", line_options, default=line_options)
        mpd_source = survey.mpd_10m[survey.mpd_10m["line"].isin(selected_lines)]
        mpd_results = nh_parser.evaluate_mpd_with_exclusions(mpd_source, mpd_spec, exclusions)
        pass_count = int((mpd_results["status"] == "PASS").sum()) if not mpd_results.empty else 0
        fail_count = int((mpd_results["status"] == "FAIL").sum()) if not mpd_results.empty else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("100 m line sections", len(mpd_results))
        m2.metric("Pass", pass_count)
        m3.metric("Fail", fail_count)

        avg_mpd = (
            mpd_source.groupby("chainage", as_index=False)["mpd_mm"]
            .mean()
            .rename(columns={"mpd_mm": "average_mpd_mm"})
        )
        st.markdown("**Average MPD**")
        if map_hover:
            chart_col, map_col = st.columns([2, 1])
            avg_marker_key = "mpd_average_marker"
            with chart_col:
                _line_chart(
                    avg_mpd,
                    "chainage",
                    "average_mpd_mm",
                    "Average MPD by chainage",
                    geometry_geo,
                    map_hover,
                    exclusions,
                    "mpd_average",
                    st.session_state.get(avg_marker_key),
                    avg_marker_key,
                )
            with map_col:
                selected_chainage = _chainage_picker(avg_mpd, "Map marker chainage", avg_marker_key)
                _survey_map(geometry_geo, height=320, selected_chainage=selected_chainage)
        else:
            _line_chart(avg_mpd, "chainage", "average_mpd_mm", "Average MPD by chainage", geometry_geo, map_hover, exclusions, "mpd_average")

        show_line_charts = st.toggle("Show individual MPD line charts", value=False)
        if show_line_charts:
            for line in selected_lines:
                line_df = mpd_source[mpd_source["line"] == line]
                st.write(f"Line {line}")
                if map_hover:
                    chart_col, map_col = st.columns([2, 1])
                    mpd_marker_key = f"mpd_marker_{line}"
                    with chart_col:
                        _line_chart(
                            line_df,
                            "chainage",
                            "mpd_mm",
                            f"MPD line {line}",
                            geometry_geo,
                            map_hover,
                            exclusions,
                            f"mpd_{line}",
                            st.session_state.get(mpd_marker_key),
                            mpd_marker_key,
                        )
                    with map_col:
                        selected_chainage = _chainage_picker(line_df, "Map marker chainage", mpd_marker_key)
                        _survey_map(geometry_geo, height=320, selected_chainage=selected_chainage)
                else:
                    _line_chart(line_df, "chainage", "mpd_mm", f"MPD line {line}", geometry_geo, map_hover, exclusions, f"mpd_{line}")

        st.dataframe(_style_status(mpd_results), use_container_width=True, hide_index=True)
        st.download_button(
            "Download MPD results CSV",
            data=nh_parser.dataframe_to_csv(mpd_results),
            file_name="mpd_results.csv",
            mime="text/csv",
        )

with tab_structure:
    if not survey.events.empty:
        st.markdown("**Events / Exclusions**")
        st.dataframe(survey.events, use_container_width=True, hide_index=True)
    if not survey.geometry.empty:
        st.markdown("**Geometry sample**")
        st.dataframe(survey.geometry.head(500), use_container_width=True, hide_index=True)
    if not survey.ride_10m.empty:
        st.markdown("**Ride table sample**")
        st.dataframe(survey.ride_10m.head(500), use_container_width=True, hide_index=True)
    if not survey.mpd_10m.empty:
        st.markdown("**MPD table sample**")
        st.dataframe(survey.mpd_10m.head(500), use_container_width=True, hide_index=True)
