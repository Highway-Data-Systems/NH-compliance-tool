from __future__ import annotations

import os
import subprocess
import sys
import base64
import zipfile
from datetime import datetime
from io import BytesIO
from importlib import reload
from pathlib import Path
from xml.sax.saxutils import escape

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
HDS_LOGO_LIGHT = ASSET_DIR / "HDS logo landscape small 2022.png"


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


def _report_text(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return "-"
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(value)


def _status_counts(df: pd.DataFrame) -> tuple[int, int, int]:
    if df.empty or "status" not in df.columns:
        return 0, 0, 0
    pass_count = int((df["status"] == "PASS").sum())
    fail_count = int((df["status"] == "FAIL").sum())
    return len(df), pass_count, fail_count


def _safe_filename(value: str, fallback: str) -> str:
    safe_value = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in value).strip()
    return safe_value or fallback


def _survey_endpoint_rows(survey) -> list[list[str]]:
    points = []
    if {"start_x", "start_y", "end_x", "end_y"}.issubset(survey.metadata):
        points = [
            ("Start", survey.metadata.get("start_x"), survey.metadata.get("start_y"), survey.metadata.get("start_z")),
            ("End", survey.metadata.get("end_x"), survey.metadata.get("end_y"), survey.metadata.get("end_z")),
        ]
    elif not survey.geometry.empty and {"x", "y"}.issubset(survey.geometry.columns):
        start = survey.geometry.iloc[0]
        end = survey.geometry.iloc[-1]
        points = [
            ("Start", start.get("x"), start.get("y"), start.get("z")),
            ("End", end.get("x"), end.get("y"), end.get("z")),
        ]

    rows = []
    for label, east, north, height in points:
        if east is None or north is None:
            continue
        lon, lat = BNG_TO_WGS84.transform(float(east), float(north))
        rows.append(
            [
                f"{label} coordinates",
                f"E {_report_text(float(east))}, N {_report_text(float(north))}"
                + (f", Z {_report_text(float(height))}" if height is not None else "")
                + f" | Lat {_report_text(float(lat))}, Lon {_report_text(float(lon))}",
            ]
        )
    return rows


def _csv_bundle_bytes(ride_results: pd.DataFrame, mpd_results: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("combined_ukri_results.csv", nh_parser.dataframe_to_csv(ride_results))
        archive.writestr("combined_mpd_results.csv", nh_parser.dataframe_to_csv(mpd_results))
    return buffer.getvalue()


def _pdf_report_bytes(
    survey,
    ride_spec_name: str,
    ride_spec: dict,
    mpd_spec_name: str,
    mpd_spec: dict,
    exclusions: list[tuple[float, float]],
    ride_results: pd.DataFrame,
    mpd_results: pd.DataFrame,
    ride_status: str,
    mpd_status: str,
    overall_status: str,
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="National Highways Ride and MPD Evaluation Report",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(
        ParagraphStyle(
            name="SmallHeader",
            parent=styles["Small"],
            textColor=colors.white,
            fontName="Helvetica-Bold",
        )
    )
    styles.add(ParagraphStyle(name="Status", parent=styles["BodyText"], fontSize=10, leading=12, alignment=1))

    story = []
    if HDS_LOGO_LIGHT.exists():
        logo = Image(str(HDS_LOGO_LIGHT), width=42 * mm, height=42 * mm * 179 / 600)
        logo.hAlign = "LEFT"
        story.extend([logo, Spacer(1, 5)])

    survey_name = survey.metadata.get("survey") or survey.metadata.get("file_name") or "Loaded survey"
    story.append(Paragraph("National Highways Ride and MPD Evaluation Report", styles["Title"]))
    story.append(Paragraph(escape(str(survey_name)), styles["Heading2"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Small"]))
    story.append(Spacer(1, 8))

    status_data = [["Overall", "UKRI", "MPD"], [overall_status, ride_status, mpd_status]]
    status_table = Table(status_data, colWidths=[55 * mm, 55 * mm, 55 * mm])
    status_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#262730")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
                ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#dcfce7" if overall_status == "PASS" else "#fee2e2" if overall_status == "FAIL" else "#f3f4f6")),
                ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#dcfce7" if ride_status == "PASS" else "#fee2e2" if ride_status == "FAIL" else "#f3f4f6")),
                ("BACKGROUND", (2, 1), (2, 1), colors.HexColor("#dcfce7" if mpd_status == "PASS" else "#fee2e2" if mpd_status == "FAIL" else "#f3f4f6")),
            ]
        )
    )
    story.extend([status_table, Spacer(1, 10)])

    ride_total, ride_pass, ride_fail = _status_counts(ride_results)
    mpd_total, mpd_pass, mpd_fail = _status_counts(mpd_results)
    metadata_rows = [
        ["File type", survey.file_type],
        ["Survey date", survey.metadata.get("survey_date", "-")],
        ["Length", _format_m(survey.metadata.get("survey_length_m"))],
        ["Geometry rows", f"{len(survey.geometry):,}"],
        ["Ride rows", f"{len(survey.ride_10m):,}"],
        ["MPD rows", f"{len(survey.mpd_10m):,}"],
        ["Excluded regions", str(len(exclusions))],
        ["UKRI assessed sections", f"{ride_total:,} ({ride_pass:,} pass, {ride_fail:,} fail)"],
        ["MPD assessed sections", f"{mpd_total:,} ({mpd_pass:,} pass, {mpd_fail:,} fail)"],
    ]
    metadata_rows.extend(_survey_endpoint_rows(survey))
    meta_table = Table(metadata_rows, colWidths=[48 * mm, 118 * mm])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f4f6")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.extend([Paragraph("Survey Summary", styles["Heading2"]), meta_table, Spacer(1, 8)])

    story.append(Paragraph("Specification", styles["Heading2"]))
    story.append(
        Paragraph(
            escape(
                f"Ride quality: {ride_spec_name}. {ride_spec['surface_type']} on {ride_spec['traffic']}; "
                f"100% of 10 m values < {ride_spec['all_lt']} and 80% of 10 m values < {ride_spec['pct80_lt']}."
            ),
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            escape(
                f"MPD: {mpd_spec_name}. {mpd_spec['material']}, {mpd_spec['application']}; "
                f"average {mpd_spec['avg_min']} to {mpd_spec['avg_max']} mm, standard deviation <= {mpd_spec['std_max']} mm, "
                "with at least 50% valid 10 m values."
            ),
            styles["BodyText"],
        )
    )

    if exclusions:
        story.extend([Spacer(1, 6), Paragraph("Exclusions", styles["Heading2"])])
        exclusion_rows = [["Start m", "End m"]] + [[f"{start:,.1f}", f"{end:,.1f}"] for start, end in exclusions[:20]]
        if len(exclusions) > 20:
            exclusion_rows.append(["...", f"{len(exclusions) - 20} more"])
        exclusion_table = Table(exclusion_rows, colWidths=[40 * mm, 40 * mm])
        exclusion_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#262730")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(exclusion_table)

    def add_result_table(title: str, df: pd.DataFrame, columns: list[str], max_rows: int = 60):
        story.extend([PageBreak(), Paragraph(title, styles["Heading2"])])
        if df.empty:
            story.append(Paragraph("No assessable data found.", styles["BodyText"]))
            return
        fail_df = df[df["status"] == "FAIL"] if "status" in df.columns else pd.DataFrame()
        table_df = fail_df if not fail_df.empty else df.head(min(20, len(df)))
        shown = table_df[columns].head(max_rows).copy()
        rows = [[Paragraph(escape(col), styles["SmallHeader"]) for col in columns]]
        for _, row in shown.iterrows():
            rows.append([Paragraph(escape(_report_text(row.get(col))), styles["Small"]) for col in columns])
        widths = [25 * mm] * len(columns)
        result_table = Table(rows, colWidths=widths, repeatRows=1)
        result_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#262730")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        caption = "Failed sections are shown below." if not fail_df.empty else "No failed sections; first assessed rows are shown below."
        if len(table_df) > max_rows:
            caption += f" Showing first {max_rows:,} of {len(table_df):,} rows."
        story.extend([Paragraph(caption, styles["BodyText"]), Spacer(1, 4), result_table])

    ride_cols = [c for c in ["metric", "tracks", "section", "valid_10m_values", "max_ri", "pct_below_lower_limit", "status"] if c in ride_results.columns]
    mpd_cols = [c for c in ["metric", "tracks", "section", "valid_10m_values", "expected_10m_values", "valid_pct", "avg_mpd_mm", "std_mpd_mm", "status"] if c in mpd_results.columns]
    add_result_table("UKRI Assessment Detail", ride_results, ride_cols)
    add_result_table("MPD Assessment Detail", mpd_results, mpd_cols)

    doc.build(story)
    return buffer.getvalue()


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


def _ukri_track_columns(ride_df: pd.DataFrame) -> list[str]:
    if ride_df.empty:
        return []
    preferred = [column for column in ["ns_ri", "mns_ri", "mos_ri", "os_ri"] if column in ride_df.columns]
    if preferred:
        return preferred
    fallback = [column for column in ["left_ri", "right_ri"] if column in ride_df.columns]
    extras = [
        column
        for column in ride_df.columns
        if column.endswith("_ri") and column not in set(preferred + fallback + ["combined_ukri"])
    ]
    return fallback + extras


def _ukri_track_label(column: str) -> str:
    labels = {
        "ns_ri": "Nearside UKRI",
        "mns_ri": "Mid-nearside UKRI",
        "mos_ri": "Mid-offside UKRI",
        "os_ri": "Offside UKRI",
        "left_ri": "Left UKRI",
        "right_ri": "Right UKRI",
    }
    return labels.get(column, column)


def _combined_ukri_chart_data(ride_df: pd.DataFrame, track_columns: list[str]) -> pd.DataFrame:
    if ride_df.empty or not track_columns:
        return pd.DataFrame()
    existing_columns = [column for column in track_columns if column in ride_df.columns]
    if not existing_columns:
        return pd.DataFrame()
    chart_df = ride_df[["chainage"] + existing_columns].copy()
    chart_df["combined_ukri"] = chart_df[existing_columns].replace(0, np.nan).mean(axis=1)
    return chart_df[["chainage", "combined_ukri"]].dropna()


def _mpd_line_options(mpd_df: pd.DataFrame) -> list[str]:
    if mpd_df.empty or "line" not in mpd_df.columns:
        return []
    return sorted(str(line) for line in mpd_df["line"].dropna().unique())


def _combined_mpd_chart_data(mpd_df: pd.DataFrame, line_columns: list[str] | None = None) -> pd.DataFrame:
    if mpd_df.empty:
        return pd.DataFrame()
    df = mpd_df.copy()
    if line_columns and "line" in df.columns:
        df = df[df["line"].isin(line_columns)]
    if df.empty:
        return pd.DataFrame()
    return (
        df[df["mpd_mm"] > 0.001]
        .groupby("chainage", as_index=False)["mpd_mm"]
        .mean()
        .rename(columns={"mpd_mm": "combined_mpd_mm"})
    )


def _endpoint_distance(summary_a: pd.DataFrame, summary_b: pd.DataFrame, index: int) -> float | None:
    if summary_a.empty or summary_b.empty or not {"x", "y"}.issubset(summary_a.columns) or not {"x", "y"}.issubset(summary_b.columns):
        return None
    row_a = summary_a.iloc[index]
    row_b = summary_b.iloc[index]
    return float(np.hypot(float(row_a["x"]) - float(row_b["x"]), float(row_a["y"]) - float(row_b["y"])))


def _route_location_checks(primary, comparison) -> list[dict]:
    rows = []
    length_a = primary.metadata.get("survey_length_m")
    length_b = comparison.metadata.get("survey_length_m")
    if length_a is not None and length_b is not None:
        diff = abs(float(length_a) - float(length_b))
        rows.append(
            {
                "check": "Survey length",
                "primary": _format_m(length_a),
                "comparison": _format_m(length_b),
                "difference": _format_m(diff),
                "status": "PASS" if diff <= 25.0 else "WARN",
            }
        )

    for label, idx in [("Start coordinates", 0), ("End coordinates", -1)]:
        distance = _endpoint_distance(primary.geometry, comparison.geometry, idx)
        if distance is not None:
            rows.append(
                {
                    "check": label,
                    "primary": "available",
                    "comparison": "available",
                    "difference": f"{distance:,.2f} m",
                    "status": "PASS" if distance <= 10.0 else "WARN",
                }
            )
    return rows


def _apply_chainage_offset(df: pd.DataFrame, offset_m: float) -> pd.DataFrame:
    if df.empty or "chainage" not in df.columns or abs(offset_m) < 0.001:
        return df
    out = df.copy()
    out["chainage"] = out["chainage"] + offset_m
    return out


def _comparison_delta(primary: pd.DataFrame, comparison: pd.DataFrame, metric: str, suffix_a: str, suffix_b: str) -> pd.DataFrame:
    if primary.empty or comparison.empty:
        return pd.DataFrame()
    left = primary[["chainage", metric]].dropna().sort_values("chainage").rename(columns={metric: suffix_a})
    right = comparison[["chainage", metric]].dropna().sort_values("chainage").rename(columns={metric: suffix_b})
    merged = pd.merge_asof(left, right, on="chainage", direction="nearest", tolerance=5.0)
    merged = merged.dropna(subset=[suffix_b])
    if merged.empty:
        return merged
    merged["delta"] = merged[suffix_b] - merged[suffix_a]
    return merged


def _overall_results(
    survey,
    ride_spec: dict,
    mpd_spec: dict,
    exclusions: list[tuple[float, float]],
    ride_tracks: list[str] | None = None,
    mpd_lines: list[str] | None = None,
):
    ride_tracks = ride_tracks or _ukri_track_columns(survey.ride_10m)
    ride_results = nh_parser.evaluate_ride_combined(survey.ride_10m, ride_tracks, ride_spec, exclusions) if ride_tracks else pd.DataFrame()
    mpd_source = survey.mpd_10m
    if mpd_lines and not mpd_source.empty and "line" in mpd_source.columns:
        mpd_source = mpd_source[mpd_source["line"].isin(mpd_lines)]
    mpd_results = nh_parser.evaluate_mpd_combined_with_exclusions(mpd_source, mpd_spec, exclusions) if not mpd_source.empty else pd.DataFrame()

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
    ride_spec_name = st.selectbox(
        "Ride quality profile",
        list(RIDE_SPECS),
        index=None,
        placeholder="Select ride quality profile",
    )
    mpd_spec_name = st.selectbox(
        "MPD profile",
        list(MPD_SPECS),
        index=None,
        placeholder="Select MPD profile",
    )
    map_hover = st.toggle(
        "Map coordinates on charts",
        value=True,
        help="Adds nearest survey coordinates to chart hover tooltips and shows a map marker controlled by a chainage slider.",
    )
    st.divider()
    uploaded = st.file_uploader("Load BCD or RCD", type=["bcd", "rcd", "txt"])
    comparison_uploaded = st.file_uploader(
        "Optional comparison BCD or RCD",
        type=["bcd", "rcd", "txt"],
        help="Load a second survey for pre/post or repeat-run comparison.",
    )
    st.caption("RCD files are preferred as they contain exclusions and structure data. BCD files include derived ride/MPD values.")

if not uploaded:
    st.info("Choose one of the example BCD files to see pass/fail sections and MPD track checks.")
    st.stop()

try:
    text = nh_parser.read_uploaded_text(uploaded)
    survey = nh_parser.parse_survey_text(text, uploaded.name)
except Exception as exc:
    st.error(f"Could not parse file: {exc}")
    st.stop()

comparison_survey = None
comparison_geometry_geo = pd.DataFrame()
comparison_exclusions = []
if comparison_uploaded:
    try:
        comparison_text = nh_parser.read_uploaded_text(comparison_uploaded)
        comparison_survey = nh_parser.parse_survey_text(comparison_text, comparison_uploaded.name)
        comparison_geometry_geo = _geometry_with_latlon(comparison_survey.geometry)
        comparison_exclusions = nh_parser.exclusion_intervals(comparison_survey.events)
    except Exception as exc:
        st.error(f"Could not parse comparison file: {exc}")
        st.stop()

geometry_geo = _geometry_with_latlon(survey.geometry)
exclusions = nh_parser.exclusion_intervals(survey.events)

st.subheader(f"{survey.file_type}: {survey.metadata.get('survey') or uploaded.name}")
export_prefix = _safe_filename(Path(uploaded.name).stem, "nh_ride_mpd_report")

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

missing_specs = []
if ride_spec_name is None:
    missing_specs.append("Ride quality profile")
if mpd_spec_name is None:
    missing_specs.append("MPD profile")
if missing_specs:
    st.warning(f"Select {' and '.join(missing_specs)} in the sidebar to run pass/fail checks and exports.")
    if not survey.geometry.empty:
        st.markdown("**Survey Location**")
        _survey_map(geometry_geo)
    st.stop()

ride_spec = RIDE_SPECS[ride_spec_name]
mpd_spec = MPD_SPECS[mpd_spec_name]
available_ukri_tracks = _ukri_track_columns(survey.ride_10m)
selected_ukri_tracks = available_ukri_tracks
if available_ukri_tracks:
    selected_ukri_tracks = st.sidebar.multiselect(
        "UKRI tracks for calculation",
        available_ukri_tracks,
        default=available_ukri_tracks,
        format_func=_ukri_track_label,
        help="Combined UKRI pass/fail is calculated from all selected track values in each 300 m section.",
    )
    if not selected_ukri_tracks:
        st.warning("Select at least one UKRI track in the sidebar to run UKRI pass/fail checks.")
        st.stop()
available_mpd_lines = _mpd_line_options(survey.mpd_10m)
selected_mpd_lines = available_mpd_lines
if available_mpd_lines:
    selected_mpd_lines = st.sidebar.multiselect(
        "MPD tracks for calculation",
        available_mpd_lines,
        default=available_mpd_lines,
        help="Combined MPD pass/fail is calculated from all selected track values in each 100 m section.",
    )
    if not selected_mpd_lines:
        st.warning("Select at least one MPD track in the sidebar to run MPD pass/fail checks.")
        st.stop()
summary_ride_results, summary_mpd_results, ride_status, mpd_status, overall_status = _overall_results(
    survey, ride_spec, mpd_spec, exclusions, selected_ukri_tracks, selected_mpd_lines
)

tab_names = ["Summary", "Ride Index", "MPD"]
if comparison_survey is not None:
    tab_names.append("Compare")
tab_names.append("File Structure")
tabs = st.tabs(tab_names)
tab_summary, tab_ride, tab_mpd = tabs[:3]
tab_compare = tabs[3] if comparison_survey is not None else None
tab_structure = tabs[-1]

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
            f"80% of 10 m values < {ride_spec['pct80_lt']}. "
            f"Combined UKRI uses {len(selected_ukri_tracks)} selected track(s): "
            f"{', '.join(_ukri_track_label(track) for track in selected_ukri_tracks)}."
        )
    with c2:
        st.markdown("**MPD Requirement**")
        st.write(
            f"{mpd_spec['material']}, {mpd_spec['application']}: "
            f"average {mpd_spec['avg_min']} to {mpd_spec['avg_max']} mm, "
            f"standard deviation <= {mpd_spec['std_max']} mm, with at least 50% valid 10 m values. "
            f"Combined MPD uses {len(selected_mpd_lines)} selected track(s): {', '.join(selected_mpd_lines)}."
        )

    try:
        report_name = survey.metadata.get("survey") or uploaded.name.rsplit(".", 1)[0]
        safe_report_name = _safe_filename(report_name, export_prefix)
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "Download PDF report",
                data=_pdf_report_bytes(
                    survey,
                    ride_spec_name,
                    ride_spec,
                    mpd_spec_name,
                    mpd_spec,
                    exclusions,
                    summary_ride_results,
                    summary_mpd_results,
                    ride_status,
                    mpd_status,
                    overall_status,
                ),
                file_name=f"{safe_report_name}_report.pdf",
                mime="application/pdf",
            )
        with export_col2:
            st.download_button(
                "Download CSV results bundle",
                data=_csv_bundle_bytes(summary_ride_results, summary_mpd_results),
                file_name=f"{export_prefix}_combined_results.zip",
                mime="application/zip",
            )
    except ModuleNotFoundError:
        st.warning("PDF export needs the reportlab package. Run `pip install -r requirements.txt` and restart the app.")

    if survey.file_type == "RCD" and not survey.mpd_10m.empty:
        st.info(
            "This RCD contains raw profile data. MPD is derived here by averaging the RCD MSD records into "
            "10 m track values before applying the 100 m specification checks. Ride Index is derived from the "
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
        side_options = selected_ukri_tracks
        combined_ride = _combined_ukri_chart_data(survey.ride_10m, side_options)
        metric = "combined_ukri"
        ride_results = nh_parser.evaluate_ride_combined(survey.ride_10m, side_options, ride_spec, exclusions)
        chart_data = combined_ride
        chart_y = "combined_ukri"
        chart_title = "Combined UKRI by chainage"
        marker_key = "ride_marker_combined_ukri"

        pass_count = int((ride_results["status"] == "PASS").sum()) if not ride_results.empty else 0
        fail_count = int((ride_results["status"] == "FAIL").sum()) if not ride_results.empty else 0
        r1, r2, r3 = st.columns(3)
        r1.metric("Assessment lengths", len(ride_results))
        r2.metric("Pass", pass_count)
        r3.metric("Fail", fail_count)

        if chart_data.empty or chart_y not in chart_data.columns:
            st.info("No combined UKRI chart data was found for the selected ride view.")
        elif map_hover:
            chart_col, map_col = st.columns([2, 1])
            with chart_col:
                _line_chart(
                    chart_data,
                    "chainage",
                    chart_y,
                    chart_title,
                    geometry_geo,
                    map_hover,
                    exclusions,
                    f"ride_{metric}",
                    st.session_state.get(marker_key),
                    marker_key,
                )
            with map_col:
                selected_chainage = _chainage_picker(chart_data, "Map marker chainage", marker_key)
                _survey_map(geometry_geo, height=360, selected_chainage=selected_chainage)
        else:
            _line_chart(chart_data, "chainage", chart_y, chart_title, geometry_geo, map_hover, exclusions, f"ride_{metric}")

        st.dataframe(_style_status(ride_results), use_container_width=True, hide_index=True)
        st.download_button(
            "Download combined UKRI results CSV",
            data=nh_parser.dataframe_to_csv(ride_results),
            file_name=f"{export_prefix}_combined_ukri_results.csv",
            mime="text/csv",
        )

        show_track_charts = st.toggle("Show individual UKRI track charts", value=False, disabled=not side_options)
        if show_track_charts:
            for track in side_options:
                track_label = _ukri_track_label(track)
                track_results = nh_parser.evaluate_ride(survey.ride_10m, track, ride_spec, exclusions)
                st.write(track_label)
                if map_hover:
                    chart_col, map_col = st.columns([2, 1])
                    track_marker_key = f"ride_marker_{track}"
                    with chart_col:
                        _line_chart(
                            survey.ride_10m,
                            "chainage",
                            track,
                            f"{track_label} by chainage",
                            geometry_geo,
                            map_hover,
                            exclusions,
                            f"ride_track_{track}",
                            st.session_state.get(track_marker_key),
                            track_marker_key,
                        )
                    with map_col:
                        selected_chainage = _chainage_picker(survey.ride_10m, "Map marker chainage", track_marker_key)
                        _survey_map(geometry_geo, height=320, selected_chainage=selected_chainage)
                else:
                    _line_chart(survey.ride_10m, "chainage", track, f"{track_label} by chainage", geometry_geo, map_hover, exclusions, f"ride_track_{track}")
                st.dataframe(_style_status(track_results), use_container_width=True, hide_index=True)

with tab_mpd:
    if survey.mpd_10m.empty:
        st.info("No derived MPD rows were found in this file.")
    else:
        if "source" in survey.mpd_10m.columns:
            st.caption("RCD MPD rows are derived from MSD records by 10 m averaging.")
        selected_lines = selected_mpd_lines
        mpd_source = survey.mpd_10m[survey.mpd_10m["line"].isin(selected_lines)]
        mpd_results = nh_parser.evaluate_mpd_combined_with_exclusions(mpd_source, mpd_spec, exclusions)
        pass_count = int((mpd_results["status"] == "PASS").sum()) if not mpd_results.empty else 0
        fail_count = int((mpd_results["status"] == "FAIL").sum()) if not mpd_results.empty else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("100 m sections", len(mpd_results))
        m2.metric("Pass", pass_count)
        m3.metric("Fail", fail_count)

        avg_mpd = _combined_mpd_chart_data(mpd_source, selected_lines)
        st.markdown("**Combined MPD**")
        if map_hover:
            chart_col, map_col = st.columns([2, 1])
            avg_marker_key = "mpd_average_marker"
            with chart_col:
                _line_chart(
                    avg_mpd,
                    "chainage",
                    "combined_mpd_mm",
                    "Combined MPD by chainage",
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
            _line_chart(avg_mpd, "chainage", "combined_mpd_mm", "Combined MPD by chainage", geometry_geo, map_hover, exclusions, "mpd_average")

        st.dataframe(_style_status(mpd_results), use_container_width=True, hide_index=True)
        st.download_button(
            "Download combined MPD results CSV",
            data=nh_parser.dataframe_to_csv(mpd_results),
            file_name=f"{export_prefix}_combined_mpd_results.csv",
            mime="text/csv",
        )

        show_line_charts = st.toggle("Show individual MPD track charts", value=False)
        if show_line_charts:
            for line in selected_lines:
                line_df = mpd_source[mpd_source["line"] == line]
                line_results = nh_parser.evaluate_mpd_with_exclusions(line_df, mpd_spec, exclusions)
                st.write(f"Track {line}")
                if map_hover:
                    chart_col, map_col = st.columns([2, 1])
                    mpd_marker_key = f"mpd_marker_{line}"
                    with chart_col:
                        _line_chart(
                            line_df,
                            "chainage",
                            "mpd_mm",
                            f"MPD track {line}",
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
                    _line_chart(line_df, "chainage", "mpd_mm", f"MPD track {line}", geometry_geo, map_hover, exclusions, f"mpd_{line}")
                st.dataframe(_style_status(line_results), use_container_width=True, hide_index=True)

if tab_compare is not None:
    with tab_compare:
        st.markdown("**Comparison Checks**")
        check_rows = _route_location_checks(survey, comparison_survey)
        if check_rows:
            st.dataframe(_style_status(pd.DataFrame(check_rows)), use_container_width=True, hide_index=True)
        else:
            st.info("No geometry/length metadata was available for route location checks.")

        comparison_ukri_tracks = _ukri_track_columns(comparison_survey.ride_10m)
        common_ukri_tracks = [track for track in selected_ukri_tracks if track in comparison_ukri_tracks]
        comparison_mpd_lines = _mpd_line_options(comparison_survey.mpd_10m)
        common_mpd_lines = [line for line in selected_mpd_lines if line in comparison_mpd_lines]

        st.markdown("**Alignment**")
        enable_offset = st.toggle(
            "Apply chainage offset to comparison dataset",
            value=False,
            help="Use this when repeat surveys start a little earlier/later along the same route.",
        )
        length_hint = float(survey.metadata.get("survey_length_m") or 0.0)
        max_offset = max(100.0, min(500.0, length_hint * 0.1 if length_hint else 100.0))
        offset_m = (
            st.slider("Comparison chainage offset (m)", -max_offset, max_offset, 0.0, 1.0)
            if enable_offset
            else 0.0
        )

        if not common_ukri_tracks and not common_mpd_lines:
            st.warning("No matching UKRI tracks or MPD tracks were found between the two datasets.")

        comp_ride = _apply_chainage_offset(comparison_survey.ride_10m, offset_m)
        comp_mpd = _apply_chainage_offset(comparison_survey.mpd_10m, offset_m)

        if common_ukri_tracks:
            st.markdown("**Combined UKRI Comparison**")
            primary_ukri = _combined_ukri_chart_data(survey.ride_10m, common_ukri_tracks)
            comparison_ukri = _combined_ukri_chart_data(comp_ride, common_ukri_tracks)
            comparison_chart = pd.concat(
                [
                    primary_ukri.assign(dataset="Primary"),
                    comparison_ukri.assign(dataset="Comparison"),
                ],
                ignore_index=True,
            )
            if not comparison_chart.empty:
                fig = px.line(
                    comparison_chart,
                    x="chainage",
                    y="combined_ukri",
                    color="dataset",
                    title="Combined UKRI comparison",
                )
                for start, end in exclusions:
                    fig.add_vrect(x0=start, x1=end, fillcolor="rgba(239, 68, 68, 0.14)", line_width=0)
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)

            ukri_delta = _comparison_delta(primary_ukri, comparison_ukri, "combined_ukri", "primary_ukri", "comparison_ukri")
            if not ukri_delta.empty:
                d1, d2, d3 = st.columns(3)
                d1.metric("Matched UKRI points", f"{len(ukri_delta):,}")
                d2.metric("Mean delta", f"{ukri_delta['delta'].mean():.3f}")
                d3.metric("Max abs delta", f"{ukri_delta['delta'].abs().max():.3f}")
                st.dataframe(ukri_delta.head(500), use_container_width=True, hide_index=True)
        elif not survey.ride_10m.empty or not comparison_survey.ride_10m.empty:
            st.info("No matching UKRI tracks were found for comparison.")

        if common_mpd_lines:
            st.markdown("**Combined MPD Comparison**")
            primary_mpd = _combined_mpd_chart_data(survey.mpd_10m, common_mpd_lines)
            comparison_mpd = _combined_mpd_chart_data(comp_mpd, common_mpd_lines)
            comparison_chart = pd.concat(
                [
                    primary_mpd.assign(dataset="Primary"),
                    comparison_mpd.assign(dataset="Comparison"),
                ],
                ignore_index=True,
            )
            if not comparison_chart.empty:
                fig = px.line(
                    comparison_chart,
                    x="chainage",
                    y="combined_mpd_mm",
                    color="dataset",
                    title="Combined MPD comparison",
                )
                for start, end in exclusions:
                    fig.add_vrect(x0=start, x1=end, fillcolor="rgba(239, 68, 68, 0.14)", line_width=0)
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)

            mpd_delta = _comparison_delta(primary_mpd, comparison_mpd, "combined_mpd_mm", "primary_mpd_mm", "comparison_mpd_mm")
            if not mpd_delta.empty:
                d1, d2, d3 = st.columns(3)
                d1.metric("Matched MPD points", f"{len(mpd_delta):,}")
                d2.metric("Mean delta", f"{mpd_delta['delta'].mean():.3f} mm")
                d3.metric("Max abs delta", f"{mpd_delta['delta'].abs().max():.3f} mm")
                st.dataframe(mpd_delta.head(500), use_container_width=True, hide_index=True)
        elif not survey.mpd_10m.empty or not comparison_survey.mpd_10m.empty:
            st.info("No matching MPD tracks were found for comparison.")

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
