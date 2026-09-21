from __future__ import annotations

import os
import subprocess
import sys
import base64
import json
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


def _display_status(value):
    return {"PASS": "COMPLIANT", "FAIL": "NON COMPLIANT"}.get(value, value)


WHOLE_NUMBER_COLUMNS = {
    "start_m", "end_m", "chainage", "section_start", "section_start_100m", "section_start_300m",
    "valid_10m_values", "expected_10m_values", "valid_track_values",
}
TWO_DECIMAL_COLUMNS = {
    "max_ri", "pct_below_lower_limit", "valid_pct", "avg_mpd_mm", "std_mpd_mm",
    "combined_ukri", "combined_mpd_mm", "primary_ukri", "comparison_pre_ukri",
    "comparison_ukri", "primary_mpd_mm", "comparison_pre_mpd_mm", "comparison_mpd_mm", "delta",
}


def _table_formatters(columns) -> dict[str, str]:
    formats = {}
    for column in columns:
        if column in WHOLE_NUMBER_COLUMNS or column.endswith("_count"):
            formats[column] = "{:.0f}"
        elif column in TWO_DECIMAL_COLUMNS:
            formats[column] = "{:.2f}"
    return formats


def _format_table(df: pd.DataFrame):
    if df.empty:
        return df
    return df.style.format(_table_formatters(df.columns), na_rep="-")


def _style_status(df: pd.DataFrame):
    if df.empty or "status" not in df.columns:
        return df
    shown = df.copy()
    shown["status"] = shown["status"].map(_display_status)
    return shown.style.map(
        lambda v: "background-color: #d8f3dc; color: #14532d"
        if v == "COMPLIANT"
        else "background-color: #fee2e2; color: #7f1d1d"
        if v == "NON COMPLIANT"
        else "",
        subset=["status"],
    ).format(_table_formatters(shown.columns), na_rep="-")


def _status_label(has_data: bool, has_fail: bool) -> str:
    if not has_data:
        return "NO DATA"
    return "FAIL" if has_fail else "PASS"


def _status_delta(status: str) -> str:
    return {
        "PASS": "All assessed sections are compliant",
        "FAIL": "One or more assessed sections are non compliant",
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
            <div style="font-size:2rem; line-height:1.2; color:{text}; font-weight:800;">{_display_status(status)}</div>
            <div style="font-size:0.85rem; color:{text};">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _report_text(value, column: str | None = None) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return "-"
        if column in WHOLE_NUMBER_COLUMNS or (column and column.endswith("_count")):
            return f"{value:,.0f}"
        if column in TWO_DECIMAL_COLUMNS:
            return f"{value:,.2f}"
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(_display_status(value))


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


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _fig_to_png_bytes(fig, dpi: int = 150, tight: bool = True) -> bytes:
    buffer = BytesIO()
    bbox_inches = "tight" if tight else None
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches=bbox_inches, facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buffer.getvalue()


def _ride_chart_png(ride_df: pd.DataFrame, tracks: list[str], ride_spec: dict, exclusions: list[tuple[float, float]]):
    data = _combined_ukri_chart_data(ride_df, tracks)
    if data.empty:
        return None
    plt = _load_pyplot()
    x = data["chainage"].to_numpy(dtype=float)
    y = data["combined_ukri"].to_numpy(dtype=float)
    all_lt = float(ride_spec["all_lt"])
    pct80 = float(ride_spec["pct80_lt"])
    ymax = max(all_lt * 1.25, float(np.nanmax(y)) * 1.1 if y.size else all_lt, all_lt + 1.0)
    fig, ax = plt.subplots(figsize=(7.3, 2.9))
    ax.axhspan(0, pct80, color="#dcfce7", zorder=0, label=f"Target < {pct80} mm")
    ax.axhspan(pct80, all_lt, color="#fef9c3", zorder=0, label=f"Caution {pct80}-{all_lt} mm")
    ax.axhspan(all_lt, ymax, color="#fee2e2", zorder=0, label=f"Non Compliant > {all_lt} mm")
    ax.axhline(pct80, color="#ca8a04", lw=1.0, ls="--")
    ax.axhline(all_lt, color="#dc2626", lw=1.1, ls="--")
    for index, (start, end) in enumerate(exclusions or []):
        ax.axvspan(start, end, color="#6b7280", alpha=0.30, zorder=1,
                   label="Exclusions" if index == 0 else "_nolegend_")
    ax.plot(x, y, color="#1d4ed8", lw=1.0, zorder=3, label="Combined UKRI (mm)")
    if x.size:
        ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Chainage (m)", fontsize=8)
    ax.set_ylabel("UKRI (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#e5e7eb", lw=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=6, ncol=3, frameon=False)
    ax.set_title("Combined UKRI vs chainage with specification bands", fontsize=9)
    return _fig_to_png_bytes(fig)


def _mpd_chart_png(mpd_df: pd.DataFrame, lines: list[str], mpd_spec: dict, exclusions: list[tuple[float, float]]):
    data = _combined_mpd_chart_data(mpd_df, lines)
    if data.empty:
        return None
    plt = _load_pyplot()
    x = data["chainage"].to_numpy(dtype=float)
    y = data["combined_mpd_mm"].to_numpy(dtype=float)
    avg_min = float(mpd_spec["avg_min"])
    avg_max = float(mpd_spec["avg_max"])
    ymax = max(avg_max * 1.35, float(np.nanmax(y)) * 1.1 if y.size else avg_max)
    ymin = min(0.0, (float(np.nanmin(y)) * 0.9 if y.size else 0.0))
    fig, ax = plt.subplots(figsize=(7.3, 2.9))
    ax.axhspan(avg_min, avg_max, color="#dcfce7", zorder=0, label=f"Spec range {avg_min}-{avg_max} mm")
    ax.axhline(avg_min, color="#16a34a", lw=1.0, ls="--")
    ax.axhline(avg_max, color="#16a34a", lw=1.0, ls="--")
    for index, (start, end) in enumerate(exclusions or []):
        ax.axvspan(start, end, color="#6b7280", alpha=0.30, zorder=1,
                   label="Exclusions" if index == 0 else "_nolegend_")
    ax.plot(x, y, color="#7c3aed", lw=1.0, zorder=3, label="Combined MPD (mm)")
    if x.size:
        ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Chainage (m)", fontsize=8)
    ax.set_ylabel("MPD (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#e5e7eb", lw=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=6, ncol=3, frameon=False)
    ax.set_title("Combined MPD vs chainage with specification range", fontsize=9)
    return _fig_to_png_bytes(fig)


def _comparison_chart_png(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    exclusions: list[tuple[float, float]],
):
    if primary.empty and comparison.empty:
        return None
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(7.3, 2.9))
    if not comparison.empty and metric in comparison.columns:
        ax.plot(
            comparison["chainage"].to_numpy(dtype=float),
            comparison[metric].to_numpy(dtype=float),
            color="#636efa",
            lw=1.0,
            alpha=0.75,
            zorder=2,
            label="Comparison/Pre",
        )
    if not primary.empty and metric in primary.columns:
        ax.plot(
            primary["chainage"].to_numpy(dtype=float),
            primary[metric].to_numpy(dtype=float),
            color="#15803d",
            lw=1.0,
            alpha=0.75,
            zorder=3,
            label="Primary",
        )
    for index, (start, end) in enumerate(exclusions or []):
        ax.axvspan(start, end, color="#6b7280", alpha=0.20, zorder=0,
                   label="Exclusions" if index == 0 else "_nolegend_")
    ax.set_xlabel("Chainage (m)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#e5e7eb", lw=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=7, ncol=3, frameon=False)
    ax.set_title(title, fontsize=9)
    return _fig_to_png_bytes(fig)


def _deg2num(lon_deg, lat_deg, zoom: int):
    n = 2.0 ** zoom
    xt = (lon_deg + 180.0) / 360.0 * n
    lat_rad = np.radians(lat_deg)
    yt = (1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n
    return xt, yt


def _expand_geo_bounds_to_aspect(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    aspect: float,
) -> tuple[float, float, float, float]:
    mean_lat = float(np.nanmean([lat_min, lat_max]))
    lon_scale = max(np.cos(np.radians(mean_lat)), 0.1)
    lon_span = max(lon_max - lon_min, 0.000001)
    lat_span = max(lat_max - lat_min, 0.000001)
    projected_aspect = (lon_span * lon_scale) / lat_span
    if projected_aspect < aspect:
        target_lon_span = aspect * lat_span / lon_scale
        extra = (target_lon_span - lon_span) / 2
        lon_min -= extra
        lon_max += extra
    elif projected_aspect > aspect:
        target_lat_span = lon_span * lon_scale / aspect
        extra = (target_lat_span - lat_span) / 2
        lat_min -= extra
        lat_max += extra
    return lon_min, lon_max, lat_min, lat_max


def _osm_route_map_png(geometry_geo: pd.DataFrame, exclusions: list[tuple[float, float]]):
    import urllib.request
    from PIL import Image

    lat = geometry_geo["lat"].to_numpy(dtype=float)
    lon = geometry_geo["lon"].to_numpy(dtype=float)
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    pad_lat = max((lat_max - lat_min) * 0.18, 0.0025)
    pad_lon = max((lon_max - lon_min) * 0.18, 0.0025)
    lat_min -= pad_lat
    lat_max += pad_lat
    lon_min -= pad_lon
    lon_max += pad_lon
    lon_min, lon_max, lat_min, lat_max = _expand_geo_bounds_to_aspect(
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        7.1 / 4.3,
    )

    zoom = 12
    for candidate in range(17, 4, -1):
        x0, y0 = _deg2num(lon_min, lat_max, candidate)
        x1, y1 = _deg2num(lon_max, lat_min, candidate)
        span = (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1)
        if span <= 24 and max(abs(x1 - x0), abs(y1 - y0)) >= 1.0:
            zoom = candidate
            break

    x0f, y0f = _deg2num(lon_min, lat_max, zoom)
    x1f, y1f = _deg2num(lon_max, lat_min, zoom)
    xt_min, xt_max = int(np.floor(min(x0f, x1f))), int(np.floor(max(x0f, x1f)))
    yt_min, yt_max = int(np.floor(min(y0f, y1f))), int(np.floor(max(y0f, y1f)))
    n_tiles = (xt_max - xt_min + 1) * (yt_max - yt_min + 1)
    if n_tiles < 1 or n_tiles > 40:
        return None

    tile = 256
    mosaic = Image.new("RGB", ((xt_max - xt_min + 1) * tile, (yt_max - yt_min + 1) * tile), "#e5e7eb")
    headers = {"User-Agent": "HDS-NH-Ride-MPD-Report/1.0 (highwaydata.systems)"}
    for tx in range(xt_min, xt_max + 1):
        for ty in range(yt_min, yt_max + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=6) as response:
                img = Image.open(BytesIO(response.read())).convert("RGB")
            mosaic.paste(img, ((tx - xt_min) * tile, (ty - yt_min) * tile))

    def to_px(lon_deg, lat_deg):
        xt, yt = _deg2num(lon_deg, lat_deg, zoom)
        return (xt - xt_min) * tile, (yt - yt_min) * tile

    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    ax.imshow(np.asarray(mosaic), origin="upper")
    route_x, route_y = to_px(lon, lat)
    ax.plot(route_x, route_y, color="#1d4ed8", lw=2.4, zorder=3, solid_capstyle="round")
    for start, end in exclusions or []:
        seg = geometry_geo[(geometry_geo["chainage"] >= start) & (geometry_geo["chainage"] <= end)]
        if not seg.empty:
            sx, sy = to_px(seg["lon"].to_numpy(dtype=float), seg["lat"].to_numpy(dtype=float))
            ax.plot(sx, sy, color="#dc2626", lw=3.4, zorder=4, solid_capstyle="round")
    sx, sy = to_px(lon[0], lat[0])
    ex, ey = to_px(lon[-1], lat[-1])
    ax.scatter([sx], [sy], s=70, color="#16a34a", edgecolor="white", linewidth=1.2, zorder=5, label="Start")
    ax.scatter([ex], [ey], s=70, color="#dc2626", edgecolor="white", linewidth=1.2, zorder=5, label="End")
    bx0, by0 = to_px(lon_min, lat_max)
    bx1, by1 = to_px(lon_max, lat_min)
    ax.set_xlim(bx0, bx1)
    ax.set_ylim(by1, by0)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.text(
        0.995,
        0.01,
        "(c) OpenStreetMap contributors",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )
    return _fig_to_png_bytes(fig, tight=False)


def _plain_route_map_png(geometry_geo: pd.DataFrame, exclusions: list[tuple[float, float]]):
    plt = _load_pyplot()
    lon = geometry_geo["lon"].to_numpy(dtype=float)
    lat = geometry_geo["lat"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    ax.plot(lon, lat, color="#1d4ed8", lw=2.0, zorder=3, label="Route")
    for start, end in exclusions or []:
        seg = geometry_geo[(geometry_geo["chainage"] >= start) & (geometry_geo["chainage"] <= end)]
        if not seg.empty:
            ax.plot(seg["lon"], seg["lat"], color="#dc2626", lw=3.0, zorder=4)
    ax.scatter([lon[0]], [lat[0]], s=55, color="#16a34a", edgecolor="white", linewidth=1.0, zorder=5, label="Start")
    ax.scatter([lon[-1]], [lat[-1]], s=55, color="#dc2626", edgecolor="white", linewidth=1.0, zorder=5, label="End")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    mean_lat = float(np.nanmean(lat)) if lat.size else 51.0
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    pad_lat = max((lat_max - lat_min) * 0.18, 0.0025)
    pad_lon = max((lon_max - lon_min) * 0.18, 0.0025)
    lon_min, lon_max, lat_min, lat_max = _expand_geo_bounds_to_aspect(
        lon_min - pad_lon,
        lon_max + pad_lon,
        lat_min - pad_lat,
        lat_max + pad_lat,
        7.1 / 4.3,
    )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / max(np.cos(np.radians(mean_lat)), 0.1))
    ax.grid(True, color="#e5e7eb", lw=0.5)
    ax.legend(loc="best", fontsize=7)
    ax.set_title("Survey route", fontsize=9)
    return _fig_to_png_bytes(fig, tight=False)


def _route_map_png(geometry_geo: pd.DataFrame, exclusions: list[tuple[float, float]]):
    if geometry_geo.empty or not {"lat", "lon", "chainage"}.issubset(geometry_geo.columns):
        return None
    if geometry_geo[["lat", "lon"]].dropna().empty:
        return None
    try:
        result = _osm_route_map_png(geometry_geo, exclusions)
        if result is not None:
            return result
    except Exception:
        pass
    try:
        return _plain_route_map_png(geometry_geo, exclusions)
    except Exception:
        return None


def _png_flowable(png_bytes: bytes, max_width_mm: float, max_height_mm: float = 180):
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RLImage
    from PIL import Image as PILImage

    with PILImage.open(BytesIO(png_bytes)) as pil_image:
        width_px, height_px = pil_image.size
    if width_px <= 0 or height_px <= 0:
        raise ValueError("Cannot add an image with invalid dimensions to the PDF report.")

    max_width = max_width_mm * mm
    max_height = max_height_mm * mm
    scale = min(max_width / width_px, max_height / height_px)
    width = width_px * scale
    height = height_px * scale
    flowable = RLImage(BytesIO(png_bytes), width=width, height=height)
    flowable.hAlign = "CENTER"
    return flowable


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
    geometry_geo: pd.DataFrame | None = None,
    ride_tracks: list[str] | None = None,
    mpd_lines: list[str] | None = None,
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    if geometry_geo is None:
        geometry_geo = _geometry_with_latlon(survey.geometry)
    ride_tracks = ride_tracks or _ukri_track_columns(survey.ride_10m)
    mpd_lines = mpd_lines or _mpd_line_options(survey.mpd_10m)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=22 * mm,
        bottomMargin=14 * mm,
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
    survey_name = survey.metadata.get("survey") or survey.metadata.get("file_name") or "Loaded survey"
    story.append(Paragraph("National Highways Ride and MPD Evaluation Report", styles["Title"]))
    story.append(Paragraph(escape(str(survey_name)), styles["Heading2"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Small"]))
    story.append(Spacer(1, 8))

    status_data = [["Overall", "UKRI", "MPD"], [_display_status(overall_status), _display_status(ride_status), _display_status(mpd_status)]]
    status_table = Table(status_data, colWidths=[55 * mm, 55 * mm, 55 * mm])
    status_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, 1), 12),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#9ca3af")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#dcfce7" if overall_status == "PASS" else "#fee2e2" if overall_status == "FAIL" else "#f3f4f6")),
                ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#dcfce7" if ride_status == "PASS" else "#fee2e2" if ride_status == "FAIL" else "#f3f4f6")),
                ("BACKGROUND", (2, 1), (2, 1), colors.HexColor("#dcfce7" if mpd_status == "PASS" else "#fee2e2" if mpd_status == "FAIL" else "#f3f4f6")),
                ("TEXTCOLOR", (0, 1), (0, 1), colors.HexColor("#14532d" if overall_status == "PASS" else "#7f1d1d" if overall_status == "FAIL" else "#374151")),
                ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor("#14532d" if ride_status == "PASS" else "#7f1d1d" if ride_status == "FAIL" else "#374151")),
                ("TEXTCOLOR", (2, 1), (2, 1), colors.HexColor("#14532d" if mpd_status == "PASS" else "#7f1d1d" if mpd_status == "FAIL" else "#374151")),
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
        ["UKRI assessed sections", f"{ride_total:,} ({ride_pass:,} compliant, {ride_fail:,} non compliant)"],
        ["MPD assessed sections", f"{mpd_total:,} ({mpd_pass:,} compliant, {mpd_fail:,} non compliant)"],
    ]
    metadata_rows.extend(_survey_endpoint_rows(survey))
    meta_table = Table(metadata_rows, colWidths=[48 * mm, 118 * mm])
    meta_style = [
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#9ca3af")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for row_index in range(len(metadata_rows)):
        if row_index % 2 == 1:
            meta_style.append(("BACKGROUND", (1, row_index), (1, row_index), colors.HexColor("#f9fafb")))
    meta_table.setStyle(TableStyle(meta_style))
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
        exclusion_table = Table(exclusion_rows, colWidths=[45 * mm, 45 * mm])
        exclusion_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#9ca3af")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for row_index in range(1, len(exclusion_rows)):
            if row_index % 2 == 0:
                exclusion_style.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f9fafb")))
        exclusion_table.setStyle(TableStyle(exclusion_style))
        story.append(exclusion_table)

    map_png = _route_map_png(geometry_geo, exclusions)
    if map_png is not None:
        story.append(Spacer(1, 10))
        story.append(
            KeepTogether(
                [
                    Paragraph("Survey Location", styles["Heading2"]),
                    _png_flowable(map_png, 172),
                    Paragraph(
                        "Blue trace shows the surveyed route; red segments mark excluded regions.",
                        styles["Small"],
                    ),
                ]
            )
        )

    def add_result_table(
        title: str,
        df: pd.DataFrame,
        columns: list[str],
        max_rows: int = 60,
        chart_png: bytes | None = None,
    ):
        story.extend([PageBreak(), Paragraph(title, styles["Heading2"])])
        if chart_png is not None:
            story.extend([_png_flowable(chart_png, 176), Spacer(1, 6)])
        if df.empty:
            story.append(Paragraph("No assessable data found.", styles["BodyText"]))
            return
        fail_df = df[df["status"] == "FAIL"] if "status" in df.columns else pd.DataFrame()
        table_df = fail_df if not fail_df.empty else df.head(min(20, len(df)))
        shown = table_df[columns].head(max_rows).copy()
        rows = [[Paragraph(escape(col), styles["SmallHeader"]) for col in columns]]
        for _, row in shown.iterrows():
            rows.append([Paragraph(escape(_report_text(row.get(col), col)), styles["Small"]) for col in columns])
        available_width = doc.width
        widths = [available_width / len(columns)] * len(columns)
        result_table = Table(rows, colWidths=widths, repeatRows=1)
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#9ca3af")),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#111827")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]
        for row_index in range(1, len(rows)):
            if row_index % 2 == 0:
                table_style.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f3f4f6")))
        if "status" in columns:
            status_col = columns.index("status")
            for offset, (_, row) in enumerate(shown.iterrows(), start=1):
                status_value = _report_text(row.get("status"))
                cell_bg = "#dcfce7" if status_value == "COMPLIANT" else "#fee2e2" if status_value == "NON COMPLIANT" else None
                if cell_bg:
                    table_style.append(
                        ("BACKGROUND", (status_col, offset), (status_col, offset), colors.HexColor(cell_bg))
                    )
        result_table.setStyle(TableStyle(table_style))
        caption = "Non compliant sections are shown below." if not fail_df.empty else "No non compliant sections; first assessed rows are shown below."
        if len(table_df) > max_rows:
            caption += f" Showing first {max_rows:,} of {len(table_df):,} rows."
        story.extend([Paragraph(caption, styles["BodyText"]), Spacer(1, 4), result_table])

    ride_cols = [c for c in ["metric", "tracks", "section", "valid_10m_values", "max_ri", "pct_below_lower_limit", "status"] if c in ride_results.columns]
    mpd_cols = [c for c in ["metric", "tracks", "section", "valid_10m_values", "expected_10m_values", "valid_pct", "avg_mpd_mm", "std_mpd_mm", "status"] if c in mpd_results.columns]
    ride_chart = _ride_chart_png(survey.ride_10m, ride_tracks, ride_spec, exclusions)
    mpd_chart = _mpd_chart_png(survey.mpd_10m, mpd_lines, mpd_spec, exclusions)
    add_result_table("UKRI Assessment Detail", ride_results, ride_cols, chart_png=ride_chart)
    add_result_table("MPD Assessment Detail", mpd_results, mpd_cols, chart_png=mpd_chart)

    def _draw_page_furniture(canvas, current_doc):
        canvas.saveState()
        if HDS_LOGO_LIGHT.exists():
            logo_width = 30 * mm
            logo_height = logo_width * 179 / 600
            canvas.drawImage(
                str(HDS_LOGO_LIGHT),
                A4[0] - 14 * mm - logo_width,
                A4[1] - 10 * mm - logo_height,
                width=logo_width,
                height=logo_height,
                mask="auto",
                preserveAspectRatio=True,
            )
        canvas.setStrokeColor(colors.HexColor("#e5e7eb"))
        canvas.line(14 * mm, 11 * mm, A4[0] - 14 * mm, 11 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawString(14 * mm, 8 * mm, "National Highways Ride and MPD Evaluation Report")
        canvas.drawRightString(A4[0] - 14 * mm, 8 * mm, f"Page {current_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_page_furniture, onLaterPages=_draw_page_furniture)
    return buffer.getvalue()


def _pdf_comparison_report_bytes(
    primary,
    comparison,
    exclusions: list[tuple[float, float]],
    ride_tracks: list[str],
    mpd_lines: list[str],
    offset_m: float = 0.0,
    resolution: str = "Automatic",
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=22 * mm,
        bottomMargin=14 * mm,
        title="National Highways Comparison Report",
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

    def make_table(rows, widths=None, font_size=8):
        table_rows = []
        for row_index, row in enumerate(rows):
            style = styles["SmallHeader"] if row_index == 0 else styles["Small"]
            table_rows.append([cell if isinstance(cell, Paragraph) else Paragraph(escape(_report_text(cell)), style) for cell in row])
        table = Table(table_rows, colWidths=widths, repeatRows=1)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#9ca3af")),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]
        for row_index in range(1, len(rows)):
            if row_index % 2 == 0:
                style.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f3f4f6")))
        table.setStyle(TableStyle(style))
        return table

    def percentage_cell(value, is_mpd):
        if pd.isna(value):
            return "N/A"
        positive, negative = ("#2563eb", "#a16207") if is_mpd else ("#15803d", "#dc2626")
        colour = positive if value > 0 else negative if value < 0 else "#737373"
        return Paragraph(f'<font color="{colour}"><b>{value:+.2f}%</b></font>', styles["Small"])

    def comparison_values(delta, primary_col, pre_col, is_mpd):
        if delta.empty:
            return [float("nan")] * 4
        percentages = _improvement_percent(delta[primary_col], delta[pre_col], is_mpd)
        overall = _improvement_percent(pd.Series([delta[primary_col].mean()]),
                                       pd.Series([delta[pre_col].mean()]), is_mpd).iloc[0]
        return [overall, percentages.mean(), percentages.max(), percentages.min()]

    def add_percentage_section(label, delta_df, primary_col, pre_col):
        is_mpd = label == "MPD"
        measure = "difference" if is_mpd else "improvement"
        story.extend([PageBreak(), Paragraph(f"{label} {measure}", styles["Heading2"])])
        if delta_df.empty:
            story.append(Paragraph("No matched comparison points were found within 5 m.", styles["BodyText"]))
            return
        span = float(delta_df["chainage"].max() - delta_df["chainage"].min() + 10)
        section_m = 100 if resolution == "100 m" or (resolution == "Automatic" and span > 1000) else 10
        table = _comparison_sections(delta_df, primary_col, pre_col, section_m)
        table["percentage"] = _improvement_percent(table[primary_col], table[pre_col], is_mpd)
        formula = "100 x (primary - pre) / pre" if is_mpd else "100 x (pre - primary) / pre"
        meaning = ("Positive differences (higher MPD) are blue; negative differences (lower MPD) are yellow."
                   if is_mpd else "Positive improvement (lower UKRI) is green; negative improvement (worse than pre) is red.")
        story.append(Paragraph(f"{label} {measure} = {formula}. {meaning} "
                               "Overall compares matched survey means; other summaries use 10 m percentages. "
                               "All matched points are included, including exclusions.", styles["Small"]))
        titles = [f"Overall {measure}", f"Mean 10 m {measure}",
                  f"{'Maximum' if is_mpd else 'Best'} 10 m {measure}",
                  f"{'Minimum' if is_mpd else 'Worst'} 10 m {measure}"]
        rows = [["Measure", "Value"], ["Matched 10 m points", f"{len(delta_df):,}"]]
        rows.extend([[title, percentage_cell(value, is_mpd)] for title, value in
                     zip(titles, comparison_values(delta_df, primary_col, pre_col, is_mpd))])
        story.extend([Spacer(1, 6), make_table(rows, [90 * mm, 76 * mm]), Spacer(1, 8)])
        undefined = int((delta_df[pre_col] == 0).sum())
        story.append(Paragraph(f"Zero pre values: {undefined:,}. Their percentages are unavailable and omitted "
                               "from 10 m percentage summaries. Section percentages use section means.", styles["Small"]))
        plt = _load_pyplot()
        fig, ax = plt.subplots(figsize=(7.3, 2.9))
        valid = table.dropna(subset=["percentage"])
        x_col = "start_m" if section_m == 100 else "chainage"
        positive, negative = ("#2563eb", "#a16207") if is_mpd else ("#15803d", "#dc2626")
        ax.bar(valid[x_col], valid["percentage"], width=section_m * 0.85,
               color=np.where(valid["percentage"] >= 0, positive, negative))
        ax.axhline(0, color="#737373", lw=0.7)
        ax.set_xlabel("Chainage (m)", fontsize=8)
        ax.set_ylabel(f"{measure.capitalize()} (%)", fontsize=8)
        ax.set_title(f"{label} {measure} ({section_m} m)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#e5e7eb", lw=0.5)
        ax.set_axisbelow(True)
        story.extend([_png_flowable(_fig_to_png_bytes(fig), 176, 76), Spacer(1, 8)])
        story.append(Paragraph(f"All {len(table):,} rows at {section_m} m resolution. "
                               "100 m sections use matched-pair means in [start, end) bins. "
                               "Matched count shows coverage, including partial sections.", styles["Small"]))
        location_cols = ["start_m", "end_m"] if section_m == 100 else ["chainage"]
        rows = [["Start (m)", "End (m)"] if section_m == 100 else ["Chainage (m)"]]
        rows[0] += ["Primary (mm)", "Pre (mm)", "Matched count", f"{measure.capitalize()} (%)"]
        for _, row in table.iterrows():
            rows.append([f"{row[c]:,.1f}" for c in location_cols] +
                        [f"{row[primary_col]:.3f}", f"{row[pre_col]:.3f}", str(int(row["matched_count"])),
                         percentage_cell(row["percentage"], is_mpd)])
        story.append(make_table(rows, [doc.width / len(rows[0])] * len(rows[0]), font_size=7))

    primary_name = primary.metadata.get("survey") or primary.metadata.get("file_name") or "Primary survey"
    comparison_name = comparison.metadata.get("survey") or comparison.metadata.get("file_name") or "Comparison/Pre survey"
    comp_ride = _apply_chainage_offset(comparison.ride_10m, offset_m)
    comp_mpd = _apply_chainage_offset(comparison.mpd_10m, offset_m)
    primary_ukri = _combined_ukri_chart_data(primary.ride_10m, ride_tracks)
    comparison_ukri = _combined_ukri_chart_data(comp_ride, ride_tracks)
    primary_mpd = _combined_mpd_chart_data(primary.mpd_10m, mpd_lines)
    comparison_mpd = _combined_mpd_chart_data(comp_mpd, mpd_lines)
    ukri_delta = _comparison_delta(primary_ukri, comparison_ukri, "combined_ukri", "primary_ukri", "comparison_pre_ukri")
    mpd_delta = _comparison_delta(primary_mpd, comparison_mpd, "combined_mpd_mm", "primary_mpd_mm", "comparison_pre_mpd_mm")

    story = [
        Paragraph("National Highways Comparison Report", styles["Title"]),
        Paragraph(f"{escape(str(primary_name))} vs {escape(str(comparison_name))}", styles["Heading2"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Small"]),
        Spacer(1, 8),
    ]
    overview_rows = [
        ["Measure", "Primary", "Comparison/Pre"],
        ["File type", primary.file_type, comparison.file_type],
        ["Survey date", primary.metadata.get("survey_date", "-"), comparison.metadata.get("survey_date", "-")],
        ["Length", _format_m(primary.metadata.get("survey_length_m")), _format_m(comparison.metadata.get("survey_length_m"))],
        ["Ride rows", f"{len(primary.ride_10m):,}", f"{len(comparison.ride_10m):,}"],
        ["MPD rows", f"{len(primary.mpd_10m):,}", f"{len(comparison.mpd_10m):,}"],
        ["Chainage offset applied", "0.0 m", f"{offset_m:,.1f} m"],
    ]
    story.extend([make_table(overview_rows, [50 * mm, 58 * mm, 58 * mm]), Spacer(1, 10)])

    check_rows = _route_location_checks(primary, comparison)
    if check_rows:
        rows = [["Check", "Primary", "Comparison/Pre", "Difference", "Status"]]
        rows.extend([[row["check"], row["primary"], row["comparison"], row["difference"], _display_status(row["status"])] for row in check_rows])
        story.extend([Paragraph("Route Checks", styles["Heading2"]), make_table(rows, [42 * mm, 38 * mm, 38 * mm, 30 * mm, 18 * mm]), Spacer(1, 8)])

    matched_rows = [["Metric", "Matched selection", "Matched points", "Overall (%)", "Mean 10 m (%)"]]
    for label, delta, primary_col, pre_col, selection in [
        ("UKRI improvement", ukri_delta, "primary_ukri", "comparison_pre_ukri", ride_tracks),
        ("MPD difference", mpd_delta, "primary_mpd_mm", "comparison_pre_mpd_mm", mpd_lines),
    ]:
        is_mpd = label.startswith("MPD")
        values = comparison_values(delta, primary_col, pre_col, is_mpd)
        matched_rows.append([label, ", ".join(selection) or "No matching selection", f"{len(delta):,}",
                             percentage_cell(values[0], is_mpd), percentage_cell(values[1], is_mpd)])
    story.extend([Paragraph("Comparison Summary", styles["Heading2"]),
                  make_table(matched_rows, [32 * mm, 54 * mm, 28 * mm, 26 * mm, 26 * mm])])

    ukri_chart = _comparison_chart_png(primary_ukri, comparison_ukri, "combined_ukri", "UKRI (mm)", "Combined UKRI Comparison", exclusions)
    mpd_chart = _comparison_chart_png(primary_mpd, comparison_mpd, "combined_mpd_mm", "MPD (mm)", "Combined MPD Comparison", exclusions)
    if ukri_chart is not None or mpd_chart is not None:
        story.extend([Spacer(1, 10), Paragraph("Comparison Charts", styles["Heading2"])])
        if ukri_chart is not None:
            story.extend([_png_flowable(ukri_chart, 176, 82), Spacer(1, 6)])
        if mpd_chart is not None:
            story.extend([_png_flowable(mpd_chart, 176, 82), Spacer(1, 6)])

    add_percentage_section("UKRI", ukri_delta, "primary_ukri", "comparison_pre_ukri")
    add_percentage_section("MPD", mpd_delta, "primary_mpd_mm", "comparison_pre_mpd_mm")

    def _draw_page_furniture(canvas, current_doc):
        canvas.saveState()
        if HDS_LOGO_LIGHT.exists():
            logo_width = 30 * mm
            logo_height = logo_width * 179 / 600
            canvas.drawImage(
                str(HDS_LOGO_LIGHT),
                A4[0] - 14 * mm - logo_width,
                A4[1] - 10 * mm - logo_height,
                width=logo_width,
                height=logo_height,
                mask="auto",
                preserveAspectRatio=True,
            )
        canvas.setStrokeColor(colors.HexColor("#e5e7eb"))
        canvas.line(14 * mm, 11 * mm, A4[0] - 14 * mm, 11 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawString(14 * mm, 8 * mm, "National Highways Comparison Report")
        canvas.drawRightString(A4[0] - 14 * mm, 8 * mm, f"Page {current_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_page_furniture, onLaterPages=_draw_page_furniture)
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


def _selected_map_style() -> str:
    if st.session_state.get("survey_map_style", "Dark") == "Satellite":
        style = {
            "version": 8,
            "sources": {
                "satellite": {
                    "type": "raster",
                    "tiles": [
                        "https://services.arcgisonline.com/ArcGIS/rest/services/"
                        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ],
                    "tileSize": 256,
                    "maxzoom": 19,
                    "attribution": "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
                }
            },
            "layers": [{"id": "satellite", "type": "raster", "source": "satellite"}],
        }
        # Streamlit expects a string and calls indexOf on mapStyle. A data URL
        # lets the map renderer load our raster style without passing a dict.
        encoded_style = base64.b64encode(json.dumps(style).encode("utf-8")).decode("ascii")
        return f"data:application/json;base64,{encoded_style}"
    return "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


def _map_route_segments(geometry_geo: pd.DataFrame, label: str) -> list[dict]:
    rows = geometry_geo.to_dict("records")
    segments = []
    for start, end in zip(rows, rows[1:]):
        if any(pd.isna(row[axis]) for row in (start, end) for axis in ("lon", "lat")):
            continue
        segments.append({
            "path": [[start["lon"], start["lat"]], [end["lon"], end["lat"]]],
            "point": f"{label} (segment start)",
            "chainage": start["chainage"],
            "x": start["x"],
            "y": start["y"],
        })
    return segments


def _survey_map(geometry_geo: pd.DataFrame, height: int = 360, selected_chainage: float | None = None):
    if geometry_geo.empty:
        st.info("No geometry coordinates were found for mapping.")
        return
    midpoint = geometry_geo[["lat", "lon"]].mean()
    endpoints = geometry_geo.iloc[[0, -1]].copy()
    endpoints["point"] = ["Start", "End"]
    endpoints["color"] = [[34, 197, 94, 230], [239, 68, 68, 230]]
    layers = [
        pdk.Layer(
            "PathLayer",
            data=_map_route_segments(geometry_geo, "Survey route"),
            get_path="path",
            get_width=5,
            get_color=[25, 118, 210],
            width_min_pixels=3,
            width_max_pixels=5,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=endpoints,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=18,
            radius_min_pixels=5,
            radius_max_pixels=7,
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
                    radius_max_pixels=10,
                    pickable=True,
                )
            )
    st.pydeck_chart(
        pdk.Deck(
            map_style=_selected_map_style(),
            map_provider="carto",
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


def _comparison_map(primary_geo: pd.DataFrame, comparison_geo: pd.DataFrame, alignment: dict | None, height: int = 420):
    if primary_geo.empty or comparison_geo.empty:
        st.info("Both surveys need geometry coordinates to show the alignment map.")
        return
    all_points = pd.concat([primary_geo[["lat", "lon"]], comparison_geo[["lat", "lon"]]]).dropna()
    midpoint = all_points.mean()
    markers = []
    for label, frame, color in [
        ("Primary start", primary_geo, [25, 118, 210, 240]),
        ("Comparison start", comparison_geo, [245, 158, 11, 240]),
    ]:
        row = frame.sort_values("chainage").iloc[0]
        markers.append({**row.to_dict(), "point": label, "color": color})
    if alignment and alignment.get("matched_row") is not None:
        matched = dict(alignment["matched_row"])
        if "lat" not in matched or "lon" not in matched:
            matched["lon"], matched["lat"] = BNG_TO_WGS84.transform(float(matched["x"]), float(matched["y"]))
        markers.append({**matched, "point": "GPS alignment point", "color": [147, 51, 234, 240]})
    layers = [
        pdk.Layer("PathLayer", data=_map_route_segments(primary_geo, "Primary"), get_path="path", get_width=6,
                  get_color=[25, 118, 210, 220], width_min_pixels=3, width_max_pixels=5, pickable=True),
        pdk.Layer("PathLayer", data=_map_route_segments(comparison_geo, "Comparison"), get_path="path", get_width=6,
                  get_color=[245, 158, 11, 220], width_min_pixels=3, width_max_pixels=5, pickable=True),
        pdk.Layer("ScatterplotLayer", data=markers, get_position="[lon, lat]", get_fill_color="color",
                  get_radius=22, radius_min_pixels=6, radius_max_pixels=8, pickable=True),
    ]
    st.pydeck_chart(
        pdk.Deck(
            map_style=_selected_map_style(),
            map_provider="carto",
            initial_view_state=pdk.ViewState(latitude=float(midpoint["lat"]), longitude=float(midpoint["lon"]), zoom=12, pitch=0),
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
    series_col = "Track statistic" if "Track statistic" in df.columns else None
    columns = [x, y] + ([series_col] if series_col else [])
    chart_df = _with_nearest_location(df[columns].dropna(), geometry_geo) if map_hover else df[columns].dropna()
    hover_cols = ["x", "y", "lat", "lon"] if map_hover and {"x", "y", "lat", "lon"}.issubset(chart_df.columns) else None
    labels = {y: "UKRI (mm)"} if y == "combined_ukri" or y.endswith("_ri") else {}
    fig = px.line(chart_df, x=x, y=y, color=series_col, title=title, hover_data=hover_cols, labels=labels)
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


def _combined_ukri_chart_data(ride_df: pd.DataFrame, track_columns: list[str], show_range: bool = False) -> pd.DataFrame:
    if ride_df.empty or not track_columns:
        return pd.DataFrame()
    existing_columns = [column for column in track_columns if column in ride_df.columns]
    if not existing_columns:
        return pd.DataFrame()
    chart_df = ride_df[["chainage"] + existing_columns].copy()
    values = chart_df[existing_columns].replace(0, np.nan)
    if show_range:
        chart_df["Minimum"] = values.min(axis=1)
        chart_df["Maximum"] = values.max(axis=1)
        return chart_df[["chainage", "Minimum", "Maximum"]].melt(
            id_vars="chainage", var_name="Track statistic", value_name="combined_ukri"
        ).dropna(subset=["combined_ukri"])
    chart_df["combined_ukri"] = values.mean(axis=1)
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


def _excluded_mask(chainage: pd.Series, exclusions: list[tuple[float, float]], length_m: float = 10.0) -> pd.Series:
    if chainage.empty or not exclusions:
        return pd.Series(False, index=chainage.index)
    starts = chainage.to_numpy(dtype=float)
    ends = starts + length_m
    excluded = np.zeros(len(chainage), dtype=bool)
    for start, end in exclusions:
        excluded |= (starts < end) & (ends > start)
    return pd.Series(excluded, index=chainage.index)


def _ukri_10m_results(ride_df: pd.DataFrame, track_columns: list[str], exclusions: list[tuple[float, float]]) -> pd.DataFrame:
    if ride_df.empty or not track_columns:
        return pd.DataFrame()
    existing_columns = [column for column in track_columns if column in ride_df.columns]
    if not existing_columns:
        return pd.DataFrame()
    out = ride_df[["chainage"] + existing_columns].copy()
    out[existing_columns] = out[existing_columns].replace(0, np.nan)
    out["combined_ukri"] = out[existing_columns].mean(axis=1)
    out["valid_track_values"] = out[existing_columns].count(axis=1)
    out["tracks"] = ", ".join(existing_columns)
    out["excluded"] = _excluded_mask(out["chainage"], exclusions)
    out["section_start_300m"] = np.floor(out["chainage"] / 300.0) * 300.0
    return out[
        ["chainage", "section_start_300m", "excluded", "tracks", "valid_track_values", "combined_ukri"]
        + existing_columns
    ].sort_values("chainage")


def _mpd_10m_results(mpd_df: pd.DataFrame, line_columns: list[str], exclusions: list[tuple[float, float]]) -> pd.DataFrame:
    if mpd_df.empty:
        return pd.DataFrame()
    df = mpd_df.copy()
    if line_columns and "line" in df.columns:
        df = df[df["line"].isin(line_columns)]
    df = df[df["mpd_mm"] > 0.001]
    if df.empty:
        return pd.DataFrame()
    pivot = (
        df.pivot_table(index="chainage", columns="line", values="mpd_mm", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    track_columns = [column for column in pivot.columns if column != "chainage"]
    rename_map = {column: f"mpd_{column}_mm" for column in track_columns}
    out = pivot.rename(columns=rename_map)
    mpd_columns = list(rename_map.values())
    out["combined_mpd_mm"] = out[mpd_columns].mean(axis=1)
    out["valid_track_values"] = out[mpd_columns].count(axis=1)
    out["tracks"] = ", ".join(track_columns)
    out["excluded"] = _excluded_mask(out["chainage"], exclusions)
    out["section_start_100m"] = np.floor(out["chainage"] / 100.0) * 100.0
    return out[
        ["chainage", "section_start_100m", "excluded", "tracks", "valid_track_values", "combined_mpd_mm"]
        + mpd_columns
    ].sort_values("chainage")


def _csv_10m_bundle_bytes(
    ride_df: pd.DataFrame,
    ride_tracks: list[str],
    mpd_df: pd.DataFrame,
    mpd_lines: list[str],
    exclusions: list[tuple[float, float]],
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "combined_ukri_10m_results.csv",
            nh_parser.dataframe_to_csv(_ukri_10m_results(ride_df, ride_tracks, exclusions)),
        )
        archive.writestr(
            "combined_mpd_10m_results.csv",
            nh_parser.dataframe_to_csv(_mpd_10m_results(mpd_df, mpd_lines, exclusions)),
        )
    return buffer.getvalue()


def _endpoint_distance(summary_a: pd.DataFrame, summary_b: pd.DataFrame, index: int) -> float | None:
    if summary_a.empty or summary_b.empty or not {"x", "y"}.issubset(summary_a.columns) or not {"x", "y"}.issubset(summary_b.columns):
        return None
    row_a = summary_a.iloc[index]
    row_b = summary_b.iloc[index]
    return float(np.hypot(float(row_a["x"]) - float(row_b["x"]), float(row_a["y"]) - float(row_b["y"])))


def _gps_chainage_alignment(primary_geometry: pd.DataFrame, comparison_geometry: pd.DataFrame) -> dict | None:
    """Estimate comparison offset by matching the start of either run onto the other route."""
    required = {"chainage", "x", "y"}
    if (primary_geometry.empty or comparison_geometry.empty
            or not required.issubset(primary_geometry.columns)
            or not required.issubset(comparison_geometry.columns)):
        return None
    primary = primary_geometry.dropna(subset=list(required)).sort_values("chainage")
    comparison = comparison_geometry.dropna(subset=list(required)).sort_values("chainage")
    if primary.empty or comparison.empty:
        return None

    def nearest(start: pd.Series, route: pd.DataFrame) -> tuple[pd.Series, float, int]:
        distances = np.hypot(route["x"] - float(start["x"]), route["y"] - float(start["y"]))
        position = int(np.argmin(distances.to_numpy()))
        return route.iloc[position], float(distances.iloc[position]), position

    primary_start, comparison_start = primary.iloc[0], comparison.iloc[0]
    comparison_match, residual_a, comparison_index = nearest(primary_start, comparison)
    primary_match, residual_b, primary_index = nearest(comparison_start, primary)
    candidates = [
        {"offset_m": float(primary_start["chainage"] - comparison_match["chainage"]), "residual_m": residual_a,
         "anchor": "Primary start matched on comparison route", "matched_row": comparison_match.to_dict(),
         "primary_index": 0, "comparison_index": comparison_index},
        {"offset_m": float(primary_match["chainage"] - comparison_start["chainage"]), "residual_m": residual_b,
         "anchor": "Comparison start matched on primary route", "matched_row": primary_match.to_dict(),
         "primary_index": primary_index, "comparison_index": 0},
    ]
    result = min(candidates, key=lambda item: item["residual_m"])
    result["direction_ok"] = None
    if len(primary) > 1 and len(comparison) > 1:
        def tangent(route: pd.DataFrame, index: int) -> np.ndarray:
            before, after = max(0, index - 5), min(len(route) - 1, index + 5)
            return route.iloc[after][["x", "y"]].to_numpy(dtype=float) - route.iloc[before][["x", "y"]].to_numpy(dtype=float)

        vector_a = tangent(primary, result["primary_index"])
        vector_b = tangent(comparison, result["comparison_index"])
        if np.linalg.norm(vector_a) > 0 and np.linalg.norm(vector_b) > 0:
            result["direction_ok"] = bool(np.dot(vector_a, vector_b) >= 0)
    result.pop("primary_index")
    result.pop("comparison_index")
    return result


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


def _comparison_sections(delta, primary_col, pre_col, section_m):
    """Average matched pairs together before calculating change from pre."""
    result = delta[["chainage", primary_col, pre_col]].copy()
    if section_m == 100:
        result["start_m"] = np.floor(result["chainage"] / 100) * 100
        result = result.groupby("start_m", as_index=False).agg(
            **{primary_col: (primary_col, "mean"), pre_col: (pre_col, "mean"),
               "matched_count": ("chainage", "size")}
        )
        result.insert(1, "end_m", result["start_m"] + 100)
    else:
        result["matched_count"] = 1
    result["change_mm"] = result[primary_col] - result[pre_col]
    result["change_pct"] = result["change_mm"].div(result[pre_col].where(result[pre_col] != 0)) * 100
    return result


def _improvement_percent(primary, pre, higher_is_better):
    direction = 1 if higher_is_better else -1
    return direction * (primary - pre).div(pre.where(pre != 0)) * 100


def _improvement_metric(column, title, value, positive_colour="#15803d", negative_colour="#dc2626"):
    colour = positive_colour if pd.notna(value) and value > 0 else negative_colour if pd.notna(value) and value < 0 else "#737373"
    text = f"{value:+.2f}%" if pd.notna(value) else "N/A"
    column.markdown(
        f'<div>{escape(title)}</div><div style="font-size:2rem;color:{colour};font-weight:600">{text}</div>',
        unsafe_allow_html=True,
    )


def _show_comparison_analysis(delta, primary_col, pre_col, label, resolution):
    if delta.empty:
        st.info(f"No matched {label} points were found within 5 m.")
        return
    span = float(delta["chainage"].max() - delta["chainage"].min() + 10)
    section_m = 100 if resolution == "100 m" or (resolution == "Automatic" and span > 1000) else 10
    table = _comparison_sections(delta, primary_col, pre_col, section_m)
    raw = _comparison_sections(delta, primary_col, pre_col, 10)
    is_mpd = label == "MPD"
    measure = "difference" if is_mpd else "improvement"
    pct_col = f"{measure}_pct"
    positive_colour, negative_colour = ("#2563eb", "#facc15") if is_mpd else ("#15803d", "#dc2626")
    positive_label, negative_label = ("Positive difference", "Negative difference") if is_mpd else ("Improved", "Worse")
    for frame in (table, raw):
        frame[pct_col] = _improvement_percent(frame[primary_col], frame[pre_col], is_mpd)
        frame.drop(columns=["change_mm", "change_pct"], inplace=True)
    overall_pct = _improvement_percent(
        pd.Series([raw[primary_col].mean()]), pd.Series([raw[pre_col].mean()]), is_mpd
    ).iloc[0]
    st.metric("Matched 10 m points", f"{len(raw):,}")
    cols = st.columns(4)
    titles = [f"Overall {measure} (%)", f"Mean 10 m {measure} (%)",
              f"{'Maximum' if is_mpd else 'Best'} 10 m {measure} (%)",
              f"{'Minimum' if is_mpd else 'Worst'} 10 m {measure} (%)"]
    values = [overall_pct, raw[pct_col].mean(), raw[pct_col].max(), raw[pct_col].min()]
    for column, title, value in zip(cols, titles, values):
        _improvement_metric(column, title, value, positive_colour, negative_colour)
    if is_mpd:
        st.caption("MPD difference = 100 x (primary - pre) / pre. "
                   "Positive differences (higher MPD) are blue; negative differences (lower MPD) are yellow. "
                   "Overall difference compares matched survey means; the other summaries use 10 m percentages. "
                   "All matched points are included, including exclusions.")
    else:
        st.caption("Lower UKRI is treated as improvement: 100 x (pre - primary) / pre. "
                   "Positive improvement is green; negative improvement (worse than pre) is red. "
                   "Overall improvement compares matched survey means; the other summaries use 10 m percentages. "
                   "All matched points are included, including exclusions.")
    undefined = int(raw[pct_col].isna().sum())
    if undefined:
        st.caption(f"{undefined:,} points have a zero pre value; their {measure} percentage is unavailable and omitted from percentage summaries.")
    x_col = "start_m" if section_m == 100 else "chainage"
    chart = table.dropna(subset=[pct_col]).copy()
    chart["Outcome"] = np.select(
        [chart[pct_col] > 0, chart[pct_col] < 0],
        [positive_label, negative_label], default="Unchanged"
    )
    fig = px.bar(chart, x=x_col, y=pct_col, color="Outcome",
                 color_discrete_map={positive_label: positive_colour, negative_label: negative_colour, "Unchanged": "#737373"},
                 title=f"{label} {measure} ({section_m} m)",
                 labels={x_col: "Chainage (m)", pct_col: f"{measure.capitalize()} (%)"},
                 hover_data=["matched_count"])
    fig.update_traces(width=section_m * 0.85)
    fig.add_hline(y=0, line_color="#737373")
    fig.update_layout(barmode="overlay", height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Showing all {len(table):,} rows at {section_m} m resolution. "
               "100 m sections use matched-pair means in [start, end) chainage bins; "
               "matched_count shows coverage, including partial sections.")
    st.dataframe(table, use_container_width=True, hide_index=True,
                 column_config={pct_col: st.column_config.NumberColumn(f"{measure.capitalize()} (%)", format="%.2f%%")})
    st.download_button(f"Download {label} table CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"{label.lower()}_comparison_{section_m}m.csv", mime="text/csv",
                       key=f"{label}_comparison_csv")


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


#st.set_page_config(page_title="NH Ride and MPD Evaluator", layout="wide")

st.set_page_config(
    page_title="NH Ride and MPD Evaluator",
    page_icon="../favicon.ico",
    layout="wide",
)


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
    st.radio(
        "Map style",
        ["Dark", "Satellite"],
        key="survey_map_style",
        horizontal=True,
        help="Applies to all survey location and comparison maps.",
    )
    map_hover = st.toggle(
        "Map coordinates on charts",
        value=True,
        help="Adds nearest survey coordinates to chart hover tooltips and shows a map marker controlled by a chainage slider.",
    )
    st.divider()
    uploaded = st.file_uploader("Load BCD or RCD", type=["bcd", "rcd", "txt"])
    comparison_uploaded = st.file_uploader(
        "Optional comparison BCD or RCD (Pre Survey)",
        type=["bcd", "rcd", "txt"],
        help="Load a second survey for pre/post or repeat-run comparison.",
    )
    st.caption("RCD files are preferred as they contain exclusions and structure data. BCD files include derived ride/MPD values.")

if not uploaded:
    st.info("Choose one of the example BCD files to see compliant/non compliant sections and MPD track checks.")
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
    st.warning(f"Select {' and '.join(missing_specs)} in the sidebar to run compliance checks and exports.")
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
        help="Combined UKRI compliance is calculated from all selected track values in each 300 m section.",
    )
    if not selected_ukri_tracks:
        st.warning("Select at least one UKRI track in the sidebar to run UKRI compliance checks.")
        st.stop()
available_mpd_lines = _mpd_line_options(survey.mpd_10m)
selected_mpd_lines = available_mpd_lines
if available_mpd_lines:
    selected_mpd_lines = st.sidebar.multiselect(
        "MPD tracks for calculation",
        available_mpd_lines,
        default=available_mpd_lines,
        help="Combined MPD compliance is calculated from all selected track values in each 100 m section.",
    )
    if not selected_mpd_lines:
        st.warning("Select at least one MPD track in the sidebar to run MPD compliance checks.")
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
        export_col1, export_col2, export_col3 = st.columns(3)
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
                    geometry_geo,
                    selected_ukri_tracks,
                    selected_mpd_lines,
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
        with export_col3:
            st.download_button(
                "Download CSV results - 10m",
                data=_csv_10m_bundle_bytes(
                    survey.ride_10m,
                    selected_ukri_tracks,
                    survey.mpd_10m,
                    selected_mpd_lines,
                    exclusions,
                ),
                file_name=f"{export_prefix}_10m_results.zip",
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
            "RI/MPD compliance checks need a BCD or a completed derived-value algorithm."
        )

    st.markdown("**Survey Location**")
    _survey_map(geometry_geo)
    if exclusions:
        st.caption(f"{len(exclusions)} excluded region(s) removed from UKRI and MPD compliance calculations.")

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
        show_average_ukri = st.toggle(
            "Show average UKRI", value=False,
            help="Off shows the minimum and maximum of the selected tracks at each chainage. On shows their average."
        )
        combined_ride = _combined_ukri_chart_data(survey.ride_10m, side_options, show_range=not show_average_ukri)
        metric = "combined_ukri"
        ride_results = nh_parser.evaluate_ride_combined(survey.ride_10m, side_options, ride_spec, exclusions)
        chart_data = combined_ride
        chart_y = "combined_ukri"
        chart_title = "Combined UKRI by chainage - " + ("average" if show_average_ukri else "minimum and maximum")
        marker_key = "ride_marker_combined_ukri"

        pass_count = int((ride_results["status"] == "PASS").sum()) if not ride_results.empty else 0
        fail_count = int((ride_results["status"] == "FAIL").sum()) if not ride_results.empty else 0
        r1, r2, r3 = st.columns(3)
        r1.metric("Assessment lengths", len(ride_results))
        r2.metric("Compliant", pass_count)
        r3.metric("Non Compliant", fail_count)

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
        m2.metric("Compliant", pass_count)
        m3.metric("Non Compliant", fail_count)

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
        gps_alignment = _gps_chainage_alignment(survey.geometry, comparison_survey.geometry)
        auto_offset = float(gps_alignment["offset_m"]) if gps_alignment else 0.0
        alignment_mode = st.radio(
            "Comparison alignment", ["GPS start alignment", "Manual offset", "No offset"], horizontal=True,
            index=0 if gps_alignment else 1,
            help="GPS alignment matches the start of the shorter/overlapping survey to the other route.",
        )
        if alignment_mode == "Manual offset":
            offset_m = float(st.number_input(
                "Comparison chainage offset (m)", value=round(auto_offset, 1), step=1.0, format="%.1f",
                help="Positive values move comparison data forwards along primary chainage; negative values move it backwards. There is no 500 m limit.",
            ))
        elif alignment_mode == "GPS start alignment":
            offset_m = auto_offset
        else:
            offset_m = 0.0

        if gps_alignment:
            residual = float(gps_alignment["residual_m"])
            st.caption(f"GPS suggested offset: {auto_offset:+,.1f} m · nearest-route error: {residual:,.1f} m · {gps_alignment['anchor']}")
            if residual > 20.0:
                st.warning(f"The survey GPS traces do not align reliably (nearest-route error {residual:,.1f} m). Check the map and use a manual offset only if these are known to be the same route.")
            if gps_alignment.get("direction_ok") is False:
                st.warning("The surveys appear to run in opposite directions, so chainage comparison may not be valid.")
        else:
            st.warning("GPS alignment is unavailable because one or both surveys have no usable geometry coordinates.")

        st.markdown("**GPS Alignment Map**")
        st.caption("Primary route is blue; comparison route is orange; the selected GPS anchor is purple.")
        _comparison_map(geometry_geo, comparison_geometry_geo, gps_alignment)

        if not common_ukri_tracks and not common_mpd_lines:
            st.warning("No matching UKRI tracks or MPD tracks were found between the two datasets.")

        comp_ride = _apply_chainage_offset(comparison_survey.ride_10m, offset_m)
        comp_mpd = _apply_chainage_offset(comparison_survey.mpd_10m, offset_m)

        comparison_resolution = st.radio(
            "Comparison detail resolution", ["Automatic", "10 m", "100 m"], horizontal=True,
            help="Automatic uses 100 m sections when the matched chainage span exceeds 1,000 m."
        )

        try:
            comparison_report_name = survey.metadata.get("survey") or uploaded.name.rsplit(".", 1)[0]
            safe_comparison_report_name = _safe_filename(comparison_report_name, export_prefix)
            st.download_button(
                "Download comparison PDF report",
                data=_pdf_comparison_report_bytes(
                    survey,
                    comparison_survey,
                    exclusions,
                    common_ukri_tracks,
                    common_mpd_lines,
                    offset_m,
                    comparison_resolution,
                ),
                file_name=f"{safe_comparison_report_name}_comparison_report.pdf",
                mime="application/pdf",
            )
        except ModuleNotFoundError:
            st.warning("PDF export needs the reportlab package. Run `pip install -r requirements.txt` and restart the app.")

        if common_ukri_tracks:
            st.markdown("**Combined UKRI Comparison**")
            primary_ukri = _combined_ukri_chart_data(survey.ride_10m, common_ukri_tracks)
            comparison_ukri = _combined_ukri_chart_data(comp_ride, common_ukri_tracks)
            comparison_chart = pd.concat(
                [
                    comparison_ukri.assign(dataset="Comparison/Pre"),
                    primary_ukri.assign(dataset="Primary"),
                ],
                ignore_index=True,
            )
            if not comparison_chart.empty:
                fig = px.line(
                    comparison_chart,
                    x="chainage",
                    y="combined_ukri",
                    color="dataset",
                    color_discrete_map={"Primary": "#22c55e", "Comparison/Pre": "#636efa"},
                    title="Combined UKRI comparison",
                    labels={"combined_ukri": "UKRI (mm)"},
                )
                for start, end in exclusions:
                    fig.add_vrect(x0=start, x1=end, fillcolor="rgba(239, 68, 68, 0.14)", line_width=0)
                fig.update_traces(opacity=0.75)
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)

            ukri_delta = _comparison_delta(primary_ukri, comparison_ukri, "combined_ukri", "primary_ukri", "comparison_ukri")
            _show_comparison_analysis(ukri_delta, "primary_ukri", "comparison_ukri", "UKRI", comparison_resolution)
        elif not survey.ride_10m.empty or not comparison_survey.ride_10m.empty:
            st.info("No matching UKRI tracks were found for comparison.")

        if common_mpd_lines:
            st.markdown("**Combined MPD Comparison**")
            primary_mpd = _combined_mpd_chart_data(survey.mpd_10m, common_mpd_lines)
            comparison_mpd = _combined_mpd_chart_data(comp_mpd, common_mpd_lines)
            comparison_chart = pd.concat(
                [
                    comparison_mpd.assign(dataset="Comparison/Pre"),
                    primary_mpd.assign(dataset="Primary"),
                ],
                ignore_index=True,
            )
            if not comparison_chart.empty:
                fig = px.line(
                    comparison_chart,
                    x="chainage",
                    y="combined_mpd_mm",
                    color="dataset",
                    color_discrete_map={"Primary": "#22c55e", "Comparison/Pre": "#636efa"},
                    title="Combined MPD comparison",
                )
                for start, end in exclusions:
                    fig.add_vrect(x0=start, x1=end, fillcolor="rgba(239, 68, 68, 0.14)", line_width=0)
                fig.update_traces(opacity=0.75)
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)

            mpd_delta = _comparison_delta(primary_mpd, comparison_mpd, "combined_mpd_mm", "primary_mpd_mm", "comparison_mpd_mm")
            _show_comparison_analysis(mpd_delta, "primary_mpd_mm", "comparison_mpd_mm", "MPD", comparison_resolution)
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
