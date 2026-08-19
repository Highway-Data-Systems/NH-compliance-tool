from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


PARSER_VERSION = "2026-08-19-bcd-ukri-elvp-edge-padding-v7"
MAX_REASONABLE_RI = 100.0
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
DATETIME_RE = re.compile(r"\d{1,2}-[A-Za-z]{3}-\d{4}\s*\d{1,2}:\d{2}")
MPD_RE = re.compile(
    r"^\s*(?P<chainage>\d+(?:\.\d+)?)(?P<line>[A-Z]{2,3})\s+"
    r"(?P<dropouts>\d+(?:\.\d+)?)\s+(?P<mpd>\d+(?:\.\d+)?)\s+"
    r"(?P<spikes>\d+(?:\.\d+)?)"
)


@dataclass
class ParsedSurvey:
    file_type: str
    metadata: dict
    quality_limits: pd.DataFrame
    geometry: pd.DataFrame
    ride_10m: pd.DataFrame
    mpd_10m: pd.DataFrame
    events: pd.DataFrame


def read_uploaded_text(uploaded_file) -> str:
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    return data.decode("ascii", errors="replace")


def parse_survey_text(text: str, name: str = "") -> ParsedSurvey:
    head = text[:32].upper()
    if head.startswith("BCD"):
        return parse_bcd(text, name)
    if head.startswith("SURFP"):
        return parse_rcd(text, name)
    raise ValueError("File does not look like a supported BCD or Surface Profile RCD file.")


def parse_bcd(text: str, name: str = "") -> ParsedSurvey:
    lines = text.splitlines()
    metadata = {
        "file_name": name,
        "format": lines[0].strip() if lines else "",
        "survey": lines[2].strip() if len(lines) > 2 else "",
        "system": lines[3].strip() if len(lines) > 3 else "",
    }
    timestamps = _extract_timestamps("\n".join(lines[:4]))
    if timestamps:
        metadata["survey_date"] = timestamps[0]
        if len(timestamps) > 1:
            metadata["survey_end_date"] = timestamps[1]
    if len(lines) > 1:
        nums = _floats(lines[1])
        if nums:
            metadata["survey_length_m"] = max((n for n in nums if 100.0 <= n <= 1_000_000.0), default=nums[-1])
    if len(lines) > 0:
        footer_nums = _floats(lines[-1])
        footer_lengths = [n for n in footer_nums if 100.0 <= n <= 1_000_000.0]
        if footer_lengths:
            metadata["route_length_m"] = max(footer_lengths)

    quality_rows = []
    for line in lines[12:60]:
        nums = _floats(line)
        if len(nums) == 1:
            label = line[: line.rfind(f"{nums[-1]:.6f}")].strip()
            if label:
                quality_rows.append({"check": label, "value": nums[-1]})

    geometry_rows = []
    ride_rows = []
    mpd_rows = []
    mode = "geometry"
    for line in lines[60:]:
        mpd_match = MPD_RE.match(line)
        if mpd_match:
            mode = "mpd"
            row = mpd_match.groupdict()
            mpd_rows.append(
                {
                    "chainage": float(row["chainage"]),
                    "line": row["line"],
                    "dropouts_pct": float(row["dropouts"]),
                    "mpd_mm": float(row["mpd"]),
                    "spikes_pct": float(row["spikes"]),
                }
            )
            continue

        nums = _floats(line)
        if mode == "geometry" and len(nums) == 4:
            geometry_rows.append(
                {"chainage": nums[0], "x": nums[1], "y": nums[2], "z": nums[3]}
            )
        elif _looks_like_bcd_ride_row(line, nums):
            mode = "ride"
            ns_ri = _roughness_index(nums[1], nums[2])
            os_ri = _roughness_index(nums[4], nums[5])
            ride_rows.append(
                {
                    "chainage": nums[0],
                    "ns_elpv3": nums[1],
                    "ns_elpv10": nums[2],
                    "ns_ri": ns_ri,
                    "ns_quality_code": nums[3],
                    "os_elpv3": nums[4],
                    "os_elpv10": nums[5],
                    "os_ri": os_ri,
                    "os_quality_code": nums[6],
                }
            )

    ride_df = _add_grouped_ride_columns(pd.DataFrame(ride_rows))
    mpd_df = pd.DataFrame(mpd_rows)
    events = _bcd_exclusion_events(lines, ride_df, mpd_df)
    return ParsedSurvey(
        file_type="BCD",
        metadata=metadata,
        quality_limits=pd.DataFrame(quality_rows),
        geometry=pd.DataFrame(geometry_rows),
        ride_10m=ride_df,
        mpd_10m=mpd_df,
        events=events,
    )


def parse_rcd(text: str, name: str = "") -> ParsedSurvey:
    lines = text.splitlines()
    metadata = {
        "file_name": name,
        "format": lines[0].strip() if lines else "",
        "survey": lines[1].strip() if len(lines) > 1 else "",
        "system": lines[2].strip() if len(lines) > 2 else "",
    }
    timestamps = _extract_timestamps("\n".join(lines[:3]))
    if timestamps:
        metadata["survey_date"] = timestamps[0]
        if len(timestamps) > 1:
            metadata["survey_end_date"] = timestamps[1]
    if len(lines) > 3:
        nums = _floats(lines[3])
        if len(nums) >= 7:
            metadata.update(
                {
                    "start_x": nums[0],
                    "start_y": nums[1],
                    "start_z": nums[2],
                    "survey_length_m": nums[3],
                    "end_x": nums[4],
                    "end_y": nums[5],
                    "end_z": nums[6],
                }
            )
    geom_interval = None
    lp_interval = None
    lp_line_count = 0
    texture_line_count = 0
    msd_interval = None
    texture_offsets = []
    if len(lines) > 4:
        layout = _parse_rcd_layout(lines[4])
        metadata.update(layout)
        geom_interval = layout.get("geometry_interval_m")
        lp_interval = layout.get("longitudinal_profile_interval_m")
        lp_line_count = int(layout.get("longitudinal_profile_lines") or 0)
        texture_line_count = int(layout.get("texture_profile_lines") or 0)
        msd_interval = layout.get("msd_interval_m")
    if len(lines) > 6:
        texture_offsets = _floats(lines[6])[:texture_line_count]

    events = []
    geometry = []
    msd_rows = []
    data_row_index = 0
    body_start = 7
    idx = body_start
    while idx < len(lines):
        line = lines[idx]
        event = _parse_event(line)
        if event:
            events.append(event)
            idx += 1
            continue
        geom = _parse_rcd_geometry(line)
        if geom:
            chainage = data_row_index * geom_interval if geom_interval else np.nan
            geometry.append(
                {
                    "chainage": chainage,
                    "x": geom["x"],
                    "y": geom["y"],
                    "z": geom["z"],
                    "speed_cm_s": geom["speed_cm_s"],
                    "speed_kmh": geom["speed_cm_s"] * 0.036,
                }
            )
            data_row_index += 1
            idx += 1
            continue
        break

    lp_record_count = 0
    if lp_interval and lp_line_count and metadata.get("survey_length_m"):
        points_per_line = int(np.ceil(float(metadata["survey_length_m"]) / lp_interval))
        records_per_line = int(np.ceil(points_per_line / 20.0))
        lp_record_count = records_per_line * lp_line_count
        idx += lp_record_count
    metadata["longitudinal_profile_records"] = lp_record_count
    ride_10m = pd.DataFrame()
    if lp_record_count:
        lp_profiles = _parse_lp_profiles(
            lines=lines,
            start_index=idx - lp_record_count,
            lp_records=lp_record_count,
            lp_line_count=lp_line_count,
            survey_length=float(metadata["survey_length_m"]),
            lp_interval=float(lp_interval),
        )
        lp_offsets = _floats(lines[5])[:lp_line_count] if len(lines) > 5 else []
        ride_10m = _derive_ride_from_profiles(lp_profiles, lp_offsets, float(lp_interval))

    msd_index = 0
    while idx < len(lines):
        line = lines[idx]
        if len(line) >= 120:
            chainage = msd_index * msd_interval if msd_interval else np.nan
            for line_index, row in enumerate(_parse_msd_record(line, texture_line_count), 1):
                offset = texture_offsets[line_index - 1] if line_index <= len(texture_offsets) else np.nan
                row.update(
                    {
                        "chainage": chainage,
                        "line": _line_label(line_index, offset),
                        "line_index": line_index,
                        "offset_m": offset,
                    }
                )
                msd_rows.append(row)
            msd_index += 1
        idx += 1

    msd_df = pd.DataFrame(msd_rows)
    mpd_10m = _aggregate_msd_to_mpd(msd_df)
    metadata["msd_records"] = msd_index

    return ParsedSurvey(
        file_type="RCD",
        metadata=metadata,
        quality_limits=pd.DataFrame(),
        geometry=pd.DataFrame(geometry),
        ride_10m=_add_grouped_ride_columns(ride_10m),
        mpd_10m=mpd_10m,
        events=pd.DataFrame(events),
    )


def exclusion_intervals(events: pd.DataFrame) -> list[tuple[float, float]]:
    if events.empty or not {"event", "chainage"}.issubset(events.columns):
        return []
    intervals = []
    start = None
    for row in events.sort_values("chainage").itertuples(index=False):
        event = str(row.event).strip().lower()
        if event == "s-exclude":
            start = float(row.chainage)
        elif event == "e-exclude" and start is not None:
            end = float(row.chainage)
            if end > start:
                intervals.append((start, end))
            start = None
    return intervals


def _apply_exclusions(df: pd.DataFrame, intervals: list[tuple[float, float]], length_m: float = 10.0) -> pd.DataFrame:
    if df.empty or not intervals:
        return df
    keep = np.ones(len(df), dtype=bool)
    starts = df["chainage"].to_numpy(dtype=float)
    ends = starts + length_m
    for start, end in intervals:
        keep &= ~((starts < end) & (ends > start))
    return df[keep].copy()


def _expected_10m_values_after_exclusions(
    section_start: float,
    section_length_m: float,
    track_count: int,
    intervals: list[tuple[float, float]],
) -> int:
    slots_per_track = int(round(section_length_m / 10.0))
    available_slots = 0
    for slot_index in range(slots_per_track):
        slot_start = float(section_start) + slot_index * 10.0
        slot_end = slot_start + 10.0
        is_excluded = any(slot_start < end and slot_end > start for start, end in intervals)
        if not is_excluded:
            available_slots += 1
    return max(0, available_slots * max(1, int(track_count)))


def evaluate_ride(
    ride_df: pd.DataFrame,
    metric_column: str,
    spec: dict,
    exclusions: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    if ride_df.empty or metric_column not in ride_df.columns:
        return pd.DataFrame()
    df = ride_df[["chainage", metric_column]].rename(columns={metric_column: "ri"})
    df = df[(df["ri"] > 0.001) & (df["ri"] <= MAX_REASONABLE_RI) & df["ri"].notna()].copy()
    df = _apply_exclusions(df, exclusions or [], 10.0)
    if df.empty:
        return pd.DataFrame()
    valid_start = 20.0
    valid_end = np.floor((df["chainage"].max() - 10.0) / 10.0) * 10.0
    final_start = np.floor(max(valid_end - 300.0, 0.0) / 300.0) * 300.0
    df = df[(df["chainage"] >= valid_start) & (df["chainage"] < valid_end)].copy()
    if df.empty:
        return pd.DataFrame()
    section_start = np.floor(df["chainage"] / 300.0) * 300.0
    df["section_start"] = np.where(section_start > final_start, final_start, section_start)
    rows = []
    for start, group in df.groupby("section_start"):
        end = valid_end if start == final_start else start + 300.0
        values = group["ri"]
        rows.append(
            {
                "section": f"{start:.0f}-{end:.0f} m",
                "start_m": start,
                "end_m": end,
                "valid_10m_values": int(values.count()),
                f"ri_lt_{spec['all_lt']}_count": int((values < spec["all_lt"]).sum()),
                f"ri_lt_{spec['pct80_lt']}_count": int((values < spec["pct80_lt"]).sum()),
                "max_ri": float(values.max()),
                "pct_below_lower_limit": float((values < spec["pct80_lt"]).mean() * 100.0),
                "all_below_limit": bool((values < spec["all_lt"]).all()),
                "pct80_requirement_met": bool((values < spec["pct80_lt"]).mean() >= 0.8),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["status"] = np.where(
            out["all_below_limit"] & out["pct80_requirement_met"], "PASS", "FAIL"
        )
    return out


def evaluate_ride_combined(
    ride_df: pd.DataFrame,
    metric_columns: list[str],
    spec: dict,
    exclusions: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    if ride_df.empty:
        return pd.DataFrame()
    metric_columns = [column for column in metric_columns if column in ride_df.columns]
    if not metric_columns:
        return pd.DataFrame()

    frames = []
    for column in metric_columns:
        track_df = ride_df[["chainage", column]].rename(columns={column: "ri"}).copy()
        track_df["track"] = column
        frames.append(track_df)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["ri"] > 0.001) & (df["ri"] <= MAX_REASONABLE_RI) & df["ri"].notna()].copy()
    df = _apply_exclusions(df, exclusions or [], 10.0)
    if df.empty:
        return pd.DataFrame()

    valid_start = 20.0
    valid_end = np.floor((df["chainage"].max() - 10.0) / 10.0) * 10.0
    final_start = np.floor(max(valid_end - 300.0, 0.0) / 300.0) * 300.0
    df = df[(df["chainage"] >= valid_start) & (df["chainage"] < valid_end)].copy()
    if df.empty:
        return pd.DataFrame()

    section_start = np.floor(df["chainage"] / 300.0) * 300.0
    df["section_start"] = np.where(section_start > final_start, final_start, section_start)
    rows = []
    for start, group in df.groupby("section_start"):
        end = valid_end if start == final_start else start + 300.0
        values = group["ri"]
        track_values = sorted(group["track"].dropna().unique())
        rows.append(
            {
                "metric": "combined_ukri",
                "tracks": ", ".join(track_values),
                "section": f"{start:.0f}-{end:.0f} m",
                "start_m": start,
                "end_m": end,
                "valid_10m_values": int(values.count()),
                f"ri_lt_{spec['all_lt']}_count": int((values < spec["all_lt"]).sum()),
                f"ri_lt_{spec['pct80_lt']}_count": int((values < spec["pct80_lt"]).sum()),
                "max_ri": float(values.max()),
                "pct_below_lower_limit": float((values < spec["pct80_lt"]).mean() * 100.0),
                "all_below_limit": bool((values < spec["all_lt"]).all()),
                "pct80_requirement_met": bool((values < spec["pct80_lt"]).mean() >= 0.8),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["status"] = np.where(
            out["all_below_limit"] & out["pct80_requirement_met"], "PASS", "FAIL"
        )
    return out


def evaluate_mpd(mpd_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    return evaluate_mpd_with_exclusions(mpd_df, spec, [])


def evaluate_mpd_with_exclusions(
    mpd_df: pd.DataFrame,
    spec: dict,
    exclusions: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    if mpd_df.empty:
        return pd.DataFrame()
    intervals = exclusions or []
    df = mpd_df[mpd_df["mpd_mm"] > 0.001].copy()
    df = _apply_exclusions(df, intervals, 10.0)
    df["section_start"] = np.floor(df["chainage"] / 100.0) * 100.0
    rows = []
    for (line, start), group in df.groupby(["line", "section_start"]):
        mpd = group["mpd_mm"]
        expected_values = _expected_10m_values_after_exclusions(start, 100.0, 1, intervals)
        valid_pct = 100.0 if expected_values == 0 else min(100.0, len(group) / expected_values * 100.0)
        avg = float(mpd.mean())
        std = float(mpd.std(ddof=0))
        rows.append(
            {
                "line": line,
                "section": f"{start:.0f}-{start + 100:.0f} m",
                "start_m": start,
                "end_m": start + 100.0,
                "valid_10m_values": int(len(group)),
                "expected_10m_values": int(expected_values),
                "valid_pct": valid_pct,
                "avg_mpd_mm": avg,
                "std_mpd_mm": std,
                "validity_met": valid_pct >= 50.0,
                "average_met": spec["avg_min"] <= avg <= spec["avg_max"],
                "std_met": std <= spec["std_max"],
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["status"] = np.where(
            out["validity_met"] & out["average_met"] & out["std_met"], "PASS", "FAIL"
        )
    return out


def evaluate_mpd_combined_with_exclusions(
    mpd_df: pd.DataFrame,
    spec: dict,
    exclusions: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    if mpd_df.empty:
        return pd.DataFrame()
    intervals = exclusions or []
    df = mpd_df[mpd_df["mpd_mm"] > 0.001].copy()
    df = _apply_exclusions(df, intervals, 10.0)
    if df.empty:
        return pd.DataFrame()
    df["section_start"] = np.floor(df["chainage"] / 100.0) * 100.0
    rows = []
    for start, group in df.groupby("section_start"):
        mpd = group["mpd_mm"]
        lines = sorted(str(line) for line in group["line"].dropna().unique()) if "line" in group.columns else []
        expected_values = _expected_10m_values_after_exclusions(start, 100.0, len(lines), intervals)
        valid_pct = 100.0 if expected_values == 0 else min(100.0, len(group) / expected_values * 100.0)
        avg = float(mpd.mean())
        std = float(mpd.std(ddof=0))
        rows.append(
            {
                "metric": "combined_mpd",
                "tracks": ", ".join(lines),
                "section": f"{start:.0f}-{start + 100:.0f} m",
                "start_m": start,
                "end_m": start + 100.0,
                "valid_10m_values": int(len(group)),
                "expected_10m_values": int(expected_values),
                "valid_pct": valid_pct,
                "avg_mpd_mm": avg,
                "std_mpd_mm": std,
                "validity_met": valid_pct >= 50.0,
                "average_met": spec["avg_min"] <= avg <= spec["avg_max"],
                "std_met": std <= spec["std_max"],
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["status"] = np.where(
            out["validity_met"] & out["average_met"] & out["std_met"], "PASS", "FAIL"
        )
    return out


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _floats(line: str) -> list[float]:
    return [float(x) for x in FLOAT_RE.findall(line)]


def _looks_like_bcd_ride_row(line: str, nums: list[float]) -> bool:
    return len(nums) == 7 and bool(re.match(r"^\s*\d+(?:\.\d+)?\s+", line))


def _bcd_exclusion_events(lines: list[str], ride_df: pd.DataFrame, mpd_df: pd.DataFrame) -> pd.DataFrame:
    intervals = []
    for line in lines:
        if "S-EXCLUDE" not in line.upper():
            continue
        nums = _floats(line)
        if len(nums) >= 2:
            start, end = nums[-2], nums[-1]
            if end > start:
                intervals.append((start, end, "BCD S-EXCLUDE"))

    for source, df in [("BCD ride gap", ride_df), ("BCD MPD gap", mpd_df)]:
        if df.empty or "chainage" not in df.columns:
            continue
        chainages = df["chainage"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
        for idx in range(1, len(chainages)):
            previous = float(chainages.iloc[idx - 1])
            current = float(chainages.iloc[idx])
            if current - previous > 15.0:
                start = previous + 10.0
                end = current
                if end > start:
                    intervals.append((start, end, source))

    deduped = []
    seen = set()
    for start, end, source in intervals:
        key = (round(float(start), 3), round(float(end), 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.extend(
            [
                {"event": "S-Exclude", "chainage": float(start), "source": source},
                {"event": "E-Exclude", "chainage": float(end), "source": source},
            ]
        )
    return pd.DataFrame(deduped)


def _extract_timestamps(text: str) -> list[str]:
    timestamps = []
    for match in DATETIME_RE.findall(text):
        normalized = re.sub(r"(\d{4})\s*(\d{1,2}:\d{2})", r"\1 \2", match)
        timestamps.append(re.sub(r"\s+", " ", normalized).strip())
    return timestamps


def _parse_event(line: str) -> dict | None:
    event_match = re.match(r"^\s*(?P<label>[A-Za-z][A-Za-z -]{1,22})\s+(?P<chainage>\d+(?:\.\d+)?)\s*$", line)
    if not event_match:
        return None
    return {
        "event": event_match.group("label").strip(),
        "chainage": float(event_match.group("chainage")),
    }


def _looks_like_geometry(nums: list[float]) -> bool:
    return len(nums) >= 4 and nums[0] > 10000 and nums[1] > 10000 and -1000 < nums[2] < 1000


def _parse_rcd_layout(line: str) -> dict:
    def to_int(value: str) -> int:
        return int(value.strip() or 0)

    def to_float(value: str) -> float:
        return float(value.strip() or 0)

    return {
        "location_markers": to_int(line[0:5]),
        "geometry_interval_m": to_float(line[5:17]),
        "longitudinal_profile_interval_m": to_float(line[17:29]),
        "longitudinal_profile_lines": to_int(line[29:31]),
        "texture_profile_interval_m": to_float(line[31:43]),
        "texture_profile_lines": to_int(line[43:45]),
        "msd_interval_m": to_float(line[45:57]),
        "texture_sensor_type": line[57:58].strip(),
        "texture_points_per_profile": to_int(line[58:62]),
    }


def _parse_rcd_geometry(line: str) -> dict | None:
    if len(line) < 35:
        return None
    try:
        x = float(line[0:11])
        y = float(line[11:22])
        z = float(line[22:31])
        speed = int(line[31:35])
    except ValueError:
        return None
    if x <= 10000 or y <= 10000 or not -1000 < z < 1000:
        return None
    return {"x": x, "y": y, "z": z, "speed_cm_s": speed}


def _parse_msd_record(line: str, texture_line_count: int) -> list[dict]:
    rows = []
    for line_index in range(texture_line_count):
        start = line_index * 12
        group = line[start : start + 12]
        if len(group) < 12:
            continue
        try:
            value = int(group[0:4])
            dropouts = float(group[4:8])
            spikes = float(group[8:12])
        except ValueError:
            continue
        if value <= 0:
            continue
        rows.append(
            {
                "msd_mm": value / 100.0,
                "dropouts_pct": dropouts,
                "spikes_pct": spikes,
            }
        )
    return rows


def _aggregate_msd_to_mpd(msd_df: pd.DataFrame) -> pd.DataFrame:
    if msd_df.empty:
        return pd.DataFrame()
    df = msd_df.copy()
    df["section_start"] = np.floor(df["chainage"] / 10.0) * 10.0
    rows = []
    for (line, line_index, offset, start), group in df.groupby(
        ["line", "line_index", "offset_m", "section_start"], dropna=False
    ):
        rows.append(
            {
                "chainage": float(start),
                "line": line,
                "line_index": int(line_index),
                "offset_m": float(offset) if pd.notna(offset) else np.nan,
                "dropouts_pct": float(group["dropouts_pct"].mean()),
                "mpd_mm": float(group["msd_mm"].mean()),
                "spikes_pct": float(group["spikes_pct"].mean()),
                "source": "RCD MSD 10 m mean",
            }
        )
    return pd.DataFrame(rows).sort_values(["line_index", "chainage"]).reset_index(drop=True)


def _parse_lp_profiles(
    lines: list[str],
    start_index: int,
    lp_records: int,
    lp_line_count: int,
    survey_length: float,
    lp_interval: float,
) -> list[np.ndarray]:
    if not lp_records or not lp_line_count:
        return []
    records_per_line = lp_records // lp_line_count
    point_count = int(np.ceil(survey_length / lp_interval))
    profiles = []
    idx = start_index
    for _line_index in range(lp_line_count):
        values = []
        for _record_index in range(records_per_line):
            if idx >= len(lines):
                break
            line = lines[idx]
            idx += 1
            if len(line) < 140:
                continue
            for start in range(0, 140, 7):
                try:
                    values.append(int(line[start : start + 7]))
                except ValueError:
                    values.append(0)
        arr = np.array(values[:point_count], dtype=float) / 10.0
        profiles.append(_sample_hold_invalid(arr))
    return profiles


def _derive_ride_from_profiles(
    profiles: list[np.ndarray],
    offsets: list[float],
    lp_interval: float,
) -> pd.DataFrame:
    if not profiles:
        return pd.DataFrame()
    hp_3m = _high_pass_coefficients(0.3333, lp_interval)
    hp_10m = _high_pass_coefficients(0.1, lp_interval)
    rows_by_chainage: dict[float, dict] = {}
    for line_index, profile in enumerate(profiles, 1):
        offset = offsets[line_index - 1] if line_index <= len(offsets) else np.nan
        label = _line_label(line_index, offset).lower()
        filtered_3m = _convolve_same_edge_padded(profile, hp_3m)
        filtered_10m = _convolve_same_edge_padded(profile, hp_10m)
        points_per_10m = max(1, int(round(10.0 / lp_interval)))
        for start in range(0, len(profile), points_per_10m):
            seg3 = filtered_3m[start : start + points_per_10m]
            seg10 = filtered_10m[start : start + points_per_10m]
            if len(seg3) < points_per_10m / 2:
                break
            chainage = round(start * lp_interval, 3)
            elpv3 = float(np.mean(seg3**2))
            elpv10 = float(np.mean(seg10**2))
            ri = _roughness_index(elpv3, elpv10)
            row = rows_by_chainage.setdefault(chainage, {"chainage": chainage})
            row[f"{label}_elpv3"] = elpv3
            row[f"{label}_elpv10"] = elpv10
            row[f"{label}_ri"] = ri
    return pd.DataFrame(rows_by_chainage.values()).sort_values("chainage").reset_index(drop=True)


def _add_grouped_ride_columns(ride_df: pd.DataFrame) -> pd.DataFrame:
    if ride_df.empty:
        return ride_df
    df = ride_df.copy()
    for column in [col for col in df.columns if col.endswith("_ri")]:
        df.loc[df[column] > MAX_REASONABLE_RI, column] = np.nan

    if "ns_ri" in df.columns:
        df["left_ri"] = df["ns_ri"]

    if "os_ri" in df.columns:
        df["right_ri"] = df["os_ri"]
    return df


def _high_pass_coefficients(frequency: float, delta: float, order: int = 3) -> np.ndarray:
    width = math.ceil(order / (frequency * delta))
    if width % 2:
        width += 1
    i = np.arange(-width // 2, width // 2 + 1)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * (i + width / 2) / width)
    x = 2 * np.pi * i * frequency * delta
    sinc = np.ones_like(x, dtype=float)
    non_zero = x != 0
    sinc[non_zero] = np.sin(x[non_zero]) / x[non_zero]
    low_pass = window * sinc
    low_pass = low_pass / low_pass.sum()
    high_pass = -low_pass
    high_pass[i == 0] = 1 - low_pass[i == 0]
    return high_pass


def _convolve_same_edge_padded(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    left_pad = len(kernel) // 2
    right_pad = len(kernel) - 1 - left_pad
    padded = np.pad(values, (left_pad, right_pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _roughness_index(elpv3: float, elpv10: float) -> float:
    return max((10.0 / 3.0 * elpv3) + math.sqrt(elpv10) - 0.1, 0.0)


def _sample_hold_invalid(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    out = values.copy()
    invalid = np.abs(out) >= 99999
    last_valid = 0.0
    have_valid = False
    for idx, value in enumerate(out):
        if invalid[idx]:
            out[idx] = last_valid if have_valid else 0.0
        else:
            last_valid = value
            have_valid = True
    return out


def _line_label(line_index: int, offset: float) -> str:
    labels = ["NS", "MNS", "MOS", "OS"]
    if line_index <= len(labels):
        return labels[line_index - 1]
    if pd.notna(offset):
        return f"L{line_index} ({offset:+.3f} m)"
    return f"L{line_index}"
