import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="CORSIA Dashboard", layout="wide")

# -----------------------
# Files
# -----------------------
BASELINE_XLSX = "2019_2020_CO2_StatePairs_table_Nov2021.xlsx"
CURRENT_XLSX  = "2024_CO2_StatePairs_table.xlsx"
ATTR_XLSX     = "CORSIA_AO_to_State_Attributions_10ed_web-2_extracted.xlsx"  # optional

# -----------------------
# Helpers
# -----------------------
def clean_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"-", "—", "", "nan", "NaN"}:
        return np.nan
    s = s.replace(",", "").replace("*", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def split_pair(val: str, delim: str):
    if pd.isna(val):
        return (None, None)
    s = str(val).strip()
    parts = s.split(delim)
    if len(parts) != 2:
        return (s, None)
    return parts[0].strip(), parts[1].strip()

def fmt_int(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "—"

def fmt_pct(x, digits=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{float(x):.{digits}f}%"

def require_file(path: str, optional: bool = False):
    import os
    if not os.path.exists(path):
        if optional:
            return False
        st.error(f"File not found: `{path}`. Pastikan file ada di folder app.")
        st.stop()
    return True

# -----------------------
# Loaders
# -----------------------
@st.cache_data
def load_baseline_2019(path: str):
    raw = pd.read_excel(path, sheet_name=0, header=None)

    idx = raw.index[raw[0].astype(str).str.contains("Afghanistan", na=False)]
    if len(idx) == 0:
        raise ValueError("Baseline file: tidak menemukan baris awal 'Afghanistan'.")
    start_idx = idx[0]

    data = raw.iloc[start_idx:].copy()
    data.columns = ["state_pair", "pilot_2019", "avg_2019_2020"]

    data["emissions_tco2"] = data["pilot_2019"].apply(clean_num)
    data[["origin", "dest"]] = data["state_pair"].apply(lambda x: pd.Series(split_pair(x, "-")))

    out = data[["origin", "dest", "emissions_tco2"]].dropna(subset=["origin", "dest"]).copy()
    out["year"] = 2019
    return out

@st.cache_data
def load_current_2024(path: str):
    raw = pd.read_excel(path, sheet_name=0, header=None)

    idx = raw.index[raw[0].astype(str).str.contains("Afghanistan", na=False)]
    if len(idx) == 0:
        raise ValueError("Current 2024 file: tidak menemukan baris awal 'Afghanistan'.")
    start_idx = idx[0]

    data = raw.iloc[start_idx:].copy()
    data.columns = ["state_pair", "subject_tco2", "not_subject_tco2"]

    data["subject_tco2"] = data["subject_tco2"].apply(clean_num)
    data["not_subject_tco2"] = data["not_subject_tco2"].apply(clean_num)
    data[["origin", "dest"]] = data["state_pair"].apply(lambda x: pd.Series(split_pair(x, "/")))

    rows = []
    for _, r in data.iterrows():
        if pd.notna(r["origin"]) and pd.notna(r["dest"]):
            if pd.notna(r["subject_tco2"]):
                rows.append((r["origin"], r["dest"], 2024, float(r["subject_tco2"]), True))
            if pd.notna(r["not_subject_tco2"]):
                rows.append((r["origin"], r["dest"], 2024, float(r["not_subject_tco2"]), False))

    df = pd.DataFrame(rows, columns=["origin", "dest", "year", "emissions_tco2", "is_subject"])
    return df

@st.cache_data
def load_attribution_optional(path: str):
    try:
        df = pd.read_excel(path, sheet_name="Attributions")
    except Exception:
        return pd.DataFrame()

    rename_map = {
        "State": "state",
        "Aeroplane Operator Name": "operator_name",
        "Attribution Method": "attribution_method",
        "Identifier": "identifier",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# -----------------------
# Session state
# -----------------------
if "A" not in st.session_state:
    st.session_state.A = None
if "B" not in st.session_state:
    st.session_state.B = None

def reset_ab():
    st.session_state.A = None
    st.session_state.B = None

# -----------------------
# App start
# -----------------------
require_file(BASELINE_XLSX)
require_file(CURRENT_XLSX)
attr_exists = require_file(ATTR_XLSX, optional=True)

try:
    baseline = load_baseline_2019(BASELINE_XLSX)
    current  = load_current_2024(CURRENT_XLSX)
    attrib   = load_attribution_optional(ATTR_XLSX) if attr_exists else pd.DataFrame()
except Exception as e:
    st.error(f"Gagal load data: {e}")
    st.stop()

# -----------------------
# Build density (IMPORTANT: must exist before map)
# Density = origin + destination totals per country
# -----------------------
origin_density = (
    current.groupby("origin", as_index=False)["emissions_tco2"]
    .sum()
    .rename(columns={"origin": "country", "emissions_tco2": "emissions"})
)
dest_density = (
    current.groupby("dest", as_index=False)["emissions_tco2"]
    .sum()
    .rename(columns={"dest": "country", "emissions_tco2": "emissions"})
)
country_density = (
    pd.concat([origin_density, dest_density], ignore_index=True)
    .groupby("country", as_index=False)["emissions"]
    .sum()
)
country_density["log_emissions"] = np.log10(country_density["emissions"] + 1)

# -----------------------
# Totals for share (%)
# -----------------------
GLOBAL_TOTAL = float(current["emissions_tco2"].sum())
GLOBAL_SUBJECT_TOTAL = float(current.loc[current["is_subject"] == True, "emissions_tco2"].sum())

# Countries list (for safety; can also be from density)
countries = sorted(set(current["origin"]).union(set(current["dest"])))

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Controls")
baseline_mode = st.sidebar.radio("Baseline", ["2019", "85% of 2019"], index=0)
baseline_mult = 0.85 if baseline_mode.startswith("85") else 1.0
if st.sidebar.button("Reset A/B"):
    reset_ab()

st.title("CORSIA State-Pair Dashboard")
st.caption("Density map: total emissions involvement by country (origin + destination). Klik dua negara: A lalu B.")

map_col, panel_col = st.columns([1.25, 1.0], gap="large")

# -----------------------
# MAP: Density Choropleth (click countries)
# -----------------------
with map_col:
    # =====================
    # 1. Base choropleth (DENSITY)
    # =====================
    fig_map = px.choropleth(
        country_density,
        locations="country",
        locationmode="country names",
        color="log_emissions",
        hover_name="country",
        hover_data={"emissions": ":,.0f", "log_emissions": False},
        labels={"log_emissions": "Emissions intensity (log10)"},
        color_continuous_scale="YlOrRd",
    )

    # =====================
    # 2. Invisible scatter layer (CLICK HANDLER)
    # =====================
    click_df = country_density.copy()

    fig_scatter = px.scatter_geo(
        click_df,
        locations="country",
        locationmode="country names",
        hover_name="country",
        custom_data=["country"],
    )

    # make scatter invisible but clickable
    fig_scatter.update_traces(
        marker=dict(size=18, opacity=0),
        showlegend=False
    )

    # =====================
    # 3. Merge layers
    # =====================
    for tr in fig_scatter.data:
        fig_map.add_trace(tr)

    fig_map.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        clickmode="event+select",  # IMPORTANT
        coloraxis_colorbar=dict(title="tCO₂ (log10)"),
    )

    fig_map.update_geos(
        showcountries=True,
        showcoastlines=True,
        projection_type="natural earth",
    )

    # =====================
    # 4. Native Streamlit select
    # =====================
    event = st.plotly_chart(
        fig_map,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    # =====================
    # 5. A / B logic
    # =====================
    if event and event.get("selection") and event["selection"].get("points"):
        p = event["selection"]["points"][0]
        loc = p.get("customdata", [None])[0]

        if loc:
            if st.session_state.A is None:
                st.session_state.A = loc
                st.rerun()
            elif st.session_state.B is None:
                if loc != st.session_state.A:
                    st.session_state.B = loc
                    st.rerun()
            else:
                st.session_state.A = loc
                st.session_state.B = None
                st.rerun()

    st.markdown(
        f"**Selected:** A = `{st.session_state.A or '—'}` | "
        f"B = `{st.session_state.B or '—'}`"
    )
# -----------------------
# PANEL: Pair details + shares + donut
# -----------------------
with panel_col:
    st.subheader("Detail Panel")

    A, B = st.session_state.A, st.session_state.B
    if not A or not B:
        st.info("Klik dua negara di map untuk melihat detail state-pair.")
        st.stop()

    pair_cur  = current[(current["origin"] == A) & (current["dest"] == B)]
    pair_base = baseline[(baseline["origin"] == A) & (baseline["dest"] == B)]

    current_total = float(pair_cur["emissions_tco2"].sum()) if not pair_cur.empty else np.nan
    baseline_2019 = float(pair_base["emissions_tco2"].sum()) if not pair_base.empty else np.nan
    baseline_used = baseline_2019 * baseline_mult if pd.notna(baseline_2019) else np.nan

    growth_abs = current_total - baseline_used if pd.notna(current_total) and pd.notna(baseline_used) else np.nan
    growth_pct = (growth_abs / baseline_used * 100) if pd.notna(baseline_used) and baseline_used > 0 else np.nan

    subject = float(pair_cur.loc[pair_cur["is_subject"] == True, "emissions_tco2"].sum()) if not pair_cur.empty else 0.0
    not_subject = float(pair_cur.loc[pair_cur["is_subject"] == False, "emissions_tco2"].sum()) if not pair_cur.empty else 0.0
    total = subject + not_subject
    subject_share_pair = (subject / total * 100) if total > 0 else np.nan

    share_global = (current_total / GLOBAL_TOTAL * 100) if pd.notna(current_total) and GLOBAL_TOTAL > 0 else np.nan
    share_global_subject = (subject / GLOBAL_SUBJECT_TOTAL * 100) if GLOBAL_SUBJECT_TOTAL > 0 else np.nan

    st.caption("Share of total international aviation emissions (2024)")

    st.markdown(f"### {A} → {B}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pair emissions (2024) tCO₂", fmt_int(current_total))
    k2.metric("Share of global total", fmt_pct(share_global))
    k3.metric("Share of CORSIA-subject", fmt_pct(share_global_subject))
    k4.metric("Subject share (within pair)", fmt_pct(subject_share_pair))

    # Donut contribution
    if pd.notna(current_total) and GLOBAL_TOTAL > 0:
        fig_donut = px.pie(
            names=["Selected pair", "Rest of world"],
            values=[current_total, max(GLOBAL_TOTAL - current_total, 0)],
            hole=0.62,
            title="Contribution to global emissions (2024)",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    # Baseline vs current
    fig1 = px.bar(
        pd.DataFrame(
            {
                "Scenario": [f"Baseline {baseline_mode}", "Current 2024"],
                "Emissions (tCO₂)": [baseline_used, current_total],
            }
        ),
        x="Scenario",
        y="Emissions (tCO₂)",
        text_auto=True,
        title="Current vs Baseline",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Subject vs not subject
    fig2 = px.bar(
        pd.DataFrame(
            {
                "Row": ["Total", "Total"],
                "Category": ["Subject", "Not subject"],
                "Emissions (tCO₂)": [subject, not_subject],
            }
        ),
        x="Emissions (tCO₂)",
        y="Row",
        color="Category",
        orientation="h",
        text_auto=True,
        title=f"Subject vs Not subject — 2024 (Subject share: {fmt_pct(subject_share_pair)})",
    )
    fig2.update_layout(yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        f"""
**Interpretation**  
The selected state-pair accounts for **{fmt_pct(share_global)}** of total global international aviation CO₂
emissions in 2024. Within this pair, **{fmt_pct(subject_share_pair)}** of emissions are subject to CORSIA.
"""
    )

    # Optional attribution view
    if not attrib.empty:
        tabs = st.tabs(["Attribution (context)", "Notes"])
        with tabs[0]:
            if "state" in attrib.columns:
                st.markdown(f"**Operators attributed to {A}**")
                st.dataframe(attrib[attrib["state"] == A].head(200), use_container_width=True, height=220)
                st.markdown(f"**Operators attributed to {B}**")
                st.dataframe(attrib[attrib["state"] == B].head(200), use_container_width=True, height=220)
            else:
                st.dataframe(attrib.head(200), use_container_width=True)

        with tabs[1]:
            st.markdown(
                """
- Map = density navigator (country involvement: origin + destination).
- Panel = state-pair explainer (baseline, growth, CORSIA subject split).
                """
            )
