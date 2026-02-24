import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="CORSIA Dashboard", layout="wide")

# -----------------------
# Files (must exist in same folder as app)
# -----------------------
BASELINE_XLSX = "2019_2020_CO2_StatePairs_table_Nov2021.xlsx"
CURRENT_XLSX  = "2024_CO2_StatePairs_table.xlsx"
ATTR_XLSX     = "CORSIA_AO_to_State_Attributions_10ed_web-2_extracted.xlsx"

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

def fmt_pct(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{float(x):+.1f}%"

def require_file(path: str):
    import os
    if not os.path.exists(path):
        st.error(f"File not found: `{path}`. Pastikan file ada di folder app.")
        st.stop()

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

    return pd.DataFrame(rows, columns=["origin", "dest", "year", "emissions_tco2", "is_subject"])

@st.cache_data
def load_attribution(path: str):
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
# App
# -----------------------
require_file(BASELINE_XLSX)
require_file(CURRENT_XLSX)
require_file(ATTR_XLSX)

try:
    baseline = load_baseline_2019(BASELINE_XLSX)
    current  = load_current_2024(CURRENT_XLSX)
    attrib   = load_attribution(ATTR_XLSX)
except Exception as e:
    st.error(f"Gagal load data: {e}")
    st.stop()

st.sidebar.header("Controls")
baseline_mode = st.sidebar.radio("Baseline", ["2019", "85% of 2019"], index=0)
baseline_mult = 0.85 if baseline_mode.startswith("85") else 1.0
if st.sidebar.button("Reset A/B"):
    reset_ab()

countries = sorted(set(current["origin"]).union(set(current["dest"])))

st.title("CORSIA State-Pair Dashboard")
st.caption("Klik negara pertama = pilih Origin (A). Klik negara kedua = pilih Destination (B).")

# OPTIONAL: show Streamlit version (helpful for debugging)
# st.write("Streamlit version:", st.__version__)

map_col, panel_col = st.columns([1.25, 1.0], gap="large")

# -----------------------
# MAP (native select)
# -----------------------
with map_col:
    pin_df = pd.DataFrame({"country": countries})

    def role(c):
        if st.session_state.A == c:
            return "A"
        if st.session_state.B == c:
            return "B"
        return "Other"

    pin_df["role"] = pin_df["country"].apply(role)

    fig_map = px.scatter_geo(
        pin_df,
        locations="country",
        locationmode="country names",
        hover_name="country",
        color="role",
        custom_data=["country"],
    )
    fig_map.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="role")
    fig_map.update_geos(
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="rgba(200,200,200,0.25)",
        projection_type="natural earth",
    )

    event = st.plotly_chart(
        fig_map,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    # Selection logic (A then B; third click resets to new A)
    if event and event.get("selection") and event["selection"].get("points"):
        p = event["selection"]["points"][0]
        loc = None

        # customdata should exist because we set custom_data=["country"]
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)) and len(cd) > 0:
            loc = cd[0]

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

    st.markdown(f"**Selected:** A = `{st.session_state.A or '—'}` | B = `{st.session_state.B or '—'}`")

# -----------------------
# DETAIL PANEL
# -----------------------
with panel_col:
    st.subheader("Detail Panel")
    A, B = st.session_state.A, st.session_state.B

    if not A or not B:
        st.info("Klik dua negara di map untuk melihat detail pair.")
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
    subject_share = (subject / total * 100) if total > 0 else np.nan

    st.markdown(f"### {A} → {B}")

    k1, k2, k3 = st.columns(3)
    k1.metric("Current total (2024) tCO₂", fmt_int(current_total))
    k2.metric(f"Baseline ({baseline_mode}) tCO₂", fmt_int(baseline_used))
    k3.metric("Growth vs baseline", fmt_int(growth_abs), fmt_pct(growth_pct))

    st.divider()

    fig1 = px.bar(
        pd.DataFrame(
            {"Scenario": [f"Baseline {baseline_mode}", "Current 2024"],
             "Emissions (tCO₂)": [baseline_used, current_total]}
        ),
        x="Scenario",
        y="Emissions (tCO₂)",
        text_auto=True,
        title="Current vs Baseline",
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        pd.DataFrame(
            {"Row": ["Total", "Total"],
             "Category": ["Subject", "Not subject"],
             "Emissions (tCO₂)": [subject, not_subject]}
        ),
        x="Emissions (tCO₂)",
        y="Row",
        color="Category",
        orientation="h",
        text_auto=True,
        title=f"Subject vs Not subject — 2024 (Subject share: {fmt_pct(subject_share).replace('+','')})",
    )
    fig2.update_layout(yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

    tabs = st.tabs(["Attribution (context)", "Notes"])

    with tabs[0]:
        if attrib.empty:
            st.info("Attribution file loaded but sheet/columns may differ. (Optional layer)")
        else:
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
- Map = selector (navigator). Angka & grafik ada di panel kanan.
- Baseline dari “Pilot phase – based on 2019 data”.
- Current 2024 dipisah subject vs not subject.
            """
        )
