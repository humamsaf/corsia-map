import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="CORSIA State-Pair Dashboard", layout="wide")

# =====================
# FILES
# =====================
BASELINE_XLSX = "2019_2020_CO2_StatePairs_table_Nov2021.xlsx"
CURRENT_XLSX  = "2024_CO2_StatePairs_table.xlsx"

# =====================
# HELPERS
# =====================
def clean_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", "").replace("*", "")
    if s in {"", "-", "—"}: return np.nan
    try: return float(s)
    except: return np.nan

def split_pair(val, delim):
    if pd.isna(val): return (None, None)
    p = str(val).split(delim)
    return (p[0].strip(), p[1].strip()) if len(p) == 2 else (p[0], None)

def fmt_int(x):
    return "—" if pd.isna(x) else f"{int(round(x)):,}"

def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.2f}%"

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_baseline():
    raw = pd.read_excel(BASELINE_XLSX, header=None)
    i = raw.index[raw[0].astype(str).str.contains("Afghanistan")][0]
    d = raw.iloc[i:].copy()
    d.columns = ["pair","v","_"]
    d["emissions"] = d["v"].apply(clean_num)
    d[["o","d"]] = d["pair"].apply(lambda x: pd.Series(split_pair(x, "-")))
    return d[["o","d","emissions"]].dropna()

@st.cache_data
def load_current():
    raw = pd.read_excel(CURRENT_XLSX, header=None)
    i = raw.index[raw[0].astype(str).str.contains("Afghanistan")][0]
    d = raw.iloc[i:].copy()
    d.columns = ["pair","sub","nsub"]
    d["sub"] = d["sub"].apply(clean_num)
    d["nsub"] = d["nsub"].apply(clean_num)
    d[["o","d"]] = d["pair"].apply(lambda x: pd.Series(split_pair(x, "/")))

    rows=[]
    for _,r in d.iterrows():
        if pd.notna(r["sub"]): rows.append((r["o"],r["d"],r["sub"],True))
        if pd.notna(r["nsub"]): rows.append((r["o"],r["d"],r["nsub"],False))
    return pd.DataFrame(rows,columns=["o","d","emissions","subject"])

baseline = load_baseline()
current  = load_current()

# =====================
# SESSION STATE
# =====================
if "A" not in st.session_state: st.session_state.A=None
if "B" not in st.session_state: st.session_state.B=None

# =====================
# TOTALS (GLOBAL CONTEXT)
# =====================
GLOBAL_TOTAL = current["emissions"].sum()
GLOBAL_SUBJECT_TOTAL = current.loc[current["subject"],"emissions"].sum()

# =====================
# UI
# =====================
st.title("CORSIA State-Pair Emissions Dashboard")
st.caption("Absolute emissions with global and CORSIA-subject context")

countries = sorted(set(current["o"]).union(set(current["d"])))

map_col, panel_col = st.columns([1.2,1])

# =====================
# MAP
# =====================
with map_col:
    fig_map = px.choropleth(
        country_density,
        locations="country",
        locationmode="country names",
        color="log_emissions",  # use log scale for readability
        hover_name="country",
        hover_data={
            "emissions": ":,.0f",
            "log_emissions": False
        },
        color_continuous_scale="YlOrRd",
        labels={"log_emissions":"Emissions intensity (log)"},
    )

    fig_map.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_colorbar=dict(
            title="tCO₂ (log scale)"
        ),
    )

    fig_map.update_geos(
        showcountries=True,
        showcoastlines=True,
        projection_type="natural earth",
    )

    # Native Streamlit select
    event = st.plotly_chart(
        fig_map,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    if event and event["selection"]["points"]:
        country_clicked = event["selection"]["points"][0]["location"]

        if st.session_state.A is None:
            st.session_state.A = country_clicked
            st.rerun()
        elif st.session_state.B is None:
            if country_clicked != st.session_state.A:
                st.session_state.B = country_clicked
                st.rerun()
        else:
            st.session_state.A = country_clicked
            st.session_state.B = None
            st.rerun()

    st.markdown(
        f"**Selected:** A = `{st.session_state.A or '—'}` | "
        f"B = `{st.session_state.B or '—'}`"
    )
# =====================
# PANEL
# =====================
with panel_col:
    if not st.session_state.A or not st.session_state.B:
        st.info("Klik dua negara pada peta")
        st.stop()

    A,B = st.session_state.A, st.session_state.B
    sel = current[(current["o"]==A)&(current["d"]==B)]

    total = sel["emissions"].sum()
    subject = sel.loc[sel["subject"],"emissions"].sum()
    nsubject = total-subject

    share_global = total/GLOBAL_TOTAL*100
    share_subject = subject/GLOBAL_SUBJECT_TOTAL*100 if subject>0 else np.nan

    # ---- CONTEXT LINE
    st.caption("Share of total international aviation emissions (2024)")

    # ---- KPI ROW
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Pair emissions", fmt_int(total)+" tCO₂")
    k2.metric("Share of global total", fmt_pct(share_global))
    k3.metric("Share of CORSIA-subject", fmt_pct(share_subject))
    k4.metric("Subject share", fmt_pct(subject/total*100 if total>0 else np.nan))

    # ---- DONUT
    fig_donut = px.pie(
        names=["Selected pair","Rest of world"],
        values=[total, GLOBAL_TOTAL-total],
        hole=0.6,
        title="Contribution to global emissions (2024)"
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    # ---- EXISTING CHARTS
    fig_bar = px.bar(
        pd.DataFrame({
            "Category":["Subject","Not subject"],
            "Emissions":[subject,nsubject]
        }),
        x="Category",y="Emissions",
        title="Subject vs not subject emissions"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        f"""
**Interpretation:**  
The state-pair **{A} → {B}** accounts for **{fmt_pct(share_global)}** of total global international aviation CO₂
emissions in 2024, of which **{fmt_pct(subject/total*100 if total>0 else np.nan)}** are subject to CORSIA.
"""
    )
