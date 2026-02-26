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
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "").replace("*", "")
    if s in {"", "-", "—"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def split_pair(val, delim):
    if pd.isna(val):
        return (None, None)
    p = str(val).split(delim)
    return (p[0].strip(), p[1].strip()) if len(p) == 2 else (p[0].strip(), None)

def fmt_int(x):
    return "—" if pd.isna(x) else f"{int(round(x)):,}"

def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.2f}%"

def safe_div(a, b):
    if b is None or b == 0 or pd.isna(b):
        return np.nan
    return a / b

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_baseline():
    raw = pd.read_excel(BASELINE_XLSX, header=None)
    i = raw.index[raw[0].astype(str).str.contains("Afghanistan", na=False)][0]
    d = raw.iloc[i:].copy()
    d.columns = ["pair", "v", "_"]
    d["emissions"] = d["v"].apply(clean_num)
    d[["o", "d"]] = d["pair"].apply(lambda x: pd.Series(split_pair(x, "-")))
    d = d[["o", "d", "emissions"]].dropna()
    return d

@st.cache_data
def load_current():
    raw = pd.read_excel(CURRENT_XLSX, header=None)
    i = raw.index[raw[0].astype(str).str.contains("Afghanistan", na=False)][0]
    d = raw.iloc[i:].copy()
    d.columns = ["pair", "sub", "nsub"]
    d["sub"] = d["sub"].apply(clean_num)
    d["nsub"] = d["nsub"].apply(clean_num)
    d[["o", "d"]] = d["pair"].apply(lambda x: pd.Series(split_pair(x, "/")))

    rows = []
    for _, r in d.iterrows():
        if pd.notna(r["sub"]):
            rows.append((r["o"], r["d"], r["sub"], True))
        if pd.notna(r["nsub"]):
            rows.append((r["o"], r["d"], r["nsub"], False))

    cur = pd.DataFrame(rows, columns=["o", "d", "emissions", "subject"]).dropna(subset=["o", "d", "emissions"])
    return cur

baseline = load_baseline()
current  = load_current()

# =====================
# SESSION STATE
# =====================
if "A" not in st.session_state: st.session_state.A = None
if "B" not in st.session_state: st.session_state.B = None

# =====================
# RANKING TABLES
# =====================
@st.cache_data
def build_rankings(cur: pd.DataFrame):
    df = cur.copy()

    # pairs (directed)
    pair_all = (
        df.groupby(["o", "d"], as_index=False)["emissions"]
          .sum()
          .sort_values("emissions", ascending=False)
    )
    pair_sub = (
        df[df["subject"]]
        .groupby(["o", "d"], as_index=False)["emissions"]
        .sum()
        .sort_values("emissions", ascending=False)
    )

    # countries: origin / destination
    o_all = df.groupby("o", as_index=False)["emissions"].sum().rename(columns={"o": "country"})
    d_all = df.groupby("d", as_index=False)["emissions"].sum().rename(columns={"d": "country"})
    o_sub = df[df["subject"]].groupby("o", as_index=False)["emissions"].sum().rename(columns={"o": "country"})
    d_sub = df[df["subject"]].groupby("d", as_index=False)["emissions"].sum().rename(columns={"d": "country"})

    # involvement = origin + destination
    c_all = (
        pd.concat([o_all, d_all], ignore_index=True)
          .groupby("country", as_index=False)["emissions"].sum()
          .sort_values("emissions", ascending=False)
    )
    c_sub = (
        pd.concat([o_sub, d_sub], ignore_index=True)
          .groupby("country", as_index=False)["emissions"].sum()
          .sort_values("emissions", ascending=False)
    )

    # sort origin/dest
    o_all = o_all.sort_values("emissions", ascending=False)
    d_all = d_all.sort_values("emissions", ascending=False)
    o_sub = o_sub.sort_values("emissions", ascending=False)
    d_sub = d_sub.sort_values("emissions", ascending=False)

    return pair_all, pair_sub, c_all, c_sub, o_all, d_all, o_sub, d_sub

pair_all, pair_sub, c_all, c_sub, o_all, d_all, o_sub, d_sub = build_rankings(current)

# =====================
# UI
# =====================
st.title("CORSIA State-Pair Emissions Dashboard")
st.caption("Klik peta atau gunakan dropdown. Ada mode All vs CORSIA-subject, plus ranking pairs & countries.")

countries = sorted(set(current["o"]).union(set(current["d"])))

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Controls")

    mode = st.radio("Dataset mode", ["All emissions", "CORSIA-subject only"], horizontal=False, index=0)
    topn = st.slider("Top N (rankings)", 5, 30, 15)

    st.divider()
    st.subheader("Select route (A → B)")

    A_sel = st.selectbox(
        "Origin (A)",
        options=["—"] + countries,
        index=(countries.index(st.session_state.A) + 1) if st.session_state.A in countries else 0
    )
    B_sel = st.selectbox(
        "Destination (B)",
        options=["—"] + countries,
        index=(countries.index(st.session_state.B) + 1) if st.session_state.B in countries else 0
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Apply", use_container_width=True):
            st.session_state.A = None if A_sel == "—" else A_sel
            st.session_state.B = None if B_sel == "—" else B_sel
            st.rerun()
    with b2:
        if st.button("Swap", use_container_width=True):
            st.session_state.A, st.session_state.B = st.session_state.B, st.session_state.A
            st.rerun()
    with b3:
        if st.button("Reset", use_container_width=True):
            st.session_state.A = None
            st.session_state.B = None
            st.rerun()

    st.divider()
    st.subheader("Country view")
    country_focus = st.selectbox("Focus country", options=["—"] + countries, index=0)

# ---------- FILTERED DATA BY MODE ----------
df_mode = current.copy()
if mode == "CORSIA-subject only":
    df_mode = df_mode[df_mode["subject"]].copy()

GLOBAL_TOTAL_MODE = df_mode["emissions"].sum()

# subject totals always from full dataset
GLOBAL_TOTAL_ALL = current["emissions"].sum()
GLOBAL_SUBJECT_TOTAL_ALL = current.loc[current["subject"], "emissions"].sum()

# =====================
# LAYOUT
# =====================
map_col, panel_col = st.columns([1.2, 1], gap="large")

# =====================
# MAP
# =====================
with map_col:
    dfm = pd.DataFrame({"country": countries})
    dfm["role"] = dfm["country"].apply(
        lambda c: "A" if c == st.session_state.A else "B" if c == st.session_state.B else "Other"
    )

    fig_map = px.scatter_geo(
        dfm,
        locations="country",
        locationmode="country names",
        color="role",
        hover_name="country",
        custom_data=["country"],
    )
    fig_map.update_traces(marker=dict(size=7))
    fig_map.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10))

    ev_map = st.plotly_chart(fig_map, on_select="rerun", selection_mode="points")

    if ev_map and ev_map.get("selection") and ev_map["selection"].get("points"):
        c = ev_map["selection"]["points"][0]["customdata"][0]
        if st.session_state.A is None:
            st.session_state.A = c
        elif st.session_state.B is None and c != st.session_state.A:
            st.session_state.B = c
        else:
            st.session_state.A = c
            st.session_state.B = None
        st.rerun()

    st.markdown(f"**Selected:** {st.session_state.A or '—'} → {st.session_state.B or '—'}")
    st.caption(
        f"Mode: **{mode}** • Global total (mode): **{fmt_int(GLOBAL_TOTAL_MODE)} tCO₂** "
        f"• Global total (all): **{fmt_int(GLOBAL_TOTAL_ALL)} tCO₂**"
    )

# =====================
# RIGHT PANEL (TABS)
# =====================
with panel_col:
    tabs = st.tabs(["Selected Pair", "Rankings", "Country view"])

    # ---------- TAB 1: Selected Pair ----------
    with tabs[0]:
        if not st.session_state.A or not st.session_state.B:
            st.info("Klik dua negara pada peta (A lalu B), atau pilih via dropdown di sidebar.")
        else:
            A, B = st.session_state.A, st.session_state.B

            # For pair panel we want full split subject vs not subject from full dataset (not mode-filtered),
            # but also show shares under current mode.
            sel_full = current[(current["o"] == A) & (current["d"] == B)].copy()
            total = sel_full["emissions"].sum()
            subject = sel_full.loc[sel_full["subject"], "emissions"].sum()
            nsubject = total - subject

            # shares (mode)
            if mode == "All emissions":
                share_global = safe_div(total, GLOBAL_TOTAL_ALL) * 100
            else:
                # share within subject-only universe
                share_global = safe_div(subject, GLOBAL_SUBJECT_TOTAL_ALL) * 100

            share_subject_all = safe_div(subject, GLOBAL_SUBJECT_TOTAL_ALL) * 100
            subject_share_within_pair = safe_div(subject, total) * 100

            st.caption("KPIs (2024). Shares depend on selected mode.")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Pair emissions (total)", f"{fmt_int(total)} tCO₂")
            k2.metric("Pair subject emissions", f"{fmt_int(subject)} tCO₂")
            k3.metric(
                "Share of universe",
                fmt_pct(share_global),
                help="Universe = All emissions (global total) or CORSIA-subject only (subject total), depending on mode."
            )
            k4.metric("Subject share (within pair)", fmt_pct(subject_share_within_pair))

            # Contribution donut (mode universe)
            if mode == "All emissions":
                universe_total = GLOBAL_TOTAL_ALL
                selected_value = total
                title = "Contribution to global emissions (All)"
            else:
                universe_total = GLOBAL_SUBJECT_TOTAL_ALL
                selected_value = subject
                title = "Contribution to global emissions (Subject-only)"

            fig_donut = px.pie(
                names=["Selected", "Rest of world"],
                values=[selected_value, max(universe_total - selected_value, 0)],
                hole=0.6,
                title=title,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            # Subject vs not subject bar (always meaningful)
            fig_bar = px.bar(
                pd.DataFrame({"Category": ["Subject", "Not subject"], "Emissions": [subject, nsubject]}),
                x="Category",
                y="Emissions",
                title="Subject vs not subject (Selected pair)",
                labels={"Emissions": "tCO₂"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown(
                f"""
**Interpretation:**  
State-pair **{A} → {B}** total emissions = **{fmt_int(total)} tCO₂** (subject: **{fmt_int(subject)} tCO₂**).  
Subject share within the pair is **{fmt_pct(subject_share_within_pair)}**.  
Share of **subject-only** global total is **{fmt_pct(share_subject_all)}**.
"""
            )

    # ---------- TAB 2: Rankings ----------
    with tabs[1]:
        st.subheader("Rankings (2024)")

        if mode == "All emissions":
            pairs = pair_all.head(topn).copy()
            countries_rank = c_all.head(topn).copy()
            origins_rank = o_all.head(topn).copy()
            dests_rank = d_all.head(topn).copy()
            universe_label = "All emissions"
        else:
            pairs = pair_sub.head(topn).copy()
            countries_rank = c_sub.head(topn).copy()
            origins_rank = o_sub.head(topn).copy()
            dests_rank = d_sub.head(topn).copy()
            universe_label = "CORSIA-subject only"

        pairs["pair"] = pairs["o"] + " → " + pairs["d"]

        # Top pairs (clickable -> set A/B)
        plot_pairs = pairs.sort_values("emissions").reset_index(drop=True)
        fig_pairs = px.bar(
            plot_pairs,
            x="emissions",
            y="pair",
            orientation="h",
            title=f"Top {topn} state-pairs by emissions ({universe_label})",
            labels={"emissions": "tCO₂", "pair": "State-pair"},
            custom_data=["o", "d"],
        )
        fig_pairs.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        ev_pairs = st.plotly_chart(fig_pairs, on_select="rerun", selection_mode="points")

        if ev_pairs and ev_pairs.get("selection") and ev_pairs["selection"].get("points"):
            idx = ev_pairs["selection"]["points"][0]["pointIndex"]
            row = plot_pairs.iloc[idx]
            st.session_state.A = row["o"]
            st.session_state.B = row["d"]
            st.rerun()

        # Top countries by involvement
        plot_c = countries_rank.sort_values("emissions").reset_index(drop=True)
        fig_c = px.bar(
            plot_c,
            x="emissions",
            y="country",
            orientation="h",
            title=f"Top {topn} countries by involvement (origin + destination) ({universe_label})",
            labels={"emissions": "tCO₂", "country": "Country"},
        )
        fig_c.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_c, use_container_width=True)

        with st.expander("Origin-only and destination-only rankings"):
            c1, c2 = st.columns(2)
            with c1:
                plot_o = origins_rank.sort_values("emissions").reset_index(drop=True)
                fig_o = px.bar(
                    plot_o,
                    x="emissions",
                    y="country",
                    orientation="h",
                    title=f"Top {topn} origins ({universe_label})",
                    labels={"emissions": "tCO₂"},
                )
                fig_o.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_o, use_container_width=True)

            with c2:
                plot_d = dests_rank.sort_values("emissions").reset_index(drop=True)
                fig_d = px.bar(
                    plot_d,
                    x="emissions",
                    y="country",
                    orientation="h",
                    title=f"Top {topn} destinations ({universe_label})",
                    labels={"emissions": "tCO₂"},
                )
                fig_d.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_d, use_container_width=True)

    # ---------- TAB 3: Country view ----------
    with tabs[2]:
        if country_focus == "—":
            st.info("Pilih 1 negara di sidebar (Country view) untuk melihat top outgoing/incoming routes.")
        else:
            C = country_focus

            # Universe: depends on mode
            dfC = df_mode.copy()

            outgoing = (
                dfC[dfC["o"] == C]
                .groupby(["o", "d"], as_index=False)["emissions"].sum()
                .sort_values("emissions", ascending=False)
                .head(topn)
            )
            incoming = (
                dfC[dfC["d"] == C]
                .groupby(["o", "d"], as_index=False)["emissions"].sum()
                .sort_values("emissions", ascending=False)
                .head(topn)
            )

            out_total = dfC[dfC["o"] == C]["emissions"].sum()
            in_total = dfC[dfC["d"] == C]["emissions"].sum()
            involvement = out_total + in_total

            share_universe = safe_div(involvement, GLOBAL_TOTAL_MODE) * 100

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Outgoing total", f"{fmt_int(out_total)} tCO₂")
            k2.metric("Incoming total", f"{fmt_int(in_total)} tCO₂")
            k3.metric("Involvement (out+in)", f"{fmt_int(involvement)} tCO₂")
            k4.metric("Share of universe", fmt_pct(share_universe))

            c1, c2 = st.columns(2)
            with c1:
                if outgoing.empty:
                    st.warning("No outgoing routes found in this mode.")
                else:
                    outgoing["pair"] = outgoing["o"] + " → " + outgoing["d"]
                    plot_out = outgoing.sort_values("emissions").reset_index(drop=True)
                    fig_out = px.bar(
                        plot_out,
                        x="emissions",
                        y="pair",
                        orientation="h",
                        title=f"Top outgoing routes from {C} ({mode})",
                        labels={"emissions": "tCO₂", "pair": "Route"},
                        custom_data=["o", "d"],
                    )
                    fig_out.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
                    ev_out = st.plotly_chart(fig_out, on_select="rerun", selection_mode="points")
                    if ev_out and ev_out.get("selection") and ev_out["selection"].get("points"):
                        idx = ev_out["selection"]["points"][0]["pointIndex"]
                        row = plot_out.iloc[idx]
                        st.session_state.A = row["o"]
                        st.session_state.B = row["d"]
                        st.rerun()

            with c2:
                if incoming.empty:
                    st.warning("No incoming routes found in this mode.")
                else:
                    incoming["pair"] = incoming["o"] + " → " + incoming["d"]
                    plot_in = incoming.sort_values("emissions").reset_index(drop=True)
                    fig_in = px.bar(
                        plot_in,
                        x="emissions",
                        y="pair",
                        orientation="h",
                        title=f"Top incoming routes to {C} ({mode})",
                        labels={"emissions": "tCO₂", "pair": "Route"},
                        custom_data=["o", "d"],
                    )
                    fig_in.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
                    ev_in = st.plotly_chart(fig_in, on_select="rerun", selection_mode="points")
                    if ev_in and ev_in.get("selection") and ev_in["selection"].get("points"):
                        idx = ev_in["selection"]["points"][0]["pointIndex"]
                        row = plot_in.iloc[idx]
                        st.session_state.A = row["o"]
                        st.session_state.B = row["d"]
                        st.rerun()

# =====================
# FOOTER (optional)
# =====================
with st.expander("Notes / Definitions"):
    st.write(
        """
- **All emissions**: subject + not subject (total international aviation CO₂ in dataset).
- **CORSIA-subject only**: hanya baris yang `subject=True`.
- **Country involvement**: outgoing + incoming (asal + tujuan). Ini lebih cocok untuk “negara paling dominan” di network.
- Klik bar chart pada Rankings/Country view untuk otomatis memilih route (A→B).
"""
    )
