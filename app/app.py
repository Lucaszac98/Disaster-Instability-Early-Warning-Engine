import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import config, data_prep
from src.forces import compute_instability
from src.train_model import load_model


@st.cache_data
def load_df() -> pd.DataFrame:
    return data_prep.load_processed()


def _event_label(i: int, row: pd.Series) -> str:
    eid = row.get(config.COL_EVENT_ID, i)
    typ = row.get(config.COL_TYPE, "")
    loc = row.get(config.COL_LOCATION, "")
    dt = row.get(config.COL_DATE, "")
    zone = row.get(config.COL_ZONE, "")
    return f"[{i}] #{eid} | {typ} | {loc} | {dt} | {zone}"


def _forces_frame(row: pd.Series) -> pd.DataFrame:
    items = [
        ("Hazard Pressure (severity)", config.COL_FORCE_HAZARD),
        ("Exposure Pressure (affected pop)", config.COL_FORCE_EXPOSURE),
        ("Response Latency (slow response)", config.COL_FORCE_LATENCY),
        ("Infrastructure Fragility", config.COL_FORCE_INFRA),
        ("Buffer Capacity (aid + speed)", config.COL_FORCE_BUFFER),
    ]
    out = []
    for name, col in items:
        v = row.get(col, np.nan)
        out.append({"Component": name, "Value": float(v) if pd.notna(v) else np.nan})
    return pd.DataFrame(out)


def _load_model_optional():
    try:
        return load_model()
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Disaster Instability Early Warning Engine", layout="wide")
    st.title("Disaster Instability Early Warning Engine")
    st.caption("Hazards become disasters when buffering capacity collapses under accumulated pressure.")

    df = load_df()
    if df.empty:
        st.error("Processed dataset is empty.")
        return

    model = _load_model_optional()

    tab1, tab2, tab3 = st.tabs(["Event Diagnostic", "Scenario Simulator", "Map / Cohort View"])

    # ---------------- Event Diagnostic ----------------
    with tab1:
        st.subheader("Event Diagnostic")
        st.markdown(
            "This view explains **why** a single event escalates by decomposing it into pressures and buffers.\n\n"
            "**Instability Index** is a leading signal: it rises before outcomes like loss/casualties fully materialize."
        )

        left, right = st.columns([2, 1])
        with left:
            options = [_event_label(i, df.iloc[i]) for i in range(min(len(df), 5000))]
            choice = st.selectbox("Choose an event (first 5k shown for performance)", options, index=0)
            idx = int(choice.split("]")[0].replace("[", "").strip())
        with right:
            if model is None:
                st.warning("ML risk model not found. Train it with: `python -m src.cli train`")
            else:
                st.success("ML risk model loaded ✓")

        row = df.iloc[idx]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Instability Index", f"{float(row[config.COL_INSTABILITY]):.3f}")
        m2.metric("Early-Warning Zone", str(row[config.COL_ZONE]))
        m3.metric("Buffer Capacity", f"{float(row[config.COL_FORCE_BUFFER]):.3f}")
        m4.metric("Observed Loss (USD)", f"{float(row[config.COL_LOSS]):,.0f}" if config.COL_LOSS in df.columns else "N/A")

        if model is not None:
            # Model expects X without target column
            X_one = row.to_frame().T
            proba = float(model.predict_proba(X_one)[:, 1][0])
            st.markdown("### ML Escalation Risk")
            st.metric("P(major disaster)", f"{proba:.3f}")

        st.markdown("### Force Decomposition")
        fdf = _forces_frame(row)

        # Separate pressures (negative) and buffer (positive) for readability
        chart = alt.Chart(fdf).mark_bar().encode(
            x=alt.X("Value:Q", title="Force value (pressures negative, buffer positive)"),
            y=alt.Y("Component:N", sort=None),
            tooltip=["Component", "Value"],
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("### Narrative Interpretation")
        st.write(
            "Think of this event as a system under competing forces:\n\n"
            "- **Hazard Pressure** increases with severity.\n"
            "- **Exposure Pressure** increases when more people are in the impact radius.\n"
            "- **Response Latency** increases instability when response is slow (coordination delay).\n"
            "- **Infrastructure Fragility** represents how easily damage propagates.\n"
            "- **Buffer Capacity** represents stabilizers such as aid + fast response.\n\n"
            "Instability rises when **pressures dominate buffers**, when pressures are **imbalanced**, and when hazard & exposure compound."
        )

        with st.expander("Raw row (debug / inspection)"):
            st.dataframe(pd.DataFrame(row).T)

    # ---------------- Scenario Simulator ----------------
    with tab2:
        st.subheader("Scenario Simulator (Counterfactual Stress Test)")
        st.markdown(
            "This simulator tests **what-if interventions** on the same event: faster response, aid delivery, reduced exposure, etc.\n\n"
            "The goal is not to produce certainty, but to identify **high-leverage actions** that reduce instability."
        )

        idx = st.number_input("Row index", min_value=0, max_value=int(len(df) - 1), value=0, step=1)
        base_row = df.iloc[int(idx)].copy()

        c1, c2, c3 = st.columns(3)
        with c1:
            severity_delta = st.slider("Δ severity_level", -5, 5, 0, 1)
        with c2:
            response_delta = st.slider("Δ response_time_hours", -48.0, 48.0, 0.0, 0.5)
        with c3:
            aid_toggle = st.selectbox("Aid provided", ["Keep", "Yes", "No"], index=0)

        c4, c5 = st.columns(2)
        with c4:
            affected_delta = st.slider("Δ affected_population", -50000, 50000, 0, 1000)
        with c5:
            infra_delta = st.slider("Δ infrastructure_damage_index", -0.5, 0.5, 0.0, 0.01)

        if st.button("Run Scenario"):
            sim = df.copy()
            r = base_row.copy()

            # Apply edits
            if config.COL_SEVERITY in r.index and pd.notna(r[config.COL_SEVERITY]):
                r[config.COL_SEVERITY] = float(r[config.COL_SEVERITY]) + float(severity_delta)
            if config.COL_RESPONSE_H in r.index and pd.notna(r[config.COL_RESPONSE_H]):
                r[config.COL_RESPONSE_H] = float(r[config.COL_RESPONSE_H]) + float(response_delta)
            if config.COL_AFFECTED in r.index and pd.notna(r[config.COL_AFFECTED]):
                r[config.COL_AFFECTED] = max(0.0, float(r[config.COL_AFFECTED]) + float(affected_delta))
            if config.COL_INFRA in r.index and pd.notna(r[config.COL_INFRA]):
                r[config.COL_INFRA] = float(r[config.COL_INFRA]) + float(infra_delta)

            if aid_toggle != "Keep" and config.COL_AID in r.index:
                r[config.COL_AID] = aid_toggle

            sim.iloc[int(idx)] = r

            # Recompute forces + instability cohort-relatively
            sim = compute_instability(sim)

            before = df.iloc[int(idx)]
            after = sim.iloc[int(idx)]

            st.markdown("### Results")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Instability (Before)", f"{float(before[config.COL_INSTABILITY]):.3f}")
            a2.metric("Instability (After)", f"{float(after[config.COL_INSTABILITY]):.3f}")
            d = float(after[config.COL_INSTABILITY]) - float(before[config.COL_INSTABILITY])
            a3.metric("Δ Instability", f"{d:+.3f}")
            a4.metric("Zone (After)", str(after[config.COL_ZONE]))

            if model is not None:
                p0 = float(model.predict_proba(before.to_frame().T)[:, 1][0])
                p1 = float(model.predict_proba(after.to_frame().T)[:, 1][0])
                st.markdown("### ML Risk (Major Disaster Escalation)")
                b1, b2, b3 = st.columns(3)
                b1.metric("P(major) Before", f"{p0:.3f}")
                b2.metric("P(major) After", f"{p1:.3f}")
                b3.metric("Δ P(major)", f"{(p1-p0):+.3f}")

            st.markdown("### Force Comparison")
            comp = pd.DataFrame({
                "Component": ["Hazard", "Exposure", "Latency", "Infra", "Buffer"],
                "Before": [
                    float(before[config.COL_FORCE_HAZARD]),
                    float(before[config.COL_FORCE_EXPOSURE]),
                    float(before[config.COL_FORCE_LATENCY]),
                    float(before[config.COL_FORCE_INFRA]),
                    float(before[config.COL_FORCE_BUFFER]),
                ],
                "After": [
                    float(after[config.COL_FORCE_HAZARD]),
                    float(after[config.COL_FORCE_EXPOSURE]),
                    float(after[config.COL_FORCE_LATENCY]),
                    float(after[config.COL_FORCE_INFRA]),
                    float(after[config.COL_FORCE_BUFFER]),
                ],
            })
            st.dataframe(comp)

            st.markdown("### Interpretation")
            st.write(
                "Look for interventions that reduce **latency pressure** and increase **buffer capacity**.\n"
                "In many real systems, small response improvements can produce nonlinear stability gains when the system is near a threshold."
            )

    # ---------------- Map / Cohort View ----------------
    with tab3:
        st.subheader("Map / Cohort View")
        st.markdown(
            "This view treats all events as a **pressure field** over geography.\n\n"
            "- Color reflects **instability** (or zone)\n"
            "- Tooltip reveals event context\n\n"
            "Use filters to reveal hotspots by disaster type, region, or warning zone."
        )

        f1, f2, f3 = st.columns(3)
        with f1:
            types = sorted(df[config.COL_TYPE].dropna().unique().tolist())
            type_filter = st.multiselect("Disaster types", types, default=types[:min(6, len(types))] if types else [])
        with f2:
            zones = df[config.COL_ZONE].dropna().unique().tolist()
            zone_filter = st.multiselect("Zones", zones, default=zones)
        with f3:
            max_points = st.slider("Max points to plot", 500, 10000, 3000, 500)

        plot_df = df.copy()
        if type_filter:
            plot_df = plot_df[plot_df[config.COL_TYPE].isin(type_filter)]
        if zone_filter:
            plot_df = plot_df[plot_df[config.COL_ZONE].isin(zone_filter)]

        plot_df = plot_df.dropna(subset=[config.COL_LAT, config.COL_LON]).head(int(max_points))

        chart = alt.Chart(plot_df).mark_circle(opacity=0.7).encode(
            x=alt.X(f"{config.COL_LON}:Q", title="Longitude"),
            y=alt.Y(f"{config.COL_LAT}:Q", title="Latitude"),
            color=alt.Color(f"{config.COL_ZONE}:N", title="Zone"),
            size=alt.Size(f"{config.COL_INSTABILITY}:Q", title="Instability", scale=alt.Scale(range=[20, 400])),
            tooltip=[
                config.COL_EVENT_ID,
                config.COL_TYPE,
                config.COL_LOCATION,
                config.COL_DATE,
                config.COL_SEVERITY,
                config.COL_AFFECTED,
                config.COL_RESPONSE_H,
                config.COL_LOSS,
                config.COL_INSTABILITY,
                config.COL_ZONE,
            ],
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        with st.expander("Show processed data (filtered)"):
            st.dataframe(plot_df)


if __name__ == "__main__":
    main()
