"""Lease rent predictor (Streamlit).

Run:  streamlit run app.py
Needs (same folder): lease_features.py and outputs_rent/ (best_model.joblib, model_meta.json,
final_test_metrics.csv) produced by the training notebook.
"""
import io
import json
from datetime import date
from pathlib import Path

import __main__
import joblib
import pandas as pd
import streamlit as st

from lease_features import NotesSVD, engineer_features, parse_property

# The model was pickled in a notebook where NotesSVD lived in __main__; make it findable here.
__main__.NotesSVD = NotesSVD

MODEL_DIR = Path(__file__).parent / "outputs_rent"
RAW_COLS = ["property_name", "start_date", "end_date", "service_charge", "notes"]
UNIT_OPTIONS = ["Auto-detect from address", "Flat", "Mini flat", "Self contained", "Shop",
                "Duplex", "Bungalow", "Terrace", "Maisonette", "Single room", "Church"]

st.set_page_config(page_title="Lease Rent Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    meta = json.load(open(MODEL_DIR / "model_meta.json"))
    typical_err = None
    metrics_path = MODEL_DIR / "final_test_metrics.csv"
    if metrics_path.exists():
        m = pd.read_csv(metrics_path)
        row = m[(m["model"] == meta["winner"]) & (m["variant"].str.startswith("C"))]
        if len(row):
            typical_err = float(row["MedAPE_%"].iloc[0])
    return model, meta, typical_err


def to_model_input(raw_df: pd.DataFrame, meta: dict):
    """Raw user data -> engineered features -> exactly the columns the model was trained on."""
    raw_df = raw_df.copy()
    for c in RAW_COLS:
        if c not in raw_df.columns:
            raw_df[c] = None
    raw_df["service_charge"] = pd.to_numeric(raw_df["service_charge"], errors="coerce")
    feats = engineer_features(raw_df[RAW_COLS])
    return feats, feats[meta["features"]]


def naira(x):
    return f"₦{x:,.0f}"


try:
    model, meta, typical_err = load_artifacts()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the model from `{MODEL_DIR}`. Run the training notebook first and copy the "
             f"`outputs_rent` folder next to this app.\n\nDetails: {e}")
    st.stop()

st.title("🏠 Lease Rent Predictor")
st.caption(f"Model: **{meta['winner']}** · predicts **annual rent (₦)** · "
           f"uses {len(meta['features'])} features: {', '.join(meta['features'])}")

tab_single, tab_batch = st.tabs(["Single lease", "Batch (CSV)"])

# ----------------------------------------------------------------------------- single
with tab_single:
    with st.form("single"):
        st.subheader("Property")
        address = st.text_input(
            "Property address *",
            placeholder="e.g. 5, Ayoade Azeez Close, Akingba Street, Baruwa, Ipaja, Lagos State",
            help="Use commas between parts, ending with area, city and state (as in your records).")
        c1, c2, c3, c4 = st.columns(4)
        unit = c1.selectbox("Unit type", UNIT_OPTIONS)
        bedrooms = c2.number_input("Bedrooms (0 = unknown)", 0, 20, 0)
        floor = c3.number_input("Floor (0 = unknown)", 0, 60, 0)
        c5, c6 = st.columns(2)
        furnished = c5.checkbox("Furnished / serviced")
        bq = c6.checkbox("Has BQ (boys' quarters)")

        st.subheader("Lease terms")
        d1, d2, d3 = st.columns(3)
        start = d1.date_input("Lease start date", value=date.today())
        months = d2.number_input("Lease length (months)", 1, 120, 12)
        service_charge = d3.number_input("Service charge (₦, 0 = none/unknown)", 0.0, value=0.0, step=10000.0)

        notes = st.text_area("Notes (optional)", placeholder="Any remarks recorded on the lease, e.g. FURNISHED",
                             height=80)
        submitted = st.form_submit_button("Predict annual rent", type="primary")

    if submitted:
        if not address.strip():
            st.warning("Please enter the property address.")
        else:
            # Fold the helper inputs into the address text so the same parser used in training reads them.
            prefix = []
            if bedrooms:
                prefix.append(f"{bedrooms} BEDROOM")
            if unit != UNIT_OPTIONS[0]:
                prefix.append(unit.upper())
            if floor:
                prefix.append(f"{floor}TH FLOOR")
            if furnished:
                prefix.append("FURNISHED")
            if bq:
                prefix.append("BQ")
            full_address = (" ".join(prefix) + " " + address.strip()).strip()

            end = (pd.Timestamp(start) + pd.DateOffset(months=int(months))).date()
            row = pd.DataFrame([{
                "property_name": full_address,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "service_charge": service_charge if service_charge > 0 else None,
                "notes": notes.strip() or None,
            }])
            feats, X = to_model_input(row, meta)
            pred = float(model.predict(X)[0])

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted annual rent", naira(pred))
            m2.metric("Monthly equivalent", naira(pred / 12))
            if typical_err:
                m3.metric("Typical error (test set)", f"±{typical_err:.0f}%",
                          help="Median absolute % error on held-out leases. A guide, not a guaranteed range.")
                lo, hi = pred * (1 - typical_err / 100), pred * (1 + typical_err / 100)
                st.info(f"Indicative range: **{naira(lo)} – {naira(hi)}** per year")

            with st.expander("How your input was interpreted"):
                p = parse_property(full_address)
                show = {k: p[k] for k in ["unit_type", "bedrooms", "floor", "street_type", "locality",
                                          "city", "state", "has_bq", "is_furnished"]}
                show.update(lease_months=feats["lease_months"].iloc[0],
                            addr_last2=feats["addr_last2"].iloc[0], addr_last3=feats["addr_last3"].iloc[0])
                st.json({k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                         for k, v in show.items()})
                st.caption("Areas the model never saw in training fall back to the average rent level, "
                           "so predictions for brand-new locations are less reliable.")

# ----------------------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a CSV with columns: **property_name, start_date, end_date** "
             "(optional: service_charge, notes). Dates as YYYY-MM-DD.")
    template = pd.DataFrame([{"property_name": "5, AYOADE AZEEZ CLOSE, AKINGBA STREET, BARUWA, IPAJA, LAGOS STATE",
                              "start_date": "2026-01-01", "end_date": "2027-01-01",
                              "service_charge": "", "notes": ""}])
    st.download_button("Download CSV template", template.to_csv(index=False), "lease_template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in ["property_name", "start_date", "end_date"] if c not in data.columns]
        if missing:
            st.error(f"Missing required column(s): {', '.join(missing)}")
        else:
            _, Xb = to_model_input(data, meta)
            out = data.copy()
            out["predicted_annual_rent"] = model.predict(Xb).round(0)
            out["predicted_monthly_equivalent"] = (out["predicted_annual_rent"] / 12).round(0)
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False),
                               "rent_predictions.csv", "text/csv")
