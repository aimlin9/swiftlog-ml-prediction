"""SwiftLog Delivery Delay Predictor: a Streamlit demo of the Week 5 ML model.

Run from the App folder:   streamlit run app.py

The model is loaded from ../Models/Best_Model.pkl (or App/Best_Model.pkl when deployed on its own).
IMPORTANT: on the held-out test set this model's ROC-AUC is not distinguishable from 0.5, so the app is a
demonstration of the workflow (input -> features -> prediction -> confidence), not an operational tool. The
app says so on screen, next to every prediction.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATHS = [HERE / "Best_Model.pkl", HERE.parent / "Models" / "Best_Model.pkl"]

st.set_page_config(page_title="SwiftLog Delay Predictor", page_icon=None, layout="centered")


@st.cache_resource
def load_bundle():
    for p in MODEL_PATHS:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError("Best_Model.pkl not found. Run the notebook first to create it.")


def build_features(bundle, inp):
    """Turn user inputs into the exact feature row the pipeline was trained on."""
    wh = bundle["warehouse_profile"][inp["warehouse"]]
    cats = inp["categories"] or bundle["categories"][:1]
    share = {f"share_{c}": (1.0 / len(cats) if c in cats else 0.0) for c in bundle["categories"]}
    qty, price = inp["qty"], inp["avg_price"]
    row = {
        "n_lines": inp["n_lines"], "qty": qty, "value": qty * price, "avg_price": price,
        "max_price": inp["max_price"], "n_categories": len(cats), "has_items": 1,
        "month": inp["date"].month, "dow": inp["date"].weekday(), "is_weekend": int(inp["date"].weekday() >= 5),
        "day_of_month": inp["date"].day, "cust_prior_orders": inp["prior_orders"],
        "cust_prior_delay_rate": inp["prior_delay_rate"] if inp["prior_orders"] > 0 else np.nan,
        "wh_headcount": wh["wh_headcount"], "wh_total_stock": wh["wh_total_stock"],
        "wh_low_stock_share": wh["wh_low_stock_share"], "wh_load": wh["wh_load"],
        "wh_stock_per_shipment": wh["wh_stock_per_shipment"],
        "WarehouseID": inp["warehouse"], "wh_location": wh["wh_location"], "City": inp["city"],
        **share,
    }
    return pd.DataFrame([row])[bundle["feature_columns"]]


def predict(bundle, X):
    proba = float(bundle["model"].predict_proba(X)[0, 1])
    return proba


def confidence_label(proba, typical):
    gap = abs(proba - typical)
    if gap < 0.05:
        return "Very low: this is essentially the model's typical output for any order"
    if gap < 0.15:
        return "Low"
    return "Moderate"


def main():
    st.title("SwiftLog Delivery Delay Predictor")
    st.caption("Week 5 | Machine Learning Track | AnalystLab Africa Experience Lab. Fictional company and synthetic data.")

    try:
        bundle = load_bundle()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    lo, hi = bundle["test_auc_ci"]
    st.warning(
        f"**Read this first.** On held-out test data this model's ROC-AUC is {bundle['test_metrics']['roc_auc']:.2f} "
        f"(95% interval {lo:.2f} to {hi:.2f}), which is not distinguishable from random guessing (0.50). "
        "The app demonstrates the workflow; its predictions should not be used for operational decisions."
    )

    st.subheader("Order details")
    c1, c2 = st.columns(2)
    with c1:
        warehouse = st.selectbox("Dispatch warehouse", sorted(bundle["warehouse_profile"]))
        city = st.selectbox("Customer city", bundle["cities"])
        date = st.date_input("Order date", value=pd.Timestamp("2026-06-15"))
        categories = st.multiselect("Product categories in the order", bundle["categories"], default=bundle["categories"][:1])
    with c2:
        n_lines = st.number_input("Order lines", 1, 30, 3)
        qty = st.number_input("Total units", 1, 300, 15)
        avg_price = st.number_input("Average unit price", 1.0, 1000.0, 260.0, step=10.0)
        max_price = st.number_input("Highest unit price in the order", 1.0, 1000.0, 400.0, step=10.0)
    c3, c4 = st.columns(2)
    prior_orders = c3.number_input("Customer's earlier orders", 0, 100, 4)
    prior_delay_rate = c4.slider("Share of those earlier orders that were delayed", 0.0, 1.0, 0.64, disabled=prior_orders == 0)

    inp = dict(warehouse=warehouse, city=city, date=date, categories=categories, n_lines=n_lines, qty=qty,
               avg_price=avg_price, max_price=max(max_price, avg_price), prior_orders=prior_orders, prior_delay_rate=prior_delay_rate)

    if st.button("Predict delay risk", type="primary"):
        try:
            proba = predict(bundle, build_features(bundle, inp))
        except Exception as e:  # never show a raw stack trace to the user
            st.error(f"Could not generate a prediction: {e}")
            st.stop()
        base = bundle.get("typical_proba", 0.5)
        st.subheader("Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Delay probability (3+ days)", f"{proba:.1%}")
        m2.metric("Typical model output", f"{base:.1%}", help=f"Actual delay rate in the data is {bundle['base_rate']:.1%}. The model is class-weighted, so its outputs centre near 50%.")
        m3.metric("Predicted class", "Delayed" if proba >= 0.5 else "On time")
        st.progress(min(max(proba, 0.0), 1.0))
        st.write(f"**Prediction confidence:** {confidence_label(proba, base)}.")
        st.info(
            "A probability close to the typical model output means the model has no real information about this order. "
            "That is the expected outcome here: the available fields do not predict delays."
        )

    with st.expander("About this model"):
        st.write(f"**Algorithm:** {bundle['model_name']}  \n**Target:** {bundle['target']}  \n**Version:** {bundle['version']}  \n**Trained:** {bundle['trained_at']}")
        st.write("Delay is defined as 3 or more days late (the rule used in SwiftLog's Delivery_Performance table). "
                 "Features use only information available before dispatch.")


if __name__ == "__main__":
    main()
