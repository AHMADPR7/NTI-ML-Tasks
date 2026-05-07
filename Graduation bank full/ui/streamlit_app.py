from __future__ import annotations

import os

import requests
import streamlit as st

from bank_predictor.config import API_URL_ENV, DEFAULT_API_URL, FEATURE_FIELDS

REQUEST_TIMEOUT_SECONDS = 10


@st.cache_data(show_spinner=False)
def fetch_metadata(api_url: str) -> dict:
    response = requests.get(f"{api_url}/metadata", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def load_example(example_input: dict[str, object]) -> None:
    for field, value in example_input.items():
        st.session_state[field] = value


def render_prediction(result: dict) -> None:
    label = result["label"]
    probability_yes = result["probability_yes"]
    probability_no = result["probability_no"]
    decision_threshold = result["decision_threshold"]
    confidence = probability_yes if label == "yes" else probability_no

    st.markdown(
        f"""
        <div style="padding: 1rem 1.2rem; border-radius: 16px; background: linear-gradient(135deg, #f4efe6, #dce9df); border: 1px solid #c8d8ca; margin-bottom: 1rem;">
            <div style="font-size: 0.9rem; letter-spacing: 0.08em; text-transform: uppercase; color: #47624f;">Prediction</div>
            <div style="font-size: 2rem; font-weight: 700; color: #153126;">{label.upper()}</div>
            <div style="font-size: 1rem; color: #264635;">Confidence: {confidence:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(2)
    metric_columns[0].metric("Probability: yes", f"{probability_yes:.2%}")
    metric_columns[1].metric("Probability: no", f"{probability_no:.2%}")
    st.caption(f"Decision threshold for yes: {decision_threshold:.2f}")

    with st.expander("Submitted input", expanded=True):
        st.json(result["normalized_input"])


def render_metrics(model_name: str, decision_threshold: float, metrics: dict) -> None:
    st.subheader("Saved Model Metrics")
    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    columns[1].metric("Precision", f"{metrics['precision']:.3f}")
    columns[2].metric("Recall", f"{metrics['recall']:.3f}")
    columns[3].metric("F1", f"{metrics['f1']:.3f}")
    st.caption(
        f"Model: {model_name} | Decision threshold: {decision_threshold:.2f} | "
        f"Positive sample weight: {metrics['positive_sample_weight']:.1f}"
    )

    cm = metrics["confusion_matrix"]
    st.caption(
        "Confusion matrix counts: "
        f"TN={cm['true_negative']} | FP={cm['false_positive']} | "
        f"FN={cm['false_negative']} | TP={cm['true_positive']}"
    )



def main() -> None:
    st.set_page_config(page_title="Bank Marketing Predictor", page_icon="B", layout="wide")

    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(237, 230, 214, 0.9), transparent 35%),
                    linear-gradient(180deg, #f7f3ea 0%, #eef4ee 100%);
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    api_url = st.sidebar.text_input("FastAPI URL", value=os.getenv(API_URL_ENV, DEFAULT_API_URL)).rstrip("/")
    st.sidebar.caption("Run the API first, then use this form for single-record predictions.")

    st.title("Bank Marketing Subscription Predictor")
    st.caption("Single-customer prediction UI backed by FastAPI and a saved sklearn pipeline.")

    try:
        metadata = fetch_metadata(api_url)
    except requests.RequestException as exc:
        st.error(f"Could not reach the FastAPI service at {api_url}. Details: {exc}")
        st.info("Start the API with: uvicorn app.main:app --host 127.0.0.1 --port 8000")
        return

    header_left, header_right = st.columns([3, 1])
    header_left.subheader("Customer Details")
    if header_right.button("Load Example Customer", use_container_width=True):
        load_example(metadata["example_input"])

    field_labels = metadata["field_labels"]
    categorical_options = metadata["categorical_options"]
    numeric_fields = metadata["numeric_fields"]

    with st.form("prediction_form"):
        left_column, right_column = st.columns(2)
        payload: dict[str, object] = {}

        for index, field in enumerate(FEATURE_FIELDS):
            container = left_column if index % 2 == 0 else right_column
            label = field_labels[field]

            if field in categorical_options:
                options = categorical_options[field]
                default_value = st.session_state.get(field, metadata["example_input"].get(field, options[0]))
                if default_value not in options:
                    default_value = options[0]
                payload[field] = container.selectbox(
                    label,
                    options=options,
                    index=options.index(default_value),
                    key=field,
                )
            else:
                numeric_config = numeric_fields[field]
                default_value = int(st.session_state.get(field, numeric_config["default"]))
                payload[field] = int(
                    container.number_input(
                        label,
                        min_value=int(numeric_config["min"]),
                        max_value=int(numeric_config["max"]),
                        value=default_value,
                        step=1,
                        key=field,
                    )
                )

        submitted = st.form_submit_button("Predict Subscription", use_container_width=True)

    if submitted:
        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")
        else:
            render_prediction(response.json())

    st.divider()
    render_metrics(metadata["model_name"], metadata["decision_threshold"], metadata["metrics"])


if __name__ == "__main__":
    main()
