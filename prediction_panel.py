# prediction_panel.py
#
# Interactive "Predict New Event" panel for the recall dashboard.
# - Fits simple LinearRegression models from Dataset_B_Clean.csv
# - Lets user manually set P_* scores via sliders
# - Optional: auto-fill those scores from the headline using Gemini
#
# Requirements for auto-fill:
#   pip install google-generativeai scikit-learn
#   export GOOGLE_API_KEY="your_api_key_here"

import os
import json
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Try to import Gemini client; if not available, we just disable auto-fill
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# =============================================================
#  MODEL + DATA LOADER (cached)
# =============================================================

@st.cache_resource
def load_models_and_metadata() -> Tuple[Dict[str, LinearRegression], List[str]]:
    """
    Load Dataset_B_Clean.csv, fit 3 regression models for CAR_Tight,
    CAR_Wide, CAR_Delayed using the 4 P_* scores, and return:

        models: {
            "tight":   LinearRegression,
            "wide":    LinearRegression,
            "delayed": LinearRegression
        }

        predictor_cols: ordered list of feature names
    """
    # 1) Load dataset (same file dashboard.py uses)
    df = pd.read_csv("Dataset_B_Clean.csv")
    df["Event_Date"] = pd.to_datetime(df["Event_Date"], errors="coerce")

    # 2) Features and targets (must match your Dataset B schema)
    predictor_cols = ["P_Safety", "P_Trust", "P_ContinueBuying", "P_Recommend"]
    target_map = {
        "tight": "CAR_Tight",
        "wide": "CAR_Wide",
        "delayed": "CAR_Delayed",
    }

    # 3) Drop rows with missing CARs or predictors
    df_reg = df.dropna(subset=predictor_cols + list(target_map.values())).copy()

    # 4) Feature matrix
    X = df_reg[predictor_cols].astype(float)

    # 5) Fit one LinearRegression per CAR window
    models: Dict[str, LinearRegression] = {}
    for key, target_col in target_map.items():
        y = df_reg[target_col].astype(float)
        model = LinearRegression()
        model.fit(X, y)
        models[key] = model

    return models, predictor_cols


# =============================================================
#  LLM HELPER: headline → 4 scores (1–100)
# =============================================================

def llm_score_headline(
    ticker: str,
    event_date: dt.date,
    headline: str,
) -> Dict[str, float]:
    """
    Use Gemini to turn a recall headline into four 1–100 scores:
        P_Safety, P_Trust, P_ContinueBuying, P_Recommend

    Relies on:
      - google-generativeai installed
      - env var GOOGLE_API_KEY set
    """
    if not HAS_GENAI:
        raise RuntimeError(
            "google-generativeai is not installed. "
            "Run 'pip install google-generativeai' in your environment."
        )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    if isinstance(event_date, (dt.datetime, dt.date)):
        date_str = event_date.isoformat()
    else:
        date_str = str(event_date)

    prompt = f"""
You are simulating survey responses from a retail investor reacting to a product recall headline.

Company ticker: {ticker}
Event date: {date_str}
Headline: {headline}

Based only on this information (and no future knowledge), assign four scores from 1 to 100:

1. P_Safety          – perceived safety of using the company's products
2. P_Trust           – how much the investor would trust the company
3. P_ContinueBuying  – how likely they are to continue buying from the company
4. P_Recommend       – how likely they are to recommend the company to others

Rules:
- 1 = extremely negative, 100 = extremely positive.
- Output MUST be a single JSON object with numeric values, like:
  {{
    "P_Safety": 42,
    "P_Trust": 35,
    "P_ContinueBuying": 28,
    "P_Recommend": 30
  }}

Do not include any explanation or text outside the JSON.
"""

    response = model.generate_content(prompt)
    text = (response.text or "").strip()

    # Try to isolate JSON object in case the model adds extra text
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    json_str = match.group(0) if match else text

    data = json.loads(json_str)

    # Normalize and coerce to floats
    def clamp01_100(x):
        try:
            v = float(x)
        except Exception:
            v = np.nan
        if np.isnan(v):
            return np.nan
        return max(1.0, min(100.0, v))

    scores = {
        "P_Safety": clamp01_100(data.get("P_Safety")),
        "P_Trust": clamp01_100(data.get("P_Trust")),
        "P_ContinueBuying": clamp01_100(data.get("P_ContinueBuying")),
        "P_Recommend": clamp01_100(data.get("P_Recommend")),
    }

    # If anything came back NaN, bail with an error
    if any(np.isnan(v) for v in scores.values()):
        raise RuntimeError(f"Model returned invalid scores: {data}")

    return scores


# =============================================================
#  CORE HELPERS
# =============================================================

def build_feature_row_from_scores(
    scores: Dict[str, float],
    predictor_cols: List[str],
) -> pd.DataFrame:
    """
    Convert a dict of scores into a single-row DataFrame aligned with
    the regression model's expected predictor columns.
    """
    row = {}
    for col in predictor_cols:
        row[col] = float(scores.get(col, 0.0))

    X_new = pd.DataFrame([row], columns=predictor_cols)
    return X_new


def predict_car_from_scores(
    scores: Dict[str, float],
    models: Dict[str, LinearRegression],
    predictor_cols: List[str],
) -> Dict[str, float]:
    """
    Given sentiment-style scores and the trained models, return
    predicted CAR for tight, wide, and delayed windows.
    """
    X_new = build_feature_row_from_scores(scores, predictor_cols)

    car_tight = float(models["tight"].predict(X_new)[0])
    car_wide = float(models["wide"].predict(X_new)[0])
    car_delayed = float(models["delayed"].predict(X_new)[0])

    return {
        "CAR_Tight": car_tight,
        "CAR_Wide": car_wide,
        "CAR_Delayed": car_delayed,
    }


# =============================================================
#  STREAMLIT UI ENTRY POINT
# =============================================================

def render_prediction_panel():
    """
    Streamlit UI for the 'Predict New Recall Event Impact' section.

    Called from dashboard.py like:
        import prediction_panel

        ...

        else:
            prediction_panel.render_prediction_panel()
    """
    st.header("🔮 Predict Impact of a New Recall Event")

    # Load models + metadata once (cached across reruns)
    models, predictor_cols = load_models_and_metadata()

    # -----------------------------------
    # Basic event inputs (ticker, date, severity, headline)
    # -----------------------------------
    col1, col2 = st.columns(2)

    with col1:
        ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "TSLA"), key="ticker")
        event_date = st.date_input(
            "Event Date",
            value=st.session_state.get("event_date", dt.date.today()),
            key="event_date",
        )

    with col2:
        severity = st.selectbox(
            "Recall Severity (qualitative)",
            ["Low", "Medium", "High"],
            index=1,
            key="severity",
        )

    headline = st.text_area(
        "Recall Headline / Description",
        value=st.session_state.get("headline", ""),
        placeholder="e.g. 'Battery overheating risk leads to voluntary recall of Model X vehicles'",
        height=120,
        key="headline",
    )

    # -----------------------------------
    # Buttons: auto-fill + predict
    # -----------------------------------
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        auto_clicked = st.button("✨ Auto-fill scores from headline")
    with btn_col2:
        predict_clicked = st.button("📈 Predict Market Impact")

    # If auto-fill requested, try to call Gemini
        # -----------------------------------
    # Optional popover to capture GOOGLE_API_KEY
    # -----------------------------------
    if "show_key_popover" not in st.session_state:
        st.session_state["show_key_popover"] = False

    if st.session_state["show_key_popover"]:
        with st.popover("Click to enter your GOOGLE_API_KEY"):
            st.markdown(
                "This key is used to call **Gemini** for auto-filling scores. "
                "It is stored only in this Streamlit session."
            )
            user_api_key = st.text_input(
                "GOOGLE_API_KEY",
                type="password",
                key="google_api_key_input",
            )
            if user_api_key:
                # Save for this session and make llm_score_headline pick it up
                os.environ["GOOGLE_API_KEY"] = user_api_key
                st.session_state["GOOGLE_API_KEY"] = user_api_key
                st.success(
                    "API key saved. Click **Auto-fill scores from headline** again."
                )

        # If auto-fill requested, try to call Gemini
    if auto_clicked:
        if not headline.strip():
            st.warning("Please enter a headline before auto-filling scores.")
        else:
            # Check whether we already have an API key
            api_key = os.getenv("GOOGLE_API_KEY") or st.session_state.get("GOOGLE_API_KEY", "")

            if not api_key:
                # Turn on the popover and ask the user to enter the key
                st.session_state["show_key_popover"] = True
                st.warning(
                    "To use auto-fill, please enter your **GOOGLE_API_KEY** "
                    "using the popover above, then click the button again."
                )
            else:
                try:
                    scores_auto = llm_score_headline(
                        ticker=ticker,
                        event_date=event_date,
                        headline=headline,
                    )
                    # Update session_state so sliders pick up new defaults
                    for key, value in scores_auto.items():
                        st.session_state[key] = float(value)
                    st.success(
                        "Scores auto-filled from headline. You can tweak them below."
                    )
                except Exception as e:
                    st.error(f"Auto-scoring failed: {e}")


    # -----------------------------------
    # Sliders for P_* scores (manual + tweak auto-fill)
    # -----------------------------------
    # Initialize defaults if not present
    defaults = {
        "P_Safety": 70.0,
        "P_Trust": 65.0,
        "P_ContinueBuying": 60.0,
        "P_Recommend": 55.0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    st.markdown("### Perceived Consumer Scores (1–100)")
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.session_state["P_Safety"] = st.slider(
            "Safety",
            min_value=1,
            max_value=100,
            value=int(round(st.session_state["P_Safety"])),
        )
    with s2:
        st.session_state["P_Trust"] = st.slider(
            "Trust",
            min_value=1,
            max_value=100,
            value=int(round(st.session_state["P_Trust"])),
        )
    with s3:
        st.session_state["P_ContinueBuying"] = st.slider(
            "Continue Buying",
            min_value=1,
            max_value=100,
            value=int(round(st.session_state["P_ContinueBuying"])),
        )
    with s4:
        st.session_state["P_Recommend"] = st.slider(
            "Recommend",
            min_value=1,
            max_value=100,
            value=int(round(st.session_state["P_Recommend"])),
        )

    # -----------------------------------
    # Run prediction and show output
    # -----------------------------------
    if not predict_clicked:
        st.info("Adjust scores or auto-fill from the headline, then click **Predict Market Impact**.")
        return

    scores = {
        "P_Safety": float(st.session_state["P_Safety"]),
        "P_Trust": float(st.session_state["P_Trust"]),
        "P_ContinueBuying": float(st.session_state["P_ContinueBuying"]),
        "P_Recommend": float(st.session_state["P_Recommend"]),
    }

    preds = predict_car_from_scores(scores, models=models, predictor_cols=predictor_cols)

    st.subheader(f"Predicted CAR for {ticker} on {event_date.isoformat()}")

    # NOTE: If your CAR columns are stored as decimals (e.g. -0.052 for -5.2%),
    # multiply by 100 here instead.
    m1, m2, m3 = st.columns(3)
    m1.metric("Tight Window CAR", f"{preds['CAR_Tight']:.2f} %")
    m2.metric("Wide Window CAR", f"{preds['CAR_Wide']:.2f} %")
    m3.metric("Delayed Window CAR", f"{preds['CAR_Delayed']:.2f} %")

    with st.expander("Show features used for prediction"):
        feature_df = build_feature_row_from_scores(scores, predictor_cols)
        st.dataframe(feature_df)

