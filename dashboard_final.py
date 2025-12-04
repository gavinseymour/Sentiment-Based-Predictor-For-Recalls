# ================================================================
#  DASHBOARD3.PY — Historical Explorer + Prediction Panel
# ================================================================

pip install plotly

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import prediction_panel
import price_preview


# ================================================================
#  PAGE CONFIG
# ================================================================
st.set_page_config(page_title="Recall Event Explorer", layout="wide")


# ================================================================
#  DATA LOADING
# ================================================================
@st.cache_data
def load_dataset_b(path: str = "Dataset_B_Clean.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Event_Date"] = pd.to_datetime(df["Event_Date"], errors="coerce")
    return df


dataset_b = load_dataset_b()

# --- Load Dataset C: coefficient t-stats by window ---
@st.cache_data
def load_dataset_c():
    return pd.read_csv("Dataset_C.csv")

dataset_c = load_dataset_c()


# ================================================================
#  SIDEBAR
# ================================================================
st.sidebar.title("Recall Event Explorer")

page = st.sidebar.radio(
    "Page",
    ["Explore Historical Events", "Predict New Event"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters (Historical Page)**")

all_tickers = sorted(dataset_b["Ticker"].dropna().unique().tolist())
selected_tickers = st.sidebar.multiselect(
    "Tickers",
    options=all_tickers,
    default=all_tickers,
)

min_date = dataset_b["Event_Date"].min()
max_date = dataset_b["Event_Date"].max()

date_range = st.sidebar.date_input(
    "Event date range",
    value=(min_date.date(), max_date.date()) if pd.notna(min_date) else None,
)


# ================================================================
#  PAGE 1: EXPLORE HISTORICAL EVENTS
# ================================================================
if page == "Explore Historical Events":

    st.title("📊 Historical Recall Events")

    # ----------------------------------------
    # Apply filters
    # ----------------------------------------
    df = dataset_b.copy()

    if selected_tickers:
        df = df[df["Ticker"].isin(selected_tickers)]

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df["Event_Date"] >= pd.to_datetime(start_date)) &
                (df["Event_Date"] <= pd.to_datetime(end_date))]

    filtered = df.sort_values("Event_Date")

    if filtered.empty:
        st.warning("No events match your current filters.")
        st.stop()

    # ----------------------------------------
    # Event selector (returns a **row**)
    # ----------------------------------------
    st.subheader("Select Event")

    def format_event(idx):
        row = filtered.loc[idx]
        date_str = pd.to_datetime(row["Event_Date"]).date()
        reason = str(row.get("Reason_For_Recall", ""))[:40]
        return f"{row['Ticker']} — {date_str} — {reason}"

    selected_idx = st.selectbox(
        "Choose an event to inspect",
        options=filtered.index,
        format_func=format_event,
    )

    event_data = filtered.loc[selected_idx]  # <-- this is the row we use everywhere

    # ----------------------------------------
    # Event Overview layout
    # ----------------------------------------
    st.markdown("---")
    st.subheader("Event Overview")

    left_col, right_col = st.columns([2, 3])

    # ================= LEFT COLUMN =================
    with left_col:
        ticker = event_data["Ticker"]
        event_date = pd.to_datetime(event_data["Event_Date"]).date()

        st.write(f"### **{ticker} — {event_date}**")

        st.markdown("#### Event CAR by Window")

        car_tight = event_data.get("CAR_Tight", float("nan"))
        car_wide = event_data.get("CAR_Wide", float("nan"))
        car_delayed = event_data.get("CAR_Delayed", float("nan"))

        # If CARs are stored as decimals, uncomment to scale:
        # car_tight *= 100
        # car_wide *= 100
        # car_delayed *= 100

        st.write("**Tight (−1 to +1)**")
        st.markdown(f"<h2 style='margin-top:0;'>{car_tight:.3f}%</h2>", unsafe_allow_html=True)

        st.write("**Wide (−3 to +5)**")
        st.markdown(f"<h2 style='margin-top:0;'>{car_wide:.3f}%</h2>", unsafe_allow_html=True)

        st.write("**Delayed (+3 to +30)**")
        st.markdown(f"<h2 style='margin-top:0;'>{car_delayed:.3f}%</h2>", unsafe_allow_html=True)

        st.markdown("#### Headline / Reason")
        st.write(str(event_data.get("Reason_For_Recall", "")))

    # ================= RIGHT COLUMN =================
    with right_col:
        # Price reaction preview for the **same event row**
        price_preview.render_price_preview(event_data)

    st.divider()

    # -------------------------------------------------------------
    # Comparison: Actual vs Predicted CAR (bar chart)
    # -------------------------------------------------------------
    st.subheader("Actual vs Predicted CAR")

    comparison = pd.DataFrame({
        "Window": ["Tight", "Wide", "Delayed"],
        "Actual CAR": [
            event_data.get("CAR_Tight"),
            event_data.get("CAR_Wide"),
            event_data.get("CAR_Delayed"),
        ],
        "Predicted CAR": [
            event_data.get("Pred_CAR_Tight"),
            event_data.get("Pred_CAR_Wide"),
            event_data.get("Pred_CAR_Delayed"),
        ],
    })

    fig_car = go.Figure()
    fig_car.add_trace(go.Bar(
        name="Actual",
        x=comparison["Window"],
        y=comparison["Actual CAR"],
    ))
    fig_car.add_trace(go.Bar(
        name="Predicted",
        x=comparison["Window"],
        y=comparison["Predicted CAR"],
    ))

    fig_car.update_layout(
        barmode="group",
        title="Actual vs Predicted CAR",
        yaxis_title="Cumulative Abnormal Return (%)",
    )

    st.plotly_chart(fig_car, use_container_width=True)
    st.divider()

    # -------------------------------------------------------------
    # Residuals table
    # -------------------------------------------------------------
    st.subheader("Residuals (Actual − Predicted)")

    residuals_df = pd.DataFrame({
        "Window": ["Tight", "Wide", "Delayed"],
        "Residual": [
            event_data.get("Resid_Tight"),
            event_data.get("Resid_Wide"),
            event_data.get("Resid_Delayed"),
        ],
    })

    st.table(residuals_df.style.format({"Residual": "{:.4f}"}))
    st.divider()
    
    st.markdown("Prediction Coefficient Significance (t-stats)")
    # Pivot so rows = predictors, columns = windows, values = t-stats
    tstats_pivot = (
    dataset_c
    .pivot(index="Predictor", columns="Window", values="t_stat")
    .reindex(index=["P_Safety", "P_Trust", "P_ContinueBuying", "P_Recommend"])
    )

    st.dataframe(tstats_pivot.round(3),use_container_width=True)

    # -------------------------------------------------------------
    # Full table of events (for current filters)
    # -------------------------------------------------------------
    st.subheader("Full Event Table (Filtered)")
    st.dataframe(
        filtered[[
            "Ticker",
            "Event_Date",
            "Reason_For_Recall",
            "CAR_Tight",
            "Pred_CAR_Tight",
            "Resid_Tight",
        ]],
        use_container_width=True,
    )

# ================================================================
#  PAGE 2: PREDICT NEW EVENT
# ================================================================
else:
    prediction_panel.render_prediction_panel()

