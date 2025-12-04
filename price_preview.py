# price_preview.py
#
# Price reaction mini-panel for the recall dashboard.
# - Uses yfinance to pull historical prices for ONE selected event
# - No CAR metrics here (those live on the left in Event Overview)
# - Just a price chart with a marker at the event date

from typing import Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


def _load_price_window(
    ticker: str,
    event_ts: pd.Timestamp,
    days_before: int = 30,
    days_after: int = 30,
) -> pd.DataFrame:
    """
    Use yfinance to pull historical prices around the event date.

    Returns a DataFrame with at least ["Date", "Adj Close"].
    """
    start = (event_ts - pd.Timedelta(days=days_before)).normalize()
    end = (event_ts + pd.Timedelta(days=days_after)).normalize()

    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # include end day
        progress=False,
    )

    if data.empty:
        return data

    data = data.reset_index()  # Date column

    # Ensure we have consistent column names
    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    return data


def render_price_preview(event_row: Union[pd.Series, pd.DataFrame]) -> None:
    """
    Streamlit UI block for the right-column price reaction preview.

    Expects a **single event row** with at least:
      - "Ticker"
      - "Event_Date"

    You will pass this from dashboard3.py, e.g.:

        focus_event = ...
        with right_col:
            price_preview.render_price_preview(focus_event)
    """
    st.write("### Price Reaction (Preview)")

    # Allow either a Series or a 1-row DataFrame
    if isinstance(event_row, pd.DataFrame):
        if event_row.empty:
            st.info("No event selected.")
            return
        event_row = event_row.iloc[0]

    # Ticker and event date
    ticker = str(event_row.get("Ticker", "")).upper()

    if not ticker:
        st.info("No ticker available for this event.")
        return

    event_ts = pd.to_datetime(event_row["Event_Date"]).normalize()
    event_date = event_ts.to_pydatetime()

    # ---------------------------
    # Load price data via yfinance
    # ---------------------------
    with st.spinner(f"Loading {ticker} price data around {event_date.date()}..."):
        price_df = _load_price_window(ticker, event_ts, days_before=30, days_after=30)

    if price_df.empty:
        st.warning(f"No price data found for {ticker} around {event_date.date()}.")
        return

    # ---------------------------
    # Price chart with event marker
    # ---------------------------
    fig = px.line(
        price_df,
        x="Date",
        y="Adj Close",
        title=f"{ticker} Price Around Event ({event_date.date()})",
    )

    # Make sure Date is datetime
    price_df["Date"] = pd.to_datetime(price_df["Date"])

    # Find the price at or just before the event date
    mask = price_df["Date"] <= event_ts
    if mask.any():
        y_event = float(price_df.loc[mask, "Adj Close"].iloc[-1])

        fig.add_trace(
            go.Scatter(
                x=[event_date],
                y=[y_event],
                mode="markers",
                marker=dict(size=10, symbol="diamond"),
                name="Event",
            )
        )

    fig.update_layout(
        height=280,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Note: price data fetched live via yfinance; CAR values from your regression dataset.")

