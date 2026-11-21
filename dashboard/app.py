
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "https://kenya-maize-price-forecasting-1.onrender.com"

st.set_page_config(
    page_title="Kenya Maize Price Forecasting",
    page_icon="🌽",
    layout="wide"
)

st.title("🌽 Kenya Maize Price Forecasting")
st.subheader("Advanced V2 | Interactive Forecasting Dashboard")

st.info(
    "Generate maize price forecasts for 1, 2, or 3 months "
    "using the deployed forecasting API."
)

with st.sidebar:
    st.header("Forecast Settings")

    market = st.text_input("Market", value="Nairobi")

    current_price = st.number_input(
        "Current maize price",
        min_value=0.01,
        value=50.0,
        step=1.0
    )

    origin_year = st.number_input(
        "Forecast origin year",
        min_value=2007,
        max_value=2100,
        value=2026,
        step=1
    )

    origin_month = st.selectbox(
        "Forecast origin month",
        options=list(range(1, 13)),
        index=7
    )

    horizon = st.selectbox(
        "Forecast horizon (months)",
        options=[1, 2, 3]
    )

    predict = st.button(
        "Generate Forecast",
        type="primary",
        use_container_width=True
    )

st.divider()

if predict:
    payload = {
        "market": market,
        "current_price": current_price,
        "origin_year": int(origin_year),
        "origin_month": int(origin_month),
        "horizon": horizon
    }

    try:
        with st.spinner("Generating forecast..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

        predicted = result["predicted_price"]
        lower = result["interval_80"]["lower"]
        upper = result["interval_80"]["upper"]

        st.success("Forecast generated successfully!")

        col1, col2, col3 = st.columns(3)

        col1.metric("Predicted Price", f"{predicted:.2f}")
        col2.metric("Lower Bound (80%)", f"{lower:.2f}")
        col3.metric("Upper Bound (80%)", f"{upper:.2f}")

        st.subheader("Forecast Visualization")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=["Current", "Forecast"],
            y=[current_price, predicted],
            mode="lines+markers",
            name="Point Forecast",
            line=dict(color="#15803d", width=3)
        ))

        fig.add_trace(go.Scatter(
            x=["Forecast", "Forecast"],
            y=[lower, upper],
            mode="lines+markers",
            name="80% Prediction Interval",
            line=dict(color="#f59e0b", width=4)
        ))

        fig.update_layout(
            title=f"{market}: {horizon}-Month Maize Price Forecast",
            xaxis_title="Forecast Period",
            yaxis_title="Price (model units)",
            template="plotly_white",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Details")

        details = {
            "Market": result["market"],
            "Origin Date": result["origin_date"],
            "Target Date": result["target_date"],
            "Forecast Horizon": result["horizon_months"],
            "Model": result["model"],
            "Estimated Test Coverage": (
                f"{result['estimated_test_coverage'] * 100:.2f}%"
            )
        }

        st.dataframe(
            pd.DataFrame(
                details.items(),
                columns=["Metric", "Value"]
            ),
            hide_index=True,
            use_container_width=True
        )

        st.warning(result.get("warning", ""))

        csv = pd.DataFrame([{
            **payload,
            **{
                "predicted_price": predicted,
                "lower_80": lower,
                "upper_80": upper
            }
        }]).to_csv(index=False)

        st.download_button(
            "Download Forecast CSV",
            data=csv,
            file_name="maize_forecast.csv",
            mime="text/csv"
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Unable to reach forecasting API: {error}")
    except (KeyError, TypeError, ValueError) as error:
        st.error(f"Unexpected API response: {error}")

st.divider()
st.caption(
    "MSc Data Science & Analytics | Strathmore University | "
    "Forecasts use a model-estimated price series."
)
