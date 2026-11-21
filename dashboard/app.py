
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "https://kenya-maize-price-forecasting-1.onrender.com"

st.set_page_config(
    page_title="Kenya Maize Forecasting | Analytics",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}
[data-testid="stMetric"] {
    padding: 18px;
    border: 1px solid #8884;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def get_models():
    response = requests.get(f"{API_URL}/models", timeout=120)
    response.raise_for_status()
    return response.json()


def get_forecast(payload):
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


st.title("🌽 Kenya Maize Price Forecasting")
st.caption(
    "Advanced V2 | Forecasting, Model Evaluation "
    "& Uncertainty Analytics"
)

st.markdown(
    "An interactive decision-support application for exploring "
    "maize price forecasts in Kenya."
)

with st.sidebar:
    st.header("Forecast Configuration")

    market = st.text_input("Market", value="Nairobi")

    current_price = st.number_input(
        "Current maize price (model units)",
        min_value=0.01,
        value=50.0,
        step=1.0,
    )

    origin_year = st.number_input(
        "Origin year",
        min_value=2007,
        max_value=2100,
        value=2026,
        step=1,
    )

    origin_month = st.selectbox(
        "Origin month",
        options=list(range(1, 13)),
        index=7,
        format_func=lambda x: pd.Timestamp(
            year=2026, month=x, day=1
        ).strftime("%B"),
    )

    run_forecast = st.button(
        "Generate All Forecasts",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.caption("Production model: Persistence")
    st.caption("Forecast horizons: 1, 2 and 3 months")


if run_forecast:
    results = []
    errors = []

    with st.spinner("Generating forecasts for all horizons..."):
        for horizon in [1, 2, 3]:
            payload = {
                "market": market.strip(),
                "current_price": float(current_price),
                "origin_year": int(origin_year),
                "origin_month": int(origin_month),
                "horizon": horizon,
            }

            try:
                results.append(get_forecast(payload))
            except (requests.RequestException, ValueError) as error:
                errors.append(f"{horizon}-month: {error}")

    if errors:
        for error in errors:
            st.error(error)

    if results:
        st.session_state["forecasts"] = results
    else:
        st.session_state.pop("forecasts", None)


tab1, tab2, tab3 = st.tabs([
    "📈 Forecast Explorer",
    "📊 Model Performance",
    "🎯 Uncertainty Analysis",
])

with tab1:
    st.subheader("Multi-Horizon Forecast Explorer")

    if "forecasts" not in st.session_state:
        st.info(
            "Configure the inputs in the sidebar and click "
            "'Generate All Forecasts'."
        )
    else:
        results = st.session_state["forecasts"]

        records = []
        for result in results:
            records.append({
                "Market": result["market"],
                "Origin": result["origin_date"],
                "Target": result["target_date"],
                "Horizon": result["horizon_months"],
                "Forecast": result["predicted_price"],
                "Lower 80%": result["interval_80"]["lower"],
                "Upper 80%": result["interval_80"]["upper"],
                "Model": result["model"],
            })

        df = pd.DataFrame(records).sort_values("Horizon")

        cols = st.columns(len(df))
        for col, (_, row) in zip(cols, df.iterrows()):
            col.metric(
                f"{int(row['Horizon'])}-Month Forecast",
                f"{row['Forecast']:.2f}",
            )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["Target"],
            y=df["Upper 80%"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=df["Target"],
            y=df["Lower 80%"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(22,163,74,0.18)",
            line=dict(width=0),
            name="80% Prediction Interval",
        ))

        fig.add_trace(go.Scatter(
            x=df["Target"],
            y=df["Forecast"],
            mode="lines+markers",
            name="Point Forecast",
            line=dict(color="#16a34a", width=3),
            marker=dict(size=10),
        ))

        fig.update_layout(
            title="Forecast Trajectory and Uncertainty",
            xaxis_title="Target Month",
            yaxis_title="Price (model units)",
            hovermode="x unified",
            height=450,
            margin=dict(l=20, r=20, t=60, b=30),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Comparison Table")
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download Forecast Comparison (CSV)",
            data=df.to_csv(index=False),
            file_name="kenya_maize_forecast_comparison.csv",
            mime="text/csv",
        )

        st.warning(
            results[0].get(
                "warning",
                "Forecast uncertainty is not guaranteed.",
            )
        )


try:
    models = get_models()
except (requests.RequestException, ValueError) as error:
    models = None
    st.error(f"Unable to retrieve model metrics: {error}")


if models:
    rows = []

    for horizon, info in sorted(
        models["horizons"].items(),
        key=lambda item: int(item[0]),
    ):
        rows.append({
            "Horizon": int(horizon),
            "Model": info["selected_model"],
            "MAE": info["test_mae"],
            "RMSE": info["test_rmse"],
            "Nominal Coverage": (
                info["interval_nominal_coverage"] * 100
            ),
            "Empirical Coverage": (
                info["interval_empirical_coverage"] * 100
            ),
        })

    metrics_df = pd.DataFrame(rows)

    with tab2:
        st.subheader("Production Model Evaluation")

        st.info(
            f"Selected production model: "
            f"{models['production_model'].title()}"
        )

        col1, col2 = st.columns(2)

        with col1:
            fig_mae = go.Figure(go.Bar(
                x=metrics_df["Horizon"].astype(str),
                y=metrics_df["MAE"],
                marker_color="#16a34a",
            ))
            fig_mae.update_layout(
                title="Test Mean Absolute Error",
                xaxis_title="Forecast Horizon (months)",
                yaxis_title="MAE",
                height=350,
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        with col2:
            fig_rmse = go.Figure(go.Bar(
                x=metrics_df["Horizon"].astype(str),
                y=metrics_df["RMSE"],
                marker_color="#f59e0b",
            ))
            fig_rmse.update_layout(
                title="Test Root Mean Squared Error",
                xaxis_title="Forecast Horizon (months)",
                yaxis_title="RMSE",
                height=350,
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

        st.dataframe(
            metrics_df,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown(
            "**Interpretation:** Forecast errors increase "
            "as the prediction horizon becomes longer. "
            "Persistence was selected based on the "
            "model evaluation results."
        )

        st.caption(
            "Research comparators: "
            + ", ".join(models.get("research_comparators", []))
        )

    with tab3:
        st.subheader("Prediction Interval Reliability")

        fig_coverage = go.Figure()

        fig_coverage.add_trace(go.Bar(
            x=metrics_df["Horizon"].astype(str),
            y=metrics_df["Empirical Coverage"],
            name="Empirical Coverage",
            marker_color="#16a34a",
        ))

        fig_coverage.add_hline(
            y=80,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="Nominal 80% coverage",
        )

        fig_coverage.update_layout(
            title="Empirical vs Nominal Prediction Interval Coverage",
            xaxis_title="Forecast Horizon (months)",
            yaxis_title="Coverage (%)",
            yaxis=dict(range=[0, 100]),
            height=400,
        )

        st.plotly_chart(
            fig_coverage,
            use_container_width=True,
        )

        st.markdown(
            "An 80% prediction interval is intended to "
            "contain approximately 80% of future outcomes "
            "under comparable evaluation conditions. "
            "Actual coverage can differ from this target."
        )

        st.dataframe(
            metrics_df[[
                "Horizon",
                "Nominal Coverage",
                "Empirical Coverage",
            ]],
            hide_index=True,
            use_container_width=True,
        )

        st.warning(
            models.get(
                "source_warning",
                "Forecasts are estimates, not guarantees.",
            )
        )

st.divider()
st.caption(
    "MSc Data Science & Analytics | Strathmore University | "
    "Kenya Maize Price Forecasting — Advanced V2"
)
