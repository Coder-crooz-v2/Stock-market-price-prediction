import streamlit as st
import os
import pandas as pd
from model import StockData
from keras.models import load_model

def get_stock_list():
    stocks = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "PYPL", "NFLX", "INTC",
        "AMD", "CSCO", "ADBE", "ORCL", "IBM", "UBER", "ZM", "SPOT", "SHOP"
    ]
    return stocks

def toggle_ma_100():
    st.session_state.show_ma_100 = not st.session_state.show_ma_100

def toggle_ma_200():
    st.session_state.show_ma_200 = not st.session_state.show_ma_200

def execute_prediction(selected_stock):
    if 'show_ma_100' not in st.session_state:
        st.session_state.show_ma_100 = True
    if 'show_ma_200' not in st.session_state:
        st.session_state.show_ma_200 = True
    
    stock_predictor = StockData()
    df = stock_predictor.import_data(selected_stock)
    indicators_df = stock_predictor.compute_technical_indicators(df.copy())
    indicator_snapshot = stock_predictor.summarize_latest_indicators(indicators_df)
    indicator_fig = stock_predictor.plot_indicator_overview(indicators_df)

    price_tab, indicator_tab, benchmark_tab = st.tabs([
        "Price History & LSTM", "Technical Indicators", "Benchmark Models"
    ])

    with price_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Stock Price History")
        with col2:
            col3, col4 = st.columns(2)
            with col3:
                st.button(
                    f"{'Hide' if st.session_state.show_ma_100 else 'Show'} 100-Day Moving Average", 
                    on_click=toggle_ma_100,
                    key="toggle_ma_100"
                )
            with col4:
                st.button(
                    f"{'Hide' if st.session_state.show_ma_200 else 'Show'} 200-Day Moving Average", 
                    on_click=toggle_ma_200,
                    key="toggle_ma_200"
                )
        fig = stock_predictor.plot_data(df, st.session_state.show_ma_100, st.session_state.show_ma_200)
        st.pyplot(fig)

        if not os.path.exists('model.keras'):
            st.warning("Training model...")
            x_train, y_train = stock_predictor.prepare_training_data(df)
            model = stock_predictor.prepare_model(x_train)
            model = stock_predictor.train_model(model, x_train, y_train)
            model.save('model.keras')
        model = load_model('model.keras')
        test_data, predictions = stock_predictor.predict(df, model)
        
        st.subheader("Actual vs Predicted Prices")
        
        # Styled metrics display
        lstm_metrics = stock_predictor.get_metrics(test_data, predictions)
        st.markdown("#### 📊 LSTM Model Performance Metrics")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("RMSE", f"{lstm_metrics['RMSE']:.4f}", help="Root Mean Squared Error - Lower is better")
        with metric_cols[1]:
            st.metric("MAE", f"{lstm_metrics['MAE']:.4f}", help="Mean Absolute Error - Lower is better")
        with metric_cols[2]:
            st.metric("MAPE", f"{lstm_metrics['MAPE']:.2f}%", help="Mean Absolute Percentage Error - Lower is better")
        with metric_cols[3]:
            r2_delta = "Good" if lstm_metrics['R2'] > 0.8 else ("Fair" if lstm_metrics['R2'] > 0.5 else "Poor")
            st.metric("R² Score", f"{lstm_metrics['R2']:.4f}", delta=r2_delta, help="Coefficient of determination - Higher is better (max 1.0)")
        
        fig = stock_predictor.plot_predictions(predictions, test_data)
        st.pyplot(fig)
        
        # Store LSTM results for benchmark tab
        lstm_benchmark_result = stock_predictor.get_lstm_benchmark_result(test_data, predictions)

    with indicator_tab:
        st.subheader("Technical Indicator Dashboard")
        st.pyplot(indicator_fig)
        st.markdown("### Latest Indicator Readings")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RSI", f"{indicator_snapshot['RSI']:.2f}", help="Relative Strength Index (14-period)")
            st.metric("MACD", f"{indicator_snapshot['MACD']:.2f}")
        with col2:
            st.metric("MACD Signal", f"{indicator_snapshot['MACD_SIGNAL']:.2f}")
            st.metric("MACD Histogram", f"{indicator_snapshot['MACD_HIST']:.2f}")
        with col3:
            st.metric("Upper Band", f"${indicator_snapshot['BB_UPPER']:.2f}")
            st.metric("Lower Band", f"${indicator_snapshot['BB_LOWER']:.2f}")
        st.caption("Bollinger Bands calculated with 20-period SMA and 2 standard deviations.")

    with benchmark_tab:
        st.subheader("Benchmark Model Comparison")
        st.markdown("Compare LSTM performance against traditional time-series (ARIMA) and machine learning (Random Forest) models.")
        
        with st.spinner("Running ARIMA and Random Forest benchmarks..."):
            benchmark_results = stock_predictor.compare_benchmarks(df, indicators_df, lstm_result=lstm_benchmark_result)
        
        # Styled metrics table
        st.markdown("#### 📈 Performance Metrics Comparison")
        metrics_data = []
        for name, result in benchmark_results.items():
            metrics = result['metrics']
            metrics_data.append({
                'Model': name,
                'RMSE': round(metrics['RMSE'], 4),
                'MAE': round(metrics['MAE'], 4),
                'MAPE (%)': round(metrics['MAPE'], 2),
                'R² Score': round(metrics['R2'], 4)
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Highlight best values
        def highlight_best(s):
            if s.name in ['RMSE', 'MAE', 'MAPE (%)']:
                is_best = s == s.min()
            elif s.name == 'R² Score':
                is_best = s == s.max()
            else:
                return ['' for _ in s]
            return ['background-color: #90EE90' if v else '' for v in is_best]
        
        styled_df = metrics_df.style.apply(highlight_best, axis=0)
        st.dataframe(styled_df, width='stretch', hide_index=True)
        
        # Find best model
        best_r2_model = metrics_df.loc[metrics_df['R² Score'].idxmax(), 'Model']
        best_rmse_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
        st.success(f"🏆 **Best Model by R² Score:** {best_r2_model} | **Best by RMSE:** {best_rmse_model}")
        
        st.markdown("#### 📉 Prediction Comparison Charts")
        benchmark_fig = stock_predictor.plot_benchmark_predictions(benchmark_results)
        st.pyplot(benchmark_fig)

st.set_page_config(layout="wide")

st.sidebar.markdown("<h1 style='text-align: center;'>Stock Market Price Prediction</h1>", unsafe_allow_html=True)

stocks = get_stock_list()

selected_stock = None

for stock in stocks:
    if st.sidebar.button(stock, key=f"btn_{stock}", width="stretch"):
        selected_stock = stock

if selected_stock is None:
    selected_stock = stocks[0]


# Main content
st.title(f"{selected_stock} Stock Analysis")
execute_prediction(selected_stock)
