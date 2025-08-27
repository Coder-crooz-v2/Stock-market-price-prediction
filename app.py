import streamlit as st
import os
from model import StockData
from keras.models import load_model

def get_stock_list():
    stocks = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "PYPL", "NFLX", "INTC",
        "AMD", "CSCO", "ADBE", "ORCL", "IBM", "UBER", "ZM", "SPOT", "SHOP", "SQ"
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
    fig = stock_predictor.plot_predictions(predictions, test_data)
    st.subheader("Actual vs Predicted Prices")
    st.pyplot(fig)

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
