# Stock Market Price Prediction

A Python web application that predicts the prices of stocks of popular public limited companies using LSTM neural networks. The application provides an interactive interface built with Streamlit to visualize stock data, moving averages, and predictions. You can visit this [link](https://stock-market-lstm.streamlit.app) and view the output.

## 🚀 Features

- **Interactive Stock Selection**: Choose from 20 popular stocks including AAPL, MSFT, AMZN, GOOGL, META, TSLA, and more
- **Real-time Data Fetching**: Automatically downloads historical stock data using Yahoo Finance API
- **Moving Averages Visualization**: Toggle 100-day and 200-day moving averages on/off
- **LSTM Neural Network Prediction**: Uses deep learning to predict future stock prices
- **Interactive Charts**: View both historical data and prediction comparisons
- **Automatic Model Training**: Trains and saves the model automatically if not present
- **Responsive Web Interface**: Clean, user-friendly Streamlit interface

## 📊 Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: TensorFlow/Keras with LSTM layers
- **Data Processing**: Pandas, NumPy
- **Data Visualization**: Matplotlib
- **Data Source**: Yahoo Finance (yfinance)
- **Preprocessing**: Scikit-learn MinMaxScaler

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Coder-crooz-v2/Stock-market-price-prediction.git
   cd Stock-market-price-prediction
   ```

2. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   The application will automatically open in your default browser at `http://localhost:8501`

## 📈 How It Works

### Data Pipeline

1. **Data Collection**: Historical stock data (2010-2019) is fetched from Yahoo Finance
2. **Data Preprocessing**: Stock prices are normalized using MinMaxScaler
3. **Feature Engineering**: Creates sequences of 100 days for LSTM input
4. **Model Training**: LSTM model with 4 layers and dropout for regularization
5. **Prediction**: Uses the trained model to predict future stock prices

### Model Architecture

- **Input Layer**: 100 time steps with 1 feature (closing price)
- **LSTM Layer 1**: 50 units with ReLU activation
- **LSTM Layer 2**: 60 units with ReLU activation
- **LSTM Layer 3**: 80 units with ReLU activation
- **LSTM Layer 4**: 120 units with ReLU activation
- **Dense Output**: Single unit for price prediction
- **Regularization**: Dropout layers (0.2, 0.3, 0.4, 0.5) to prevent overfitting

### Training Details

- **Training Data**: 70% of historical data (2010-2017)
- **Test Data**: 30% of historical data (2017-2019)
- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam
- **Epochs**: 50

## 🎯 Usage

1. **Select a Stock**: Click on any stock symbol in the sidebar to analyze
2. **View Historical Data**: See the stock price history with optional moving averages
3. **Toggle Moving Averages**: Use the buttons to show/hide 100-day and 200-day moving averages
4. **View Predictions**: Scroll down to see the model's predictions vs actual prices
5. **Model Training**: If running for the first time, the model will train automatically

## 📂 Project Structure

```
Stock-market-price-prediction/
│
├── app.py                 # Main Streamlit application
├── model.py              # StockData class with ML pipeline
├── model.keras           # Trained LSTM model (generated after first run)
├── lstm-model.ipynb      # Jupyter notebook for model development
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── LICENSE              # Project license
```

## 🎮 Available Stocks

The application supports prediction for the following stocks:

**Technology**: AAPL, MSFT, GOOGL, META, NVDA, INTC, AMD, CSCO, ADBE, ORCL, IBM
**E-commerce & Services**: AMZN, PYPL, UBER, ZM, SPOT, SHOP, SQ
**Entertainment**: NFLX
**Automotive**: TSLA

## ⚠️ Limitations

- **Historical Data Only**: Predictions are based on 2010-2019 data
- **Single Feature**: Only uses closing prices (doesn't include volume, news, etc.)
- **Not Financial Advice**: This is for educational purposes only
- **Model Accuracy**: LSTM predictions may not reflect real market conditions

## 🔮 Future Enhancements

- [ ] Add more technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Include volume and sentiment analysis
- [ ] Real-time data updates
- [ ] Multiple timeframe predictions
- [ ] Portfolio analysis features
- [ ] Model performance metrics display

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for providing free stock data API
- Streamlit team for the amazing web framework
- TensorFlow/Keras for deep learning capabilities

## ⚡ Quick Start

```bash
# Clone and setup
git clone https://github.com/Coder-crooz-v2/Stock-market-price-prediction.git
cd Stock-market-price-prediction
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

**Disclaimer**: This application is for educational and research purposes only. Stock market predictions are inherently uncertain and this tool should not be used as the sole basis for investment decisions. Always consult with financial professionals before making investment choices.
