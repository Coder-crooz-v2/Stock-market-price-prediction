import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from statsmodels.tsa.arima.model import ARIMA

class StockData:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.past_100_days = None

    def import_data(self, stock_name):
        start = '2010-01-01'
        end = '2019-12-31'
        df = yf.download(stock_name, start=start, end=end)
        df = df.reset_index()
        df = df.drop(columns=['Date'], axis=1)
        return df

    def plot_data(self, df, show_ma_100=False, show_ma_200=False):
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()
        fig = plt.figure(figsize=(12,6))
        plt.plot(df.Close, label='AAPL Close Price')
        if show_ma_100:
            plt.plot(ma100, label='100-Day Moving Average', color='red')
        if show_ma_200:
            plt.plot(ma200, label='200-Day Moving Average', color='green')
        plt.title('AAPL Stock Price and Moving Averages')
        plt.legend()
        return fig

    def prepare_training_data(self, df):
        train_data = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        self.past_100_days = train_data.tail(100)
        train_data_array = self.scaler.fit_transform(train_data)

        x_train = []
        y_train = []

        for i in range(100, train_data_array.shape[0]):
            x_train.append(train_data_array[i-100:i])
            y_train.append(train_data_array[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        return x_train, y_train

    def prepare_model(self, x_train):
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))
        return model
    
    def train_model(self, model, x_train, y_train):
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=50)
        return model
    
    def predict(self, df, model):
        test_data = pd.DataFrame(df['Close'][int(len(df)*0.70):])
        final_df = pd.concat([self.past_100_days, test_data], ignore_index=True)
        input_data = self.scaler.fit_transform(final_df)
        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)
        return y_test, y_predicted

    def get_metrics(self, y_test, y_predicted):
        # Inverse transform if needed
        if len(y_predicted.shape) == 2 and y_predicted.shape[1] == 1:
            y_pred = y_predicted.flatten()
        else:
            y_pred = y_predicted
        if len(y_test.shape) == 2 and y_test.shape[1] == 1:
            y_true = y_test.flatten()
        else:
            y_true = y_test

        rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }

    def plot_predictions(self, y_predicted, y_test):
        y_predicted = self.scaler.inverse_transform(y_predicted)
        y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        fig = plt.figure(figsize=(12,6))
        plt.plot(y_test, color='blue', label='Real Stock Price')
        plt.plot(y_predicted, color='red', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        return fig

    # --- Technical Indicators & Benchmarking Utilities --- #

    def compute_technical_indicators(self, df):
        """Return a copy of df enriched with RSI, MACD and Bollinger Bands."""
        indicators = df.copy()

        close = indicators['Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        indicators['RSI'] = 100 - (100 / (1 + rs))
        indicators['RSI'] = indicators['RSI'].replace([np.inf, -np.inf], np.nan)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        indicators['MACD'] = ema12 - ema26
        indicators['MACD_SIGNAL'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        indicators['MACD_HIST'] = indicators['MACD'] - indicators['MACD_SIGNAL']

        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        indicators['BB_MIDDLE'] = rolling_mean
        indicators['BB_UPPER'] = rolling_mean + (rolling_std * 2)
        indicators['BB_LOWER'] = rolling_mean - (rolling_std * 2)

        indicators = indicators.fillna(method='bfill').fillna(method='ffill')
        return indicators

    def plot_indicator_overview(self, indicators_df):
        """Plot Close with Bollinger Bands, RSI, and MACD."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        axes[0].plot(indicators_df['Close'], label='Close', color='black', linewidth=1.5)
        axes[0].plot(indicators_df['BB_UPPER'], label='Upper Band', linestyle='--', color='red', alpha=0.7)
        axes[0].plot(indicators_df['BB_MIDDLE'], label='Middle Band', linestyle='--', color='grey')
        axes[0].plot(indicators_df['BB_LOWER'], label='Lower Band', linestyle='--', color='green', alpha=0.7)
        axes[0].fill_between(indicators_df.index, indicators_df['BB_LOWER'], indicators_df['BB_UPPER'], alpha=0.1)
        axes[0].set_title('Bollinger Bands (20, 2σ)')
        axes[0].legend(loc='upper left')

        axes[1].plot(indicators_df['RSI'], color='purple', linewidth=1.5)
        axes[1].axhline(70, color='red', linestyle='--', alpha=0.6)
        axes[1].axhline(30, color='green', linestyle='--', alpha=0.6)
        axes[1].set_title('Relative Strength Index (14)')
        axes[1].set_ylim(0, 100)

        axes[2].plot(indicators_df['MACD'], label='MACD', color='blue')
        axes[2].plot(indicators_df['MACD_SIGNAL'], label='Signal', color='orange')
        axes[2].bar(indicators_df.index, indicators_df['MACD_HIST'], label='Histogram', color='gray', alpha=0.5)
        axes[2].set_title('MACD (12,26,9)')
        axes[2].legend(loc='upper left')

        plt.tight_layout()
        return fig

    def summarize_latest_indicators(self, indicators_df):
        latest = indicators_df.iloc[-1]
        return {
            'RSI': float(latest['RSI']),
            'MACD': float(latest['MACD']),
            'MACD_SIGNAL': float(latest['MACD_SIGNAL']),
            'MACD_HIST': float(latest['MACD_HIST']),
            'BB_UPPER': float(latest['BB_UPPER']),
            'BB_MIDDLE': float(latest['BB_MIDDLE']),
            'BB_LOWER': float(latest['BB_LOWER'])
        }

    def _benchmark_metrics(self, actual, predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        actual_safe = np.where(actual == 0, np.nan, actual)
        mape = np.nanmean(np.abs((actual - predicted) / actual_safe)) * 100
        r2 = r2_score(actual, predicted)
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

    def run_arima_benchmark(self, df):
        close = df['Close'].values
        split_idx = int(len(close) * 0.70)
        train_series = list(close[:split_idx])
        test_series = close[split_idx:]
        preds = []
        try:
            for t in range(len(test_series)):
                model = ARIMA(train_series, order=(5, 1, 0))
                fitted = model.fit()
                forecast = fitted.forecast(steps=1)
                pred_val = float(forecast.iloc[0]) if hasattr(forecast, 'iloc') else float(forecast[0])
                preds.append(pred_val)
                train_series.append(test_series[t])
            preds = np.array(preds)
        except Exception as exc:
            preds = np.repeat(train_series[-1], len(test_series))
        metrics = self._benchmark_metrics(test_series, preds)
        return {'model': 'ARIMA(5,1,0)', 'actual': test_series, 'predictions': preds, 'metrics': metrics}

    def run_random_forest_benchmark(self, indicators_df):
        feature_cols = ['RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']
        dataset = indicators_df[feature_cols + ['Close']].dropna()
        if dataset.empty:
            raise ValueError('Insufficient data to compute technical indicators for benchmarking.')
        split_idx = int(len(dataset) * 0.70)
        X_train, X_test = dataset[feature_cols][:split_idx], dataset[feature_cols][split_idx:]
        y_train, y_test = dataset['Close'][:split_idx], dataset['Close'][split_idx:]
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = self._benchmark_metrics(y_test.values, preds)
        return {'model': 'Random Forest', 'actual': y_test.values, 'predictions': preds, 'metrics': metrics}

    def get_lstm_benchmark_result(self, y_test, y_predicted):
        """Convert LSTM predictions to benchmark result format with actual prices."""
        y_pred_inv = self.scaler.inverse_transform(y_predicted).flatten()
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        metrics = self._benchmark_metrics(y_test_inv, y_pred_inv)
        return {'model': 'LSTM', 'actual': y_test_inv, 'predictions': y_pred_inv, 'metrics': metrics}

    def compare_benchmarks(self, df, indicators_df=None, lstm_result=None):
        if indicators_df is None:
            indicators_df = self.compute_technical_indicators(df)
        results = {}
        if lstm_result is not None:
            results['LSTM'] = lstm_result
        arima = self.run_arima_benchmark(df)
        rf = self.run_random_forest_benchmark(indicators_df)
        results['ARIMA'] = arima
        results['Random Forest'] = rf
        return results

    def plot_benchmark_predictions(self, benchmark_results):
        rows = len(benchmark_results)
        fig, axes = plt.subplots(rows, 1, figsize=(12, 4 * rows))
        if rows == 1:
            axes = [axes]
        for ax, (name, result) in zip(axes, benchmark_results.items()):
            ax.plot(result['actual'], label='Actual Price', color='black', linewidth=1.5)
            ax.plot(result['predictions'], label=f'{name} Prediction', linestyle='--')
            ax.set_title(f'{name} vs Actual')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig