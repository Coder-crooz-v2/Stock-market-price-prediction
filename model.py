import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

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