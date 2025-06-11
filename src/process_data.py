import streamlit as st

import pandas as pd
import numpy as np
import ta
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

__all__ = ['fetch_stock_data_from_file', 'process_data_arima', 'process_data', 'calculate_metrics', 'add_technical_indicators', 'load_node_model']


def fetch_stock_data_from_file(file):
    if file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Không thể đọc file: {e}")
        return pd.DataFrame()
    
    try:
        df['Datetime'] = pd.to_datetime(df['time'], format='%Y-%m-%d', errors='raise')
    except:
        try:
            df['Datetime'] = pd.to_datetime(df['time'], format='%d/%m/%Y', errors='raise')
        except:
            df['Datetime'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df = df.sort_values(by='Datetime').reset_index(drop=True)
    return df

def process_data_arima(data):
    #định dạng lại cấu trúc thời gian
    data['time']=pd.to_datetime(data['time'], format='mixed' ,dayfirst=False)

    # Lấy chuỗi thời gian giá 'close'
    data = np.log(data['close'])

    # Chia tập dữ liệu thành train/test (80% train, 20% test)
    train, test = train_test_split(data, test_size=0.2, shuffle=False)

    return train, test

def process_data(data, train_ratio=0.8):
    # Chuyển đổi định dạng thời gian và chọn cột 'close'
    data['time'] = pd.to_datetime(data['time'], format='mixed', dayfirst=False)
    data = data[['close']]

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Chia dữ liệu theo tỷ lệ train_ratio
    train_size = int(len(scaled_data) * train_ratio)
    train_scaled = scaled_data[:train_size]
    test_scaled = scaled_data[train_size:]

    # Tạo dữ liệu huấn luyện
    x_train, y_train = [], []
    for i in range(60, len(train_scaled)):
        x_train.append(train_scaled[i - 60:i, 0])
        y_train.append(train_scaled[i, 0])

    # Tạo dữ liệu kiểm thử
    x_test, y_test = [], []
    for i in range(60, len(test_scaled)):
        x_test.append(test_scaled[i - 60:i, 0])
        y_test.append(test_scaled[i, 0])

    # Chuyển thành mảng numpy
    x_train = np.array(x_train).reshape(-1, 60, 1)
    y_train = np.array(y_train)
    x_test = np.array(x_test).reshape(-1, 60, 1)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test, scaler, test_scaled, data

def load_node_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'),weights_only=False)
    model.eval()
    return model

def calculate_metrics(data):
    last_close = data['close'].iloc[-1]
    prev_close = data['close'].iloc[-2]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['high'].iloc[-1]
    low = data['low'].iloc[-1]
    volume = data['volume'].iloc[-1]
    return last_close, change, pct_change, high, low, volume

def add_technical_indicators(data):
    data['SMA_20'] = ta.trend.sma_indicator(data['close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['close'], window=20)
    # MACD
    macd = ta.trend.macd(data['close'])
    data['MACD'] = macd
    # RSI
    data['RSI'] = ta.momentum.rsi(data['close'], window=14)

    return data
