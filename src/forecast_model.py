
import pandas as pd
import numpy as np

import torch
import torch
import torch.nn as nn

__all__ = ['forecast_node', 'forecast_keras', 'forecast_arima', 'compute_optimal_alpha', 'forecast_arima_gru']

# === Hàm dự báo NODE ===
def forecast_node(model, scaler, test_scaled):
    #Chuẩn bị dữ liệu để dự đoán 30 ngày tiếp theo
    last_day = 60
    predicted_values = []
    last_n_day = test_scaled[-last_day:]

    # Tạo dự đoán 10 ngày tiếp theo
    for _ in range(10):
        #Dự đoán giá trị tiếp theo
        n = torch.from_numpy(last_n_day).float().unsqueeze(0)
        pred = model(n).detach().numpy()
        pred = pred[:,-1]
        predicted_values.append(pred[0])

        #Cập nhật last_n_day cho lần dự đoán tiếp theo, thêm giá trị dự đoán mới và loại bỏ giá trị cũ đi
        last_n_day = np.append(last_n_day[1:],[[pred[0]]], axis=0)

    predicted_values = np.array(predicted_values)
    predicted_values = predicted_values.reshape(-1,1)
    predicted_values = scaler.inverse_transform(predicted_values)

    return predicted_values

# === Hàm dự báo (Keras model) ===
def forecast_keras(model, scaler, test_scaled):
    last_day = 60
    predicted_values = []
    last_n_day = test_scaled[-last_day:]

    #Tạo dự đoán 10 ngày tiếp theo
    for _ in range(10):
        #Dự đoán giá trị tiếp theo
        pred = model.predict(last_n_day.reshape((1,last_n_day.shape[0],1)))
        predicted_values.append(pred[0,0])

        #Cập nhật last_n_day cho lần dự đoán tiếp theo, thêm giá trị dự đoán mới và loại bỏ giá trị cũ đi
        last_n_day = np.append(last_n_day[1:],[[pred[0,0]]], axis=0)

    predicted_values = np.array(predicted_values)
    predicted_values = predicted_values.reshape(-1,1)
    predicted_values = scaler.inverse_transform(predicted_values)
    return predicted_values

def forecast_arima(model_fit, steps=10):
    forecast = model_fit.forecast(steps=steps)
    return np.array(forecast)

# Tính alpha 
def compute_optimal_alpha(y_true, y_arima, y_gru):
    numerator = np.sum((y_arima - y_gru) * (y_true - y_gru))
    denominator = np.sum((y_arima - y_gru) ** 2)
    alpha = numerator / denominator if denominator != 0 else 0
    return max(0, min(1, alpha))

def forecast_arima_gru(gru_model, arima_model, x_test, y_test, test_scaled, scaler, forecast_len=10):
    # Dự đoán với ARIMA và GRU
    arima_forecast_test = arima_model.predict(n_periods=len(y_test))
    gru_forecast_test = gru_model.predict(x_test).flatten()

    # Tính alpha tối ưu
    min_len = min(len(y_test), len(gru_forecast_test), len(arima_forecast_test))
    y_true = y_test[:min_len]
    gru_pred = gru_forecast_test[:min_len]
    arima_pred = arima_forecast_test[:min_len]
    alpha = compute_optimal_alpha(y_true, arima_pred, gru_pred)

    # Dự báo tương lai với GRU
    last_60 = test_scaled[-60:]
    gru_input = last_60.reshape(1, 60, 1)
    gru_forecast_10 = []
    input_seq = gru_input.copy()
    for _ in range(forecast_len):
        next_pred = gru_model.predict(input_seq, verbose=0)[0][0]
        gru_forecast_10.append(next_pred)
        input_seq = np.append(input_seq[0, 1:, 0], next_pred).reshape(1, 60, 1)

    # Dự báo tương lai với ARIMA
    arima_forecast_10 = arima_model.predict(n_periods=forecast_len)

    # Kết hợp dự báo theo alpha
    combined_pred_10 = alpha * np.array(arima_forecast_10) + (1 - alpha) * np.array(gru_forecast_10)
    predicted_values = scaler.inverse_transform(combined_pred_10.reshape(-1, 1))

    return predicted_values
