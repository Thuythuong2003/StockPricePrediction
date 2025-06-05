import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch



def evaluate_node(node_model, x_test, y_test, scaler):
    test_tensor = torch.from_numpy(x_test).float()
    predict = node_model(test_tensor).detach().numpy()
    predict = predict.squeeze()

    rmse = np.sqrt(mean_squared_error(y_test,predict))
    mae = mean_absolute_error(y_test, predict)

    predict_test_price = scaler.inverse_transform(predict.reshape(-1, 1))
    actual_test_price = scaler.inverse_transform(y_test.reshape(-1, 1))

    return rmse, mae, predict_test_price, actual_test_price

def evaluate_lstm(lstm_model, x_test, y_test, scaler):
    predict = lstm_model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test,predict))
    mae = mean_absolute_error(y_test, predict)

    predict_test_price = scaler.inverse_transform(predict)
    actual_test_price = scaler.inverse_transform(y_test.reshape(-1, 1))

    return rmse, mae, predict_test_price, actual_test_price

def compute_optimal_alpha(y_true, y_arima, y_gru):
    numerator = np.sum((y_arima - y_gru) * (y_true - y_gru))
    denominator = np.sum((y_arima - y_gru) ** 2)
    alpha = numerator / denominator if denominator != 0 else 0
    return max(0, min(1, alpha))

def evaluate_arimagru(model_gru, arima_model, x_test, y_test,test ,scaler):

    gru_predict = model_gru.predict(x_test)

    arima_predict = arima_model.predict(n_periods=len(y_test))

    # Giữ dữ liệu ARIMA ở dạng chuẩn hóa (không exp)
    arima_pred_scaled = arima_predict   
    arima_true_scaled = test       

    # GRU  dữ liệu scaled, chưa inverse
    gru_pred_scaled = gru_predict.reshape(-1)
    gru_true_scaled = y_test.reshape(-1)

    # Cắt độ dài cho khớp
    min_len = min(len(arima_true_scaled), len(gru_true_scaled))
    arima_pred_scaled = arima_pred_scaled[:min_len]
    gru_pred_scaled = gru_pred_scaled[:min_len]
    y_true_scaled = gru_true_scaled[:min_len]

    alpha = compute_optimal_alpha(y_true_scaled, arima_pred_scaled, gru_pred_scaled)
    combined_pred_scaled = alpha * arima_pred_scaled + (1 - alpha) * gru_pred_scaled

    rmse = np.sqrt(mean_squared_error(y_true_scaled, combined_pred_scaled))
    mae = mean_absolute_error(y_true_scaled, combined_pred_scaled)

    predict_test_price = scaler.inverse_transform(combined_pred_scaled.values.reshape(-1, 1))
    
    actual_test_price = scaler.inverse_transform(y_true_scaled.reshape(-1, 1))
    
    return rmse, mae, predict_test_price, actual_test_price
