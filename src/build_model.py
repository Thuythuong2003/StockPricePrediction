from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
import streamlit as st
import torch
from NODE import ODEfunc, NeuralODE 
import os
import pandas as pd
from glob import glob
from pmdarima import auto_arima
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import r2_score
from mealpy import FloatVar, ARO
import torch
import torch.nn as nn
from process_data import *
import torch
import torch.nn as nn
from torchdiffeq import odeint

__all__ = ['build_lstm_model', 'build_gru_model', 'build_node_model', 'build_arima_model', 'objective_function', 'train_NeuralODE']

folder_path = '../data'
import importlib
glob = importlib.import_module('glob')
# Lấy danh sách tất cả các file .csv
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Duyệt và đọc từng file
for file in csv_files:
    print(f"Đang đọc file: {file}")
    data = pd.read_csv(file)

x_train, y_train, x_test, y_test, scaler, test_scaled, data = process_data(data)

def build_lstm_model(lstm_unit1,lstm_unit2 ,dropout_rate, dense_unit):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=int(lstm_unit1),return_sequences=True, input_shape=(x_train.shape[1],1)))
    lstm_model.add(Dropout(dropout_rate))
    lstm_model.add(LSTM(units=int(lstm_unit2)))
    lstm_model.add(Dropout(dropout_rate))
    lstm_model.add(Dense(int(dense_unit), activation='relu'))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    return lstm_model

def objective_function(solution):

    lstm_unit1 = int(round(solution[0]))
    lstm_unit2 = int(round(solution[1]))
    dropout_rate = round(solution[2],1)
    dense_unit = int(round(solution[3]))

    print(solution)

    lstm_model = build_lstm_model(lstm_unit1,lstm_unit2,dropout_rate,dense_unit)

    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 10)

    lstm_model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data = (x_test, y_test), verbose=1, callbacks=[early_stop])

    predict = lstm_model.predict(x_test)
    r2 = r2_score(y_test,predict)
    return -r2

def aro(objective_function):

    problem_dict = {
        "bounds": FloatVar(lb=[64,64,0.2,10], ub=[256,256,0.7,25], name="hyperparameters"),
        "obj_func": objective_function,
        "minmax": "min",
    }

    model = ARO.OriginalARO(epoch=1,pop_size=5)
    gbest = model.solve(problem_dict)
    return gbest

def train_NeuralODE(model, x_train, y_train,x_test, y_test ,epochs=100, lr=0.01, patience=10):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    best_val_loss = float('inf')
    best_model = None
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()     #Xóa gradients từ bước trước
        prediction = model(x_train).squeeze()
        loss = criterion(prediction, y_train)
        loss.backward()
        optimizer.step()    #Cập nhật tham số mô hình

        model.eval()
        with torch.no_grad():
            val_prediction = model(x_test).squeeze()
            val_loss = criterion(val_prediction, y_test)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model = model.state_dict()
            patience_count = 0
        else:
            patience_count +=1

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Training loss: {loss.item()}, Val loss: {val_loss.item()}')

        if patience_count > patience:
            print(f'Early stopping at epoch {epoch}')
            break

    if model is not None:
        model.load_state_dict(best_model)

    return model

def build_gru_model(units, dropout):
    model = Sequential()
    model.add(GRU (units = units, return_sequences = True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(dropout))
    model.add(GRU(units = units))
    model.add(Dropout(dropout))
    model.add(Dense(units = 1))
    model.compile(optimizer='adam',loss='mse')
    return model

def build_node_model(input_size):
    odefunc = ODEfunc(input_size)
    model = NeuralODE(odefunc, num_feature=input_size)
    return model

def build_arima_model(train_series):
    
    model = auto_arima(train_series,
                       start_p=0, start_q=0,
                       test='adf',               # Kiểm tra tính dừng bằng Augmented Dickey-Fuller
                       max_p=5, max_q=5,
                       d=None,                  # Để auto_arima tự chọn bậc d
                       seasonal=False,          # Không dùng yếu tố mùa vụ
                       start_P=0,
                       D=0,
                       trace=True,              # Hiển thị quá trình lựa chọn mô hình
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    return model
