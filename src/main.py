import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import  pickle
from mealpy import FloatVar, ARO
import tensorflow as tf
from tensorflow.keras.models import load_model

import plotly.express as px
from NODE import ODEfunc, NeuralODE 

from forecast_model import *
from process_data import *
from build_model import *
from evaluate_model import *

# ================================
# Streamlit UI
# ================================

# Sidebar
st.sidebar.title("Settings")

uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu cổ phiếu", type=["csv"])

if uploaded_file is not None:
    # Lấy tên file, ví dụ: FPT.csv → ticker = "FPT"
    ticker = os.path.splitext(uploaded_file.name)[0].upper()  # Chuyển về in hoa nếu cần

chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])

st.sidebar.markdown("### Cập nhật dữ liệu")
update_data_btn = st.sidebar.button('Update')  # Nút để cập nhật dữ liệu

st.sidebar.markdown("---")

# Tabs
st.title("Stock Prediction Dashboard")
tab1, tab2 = st.tabs(["📊 Preview Data", "📈 Prediction"])

with tab1:
    st.header("Data Preview")
    
    if update_data_btn:
        data = fetch_stock_data_from_file(uploaded_file)
        
        if data.empty:
            st.warning("Không tìm thấy dữ liệu hoặc dữ liệu không hợp lệ!")
            st.session_state.data = None
        else:
            data = add_technical_indicators(data)
            st.session_state.data = data  # lưu vào session
    else:
        data = st.session_state.get('data', None)

    if data is not None and not data.empty:
        # Ví dụ hiển thị chỉ số, biểu đồ giống code cũ nhưng bỏ ticker
        last_close, change, pct_change, high, low, volume = calculate_metrics(data)
        st.metric(label="Last Price", value=f"{last_close:.2f} ", delta=f"{change:.2f} ({pct_change:.2f}%)")

        col1, col2, col3 = st.columns(3)
        col1.metric("High", f"{high:.2f}")
        col2.metric("Low", f"{low:.2f} ")
        col3.metric("Volume", f"{volume:,}")

        fig = go.Figure()
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(x=data['Datetime'],
                                         open=data['open'],
                                         high=data['high'],
                                         low=data['low'],
                                         close=data['close']))
        else:
            fig = px.line(data, x='Datetime', y='close')

        # Thêm indicator nếu có
        if 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'))
        if 'EMA_20' in data.columns:
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))
        if 'MACD' in data.columns:
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['MACD'], name='MACD'))
        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['RSI'], name='RSI'))

        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Historical Data')
        st.dataframe(data[['Datetime', 'open', 'high', 'low', 'close', 'volume']], use_container_width=True)

        st.subheader('Technical Indicators')
        st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20', 'MACD', 'RSI']], use_container_width=True)
    else:
        st.info("Vui lòng tải file dữ liệu và nhấn 'Update' để load dữ liệu.")

    # Đảm bảo session_state được khởi tạo từ đầu
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'predicted_values' not in st.session_state:
        st.session_state['predicted_values'] = None

    # --- Giao diện chọn mô hình và nút Prediction ---
    st.header("📈 Dự đoán giá cổ phiếu")
    st.markdown("### Chọn mô hình")
    model = st.selectbox("Chọn mô hình", ["NODE", "ARIMA-GRU", "LSTM", "LSTM-ARO", "GRU"])
    forecast_btn = st.button("Predict")

    # --- Xử lý khi bấm Forecast ---
    if forecast_btn:
        data = st.session_state.get('data', None)
        if data is None or data.empty:
            st.warning("Không có dữ liệu để dự báo. Vui lòng cập nhật dữ liệu trước.")
            st.stop()

        data = data.copy()

        with st.spinner("🔄 Đang xử lý dữ liệu..."):
            x_train, y_train, x_test, y_test, scaler, test_scaled, processed_data = process_data(data)
            train, test = process_data_arima(data)

        st.subheader("🔮 Kết quả dự đoán xu hướng giá đóng cửa 10 ngày tiếp theo")

        try:
            if model == 'NODE':
                model_path = f'../model/NODE-{ticker}.pth'
                node_model = load_node_model(model_path)
                st.session_state.predicted_values = forecast_node(node_model, scaler, test_scaled)

            elif model == 'ARIMA-GRU':
                gru_model = load_model(f'../model/GRU-{ticker}.h5', custom_objects={'mse': 'mean_squared_error'})
                with open(f'../model/ARIMA-{ticker}.pkl', 'rb') as f:
                    arima_model = pickle.load(f)

                predicted_values = forecast_arima_gru(
                    gru_model=gru_model,
                    arima_model=arima_model,
                    x_test=x_test,
                    y_test=y_test,
                    test_scaled=test_scaled,
                    scaler=scaler,
                    forecast_len=10
                )
            
            else:  # Các mô hình Keras khác
                keras_model_path = f'../model/{model}-{ticker}.h5'
                keras_model = load_model(keras_model_path, custom_objects={'mse': 'mean_squared_error'})
                st.session_state.predicted_values = forecast_keras(keras_model, scaler, test_scaled)

        except Exception as e:
            st.error(f"❌ Lỗi khi chạy mô hình {model}: {e}")
            st.stop()

        # Hiển thị kết quả dự báo nếu thành công
        if st.session_state.predicted_values is not None:
            forecast_dates = pd.date_range(start=data['Datetime'].iloc[-1] + timedelta(days=1), periods=10, freq=BDay())
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predict_Close': st.session_state.predicted_values.flatten()
            })

            fig = px.line(forecast_df,  y='Predict_Close', title="📈 Dự đoán giá đóng cửa")
            fig.update_traces(line=dict(color='red'))
            fig.update_layout(xaxis_title='Ngày', yaxis_title='Giá dự đoán')
            st.plotly_chart(fig, use_container_width=True)

            next_day_price = forecast_df['Predict_Close'].iloc[0]
            st.metric("📌 Giá ngày tiếp theo", f"{next_day_price:.2f}")

with tab2:
    st.header("📈 Dự đoán giá cổ phiếu")

    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Vui lòng cập nhật dữ liệu ở tab Preview Data trước.")
        st.stop()

    data = st.session_state.data.copy()
    predicted = None
    model = st.selectbox("🔍 Chọn mô hình", ['NODE', 'ARIMA-GRU', 'LSTM-ARO'])
    
    # Bước 1: Hiển thị các tham số phù hợp theo model đã chọn
    if model == 'NODE':
        st.subheader("⚙️ Tham số cho NODE")
        epochs_node = st.slider("Epochs", 10, 100, step=1)
        train_ratio = st.slider("📊 Training Rate", 0.5, 0.9, value=0.8, step=0.1)

    elif model == 'ARIMA-GRU':
        st.subheader("⚙️ Tham số cho ARIMA-GRU")
        units = st.slider("Units", 64, 128, step=64)
        epochs = st.slider("Epochs", 10, 100, step=1, value=50)
        dropout = st.slider("Dropout", 0.0, 0.9, step=0.05, value=0.2)
        batch_size = st.slider("Batch Size", 32, 128, step=8, value=32)
        train_ratio = st.slider("📊 Training Rate", 0.5, 0.9, value=0.8, step=0.1)

    elif model == 'LSTM-ARO':
        st.subheader("⚙️ Tham số huấn luyện lại sau tối ưu")
        train_epochs = st.slider("Epochs", 10, 100, step=1, value=50)
        batch_size = st.slider("Batch Size", 16, 128, step=8, value=32)
        train_ratio = st.slider("📊 Training Rate", 0.5, 0.9, value=0.8, step=0.1)

    # Bước 2: Nút dự đoán
    if st.button("🔮 Dự đoán"):
        if model == 'NODE':
            x_train, y_train, x_test, y_test, scaler, test_scaled, _ = process_data(data, train_ratio)
            node_model = NeuralODE(ODEfunc(1))
            trained_model = train_NeuralODE(
                node_model, x_train, y_train, x_test, y_test,
                epochs=epochs_node, lr=0.01, patience=10
            )
            rmse, mae, predict_test_price, actual_test_price = evaluate_node(trained_model, x_test, y_test, scaler)
            predicted = forecast_node(trained_model, scaler, test_scaled)

        elif model == 'ARIMA-GRU':
            train, test = process_data_arima(data)
            x_train, y_train, x_test, y_test, scaler, test_scaled, _ = process_data(data)

            arima_model = build_arima_model(train)
            gru_model = build_gru_model(units=units, dropout=dropout)
            gru_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

            rmse, mae, predict_test_price, actual_test_price = evaluate_arimagru(gru_model,arima_model, x_test, y_test,test ,scaler)

            predicted = forecast_arima_gru(
                gru_model=gru_model,
                arima_model=arima_model,
                x_test=x_test,
                y_test=y_test,
                test_scaled=test_scaled,
                scaler=scaler,
                forecast_len=10
            )

        elif model == 'LSTM-ARO':
            x_train, y_train, x_test, y_test, scaler, test_scaled, data = process_data(data)

            with st.spinner("🔍 Đang tối ưu tham số bằng ARO..."):
                problem = {
                    "bounds": FloatVar(lb=[64, 64, 0.2, 10], ub=[256, 256, 0.7, 25], name="hyperparameters"),
                    "obj_func": objective_function,
                    "minmax": "min"
                }

                model_aro = ARO.OriginalARO(epoch=1, pop_size=5)
                gbest = model_aro.solve(problem)
                best_params = gbest.solution
                lstm_unit1, lstm_unit2 = map(int, best_params[:2])
                dropout_rate = round(best_params[2], 1)
                dense_unit = int(best_params[3])

            st.success(f"Tối ưu thành công: LSTM1={lstm_unit1}, LSTM2={lstm_unit2}, Dropout={dropout_rate}, Dense={dense_unit}")

            best_model = build_lstm_model(lstm_unit1, lstm_unit2, dropout_rate, dense_unit)
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            best_model.fit(
                x_train, y_train,
                epochs=train_epochs, batch_size=batch_size,
                validation_data=(x_test, y_test),
                verbose=0, callbacks=[early_stop]
            )
            rmse, mae, predict_test_price, actual_test_price = evaluate_lstm(best_model, x_test, y_test, scaler)
            predicted = forecast_keras(best_model, scaler, test_scaled)

    if predicted is not None:
        forecast_dates = pd.date_range(start=data['Datetime'].iloc[-1] + timedelta(days=1), periods=10, freq=BDay())
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predict_Close': predicted.flatten()
        })

        compare_df = pd.DataFrame({
            'Actual Prices': actual_test_price.flatten(),
            'Predicted Prices': predict_test_price.flatten()
        }, index=range(len(y_test)))
        compare_df.index.name = "Ngày"
        compare_df_long = compare_df.reset_index().melt(id_vars='Ngày',value_vars=['Actual Prices', 'Predicted Prices'],
                                                         var_name='Price',value_name='Giá dự báo')

        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAE: {mae:.4f}")

        fig_compare = px.line(compare_df_long,x='Ngày', y='Giá dự đoán', color='Price', title="📈 So sánh giá thực tế và giá dự đoán", color_discrete_map={'Actual Prices': 'blue','Predicted Prices': 'orange'})
        fig.update_layout(xaxis_title='Ngày', yaxis_title='Giá dự đoán')
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("---")

        fig = px.line(forecast_df, y='Predict_Close', title="📈 Dự đoán xu hướng giá đóng cửa 10 ngày tiếp theo")
        fig.update_traces(line=dict(color='red'))
        fig.update_layout(xaxis_title='Ngày', yaxis_title='Giá dự đoán (VND)')

        st.plotly_chart(fig, use_container_width=True)

        next_day_price = forecast_df['Predict_Close'].iloc[0]
        st.metric("📌 Giá ngày tiếp theo", f"{next_day_price:.2f} VND")


