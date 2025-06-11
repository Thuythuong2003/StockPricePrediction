import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import torch.nn as nn
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
        fig = px.line(data, x='Datetime', y='close')

        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                            subplot_titles=("Closing Price", "SMA 20", "EMA 20", "MACD", "RSI"))

        # Biểu đồ giá đóng cửa
        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['close'], name='Close Price'), row=1, col=1)

        # Chỉ báo SMA
        if 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'), row=2, col=1)

        # Chỉ báo EMA
        if 'EMA_20' in data.columns:
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'), row=3, col=1)

        # MACD
        if 'MACD' in data.columns:
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['MACD'], name='MACD'), row=4, col=1)

        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['RSI'], name='RSI'), row=5, col=1)

            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=[70]*len(data),
                mode= 'lines',
                name = 'Overbought (70)',
                line = dict(color='red', dash='dash') 
            ), row=5, col=1)

            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=[30]*len(data),
                mode= 'lines',
                name = 'Oversold (30)',
                line = dict(color='green', dash='dash') 
            ), row=5, col=1)

        # Cập nhật trục x để hiển thị ngày ở tất cả biểu đồ
        fig.update_xaxes(tickformat="%Y-%m-%d", row=5, col=1)  

        fig.update_layout(height=1000, showlegend=True)
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

    
with tab2:
    st.header("📈 Dự đoán giá cổ phiếu")

    st.markdown("### Chọn mô hình")
    model = st.selectbox("Chọn mô hình", ["NODE", "ARIMA", "ARIMA-GRU", "LSTM", "LSTM-ARO", "GRU"])
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


        try:
            if model == 'NODE':
                model_path = f'../model/NODE-{ticker}.pth'
                node_model = load_node_model(model_path)
                with st.expander("📋 Cấu trúc mô hình NODE"):
                    for i, layer in enumerate(node_model.odefunc.net):
                        if isinstance(layer, nn.Linear):
                            st.write(f"**Linear layer {i}:** `{layer.in_features}` → `{layer.out_features}`")

                st.session_state.predicted_values = forecast_node(node_model, scaler, test_scaled)
                rmse, mae, predict_test_price, actual_test_price = evaluate_node(node_model, x_test, y_test, scaler)
            
            elif model == 'ARIMA':
                with st.spinner("🔄 Đang chạy mô hình ARIMA..."):
                    with open(f'../model/ARIMA-{ticker}.pkl', 'rb') as f:
                        arima_model = pickle.load(f)

                p, d, q = arima_model.order
                with st.expander("📋 Tham số mô hình ARIMA"):
                    st.write(f"**ARIMA order:** (p={p}, d={d}, q={q})")

                rmse, mae, predict_test_price, actual_test_price = evaluate_arima(arima_model, test)

            elif model == 'ARIMA-GRU':
                gru_model = load_model(f'../model/GRU-{ticker}.h5', custom_objects={'mse': 'mean_squared_error'})
                with open(f'../model/ARIMA-{ticker}.pkl', 'rb') as f:
                    arima_model = pickle.load(f)

                with st.expander("📋 Cấu trúc mô hình GRU"):
                    for layer in gru_model.layers:
                        st.write(f"{layer.__class__.__name__} layer:")
                        if hasattr(layer, 'units'):
                            st.write(f"Units: {layer.units}")

                p, d, q = arima_model.order
                with st.expander("📋 Tham số mô hình ARIMA"):
                    st.write(f"**ARIMA order:** (p={p}, d={d}, q={q})")

                predicted_values = forecast_arima_gru(
                    gru_model=gru_model,
                    arima_model=arima_model,
                    x_test=x_test,
                    y_test=y_test,
                    test_scaled=test_scaled,
                    scaler=scaler,
                    forecast_len=10
                )
                rmse, mae, predict_test_price, actual_test_price = evaluate_arimagru(gru_model,arima_model, x_test, y_test,test ,scaler)

            
            else:  # Các mô hình Keras khác
                keras_model_path = f'../model/{model}-{ticker}.h5'
                keras_model = load_model(keras_model_path, custom_objects={'mse': 'mean_squared_error'})
                with st.expander(f"📋 Cấu trúc mô hình {model}"):
                    for layer in keras_model.layers:
                        if 'LSTM' in layer.__class__.__name__ or 'GRU' in layer.__class__.__name__:
                            st.write(f"{layer.__class__.__name__} layer: | Units: `{layer.units}`")
                        elif 'Dropout' in layer.__class__.__name__:
                            st.write(f"Dropout layer: | Rate: `{layer.rate}`")
                        elif 'Dense' in layer.__class__.__name__:
                            st.write(f"Dense layer: | Units: `{layer.units}`")
                st.session_state.predicted_values = forecast_keras(keras_model, scaler, test_scaled)
                rmse, mae, predict_test_price, actual_test_price = evaluate_lstm(keras_model, x_test, y_test, scaler)



        except Exception as e:
            st.error(f"❌ Lỗi khi chạy mô hình {model}: {e}")
            st.stop()

        # Hiển thị kết quả dự báo nếu thành công
        if st.session_state.predicted_values is not None and model != 'ARIMA':
            forecast_dates = pd.date_range(start=data['Datetime'].iloc[-1] + timedelta(days=1), periods=10, freq=BDay())
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predict_Close': st.session_state.predicted_values.flatten()
            })
            
            compare_df = pd.DataFrame({
                'Actual Prices': actual_test_price.flatten(),
                'Predicted Prices': predict_test_price.flatten()
            }, index=range(len(y_test)))
            compare_df.index.name = "Ngày"
            compare_df_long = compare_df.reset_index().melt(id_vars='Ngày',value_vars=['Actual Prices', 'Predicted Prices'],
                                                            var_name='Price',value_name='Giá dự đoán')

            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"MAE: {mae:.4f}")

            fig_compare = px.line(compare_df_long,x='Ngày', y='Giá dự đoán', color='Price', title="📈 So sánh giá thực tế và giá dự đoán", color_discrete_map={'Actual Prices': 'blue','Predicted Prices': 'orange'})
            fig.update_layout(xaxis_title='Ngày', yaxis_title='Giá dự đoán')
            st.plotly_chart(fig_compare, use_container_width=True)

            st.markdown("---")
            st.subheader("🔮 Kết quả dự đoán xu hướng giá đóng cửa 10 ngày tiếp theo")

            fig = px.line(forecast_df,  y='Predict_Close', title="📈 Dự đoán xu hướng giá đóng cửa")
            fig.update_traces(line=dict(color='red'))
            fig.update_layout(xaxis_title='Ngày', yaxis_title='Giá dự đoán')
            st.plotly_chart(fig, use_container_width=True)

            next_day_price = forecast_df['Predict_Close'].iloc[0]
            st.metric("📌 Giá ngày tiếp theo", f"{next_day_price:.2f}")

        elif st.session_state.predicted_values is not None and model == 'ARIMA':

                
            compare_df = pd.DataFrame({
                'Actual Prices': actual_test_price.values,
                'Predicted Prices': predict_test_price.values
            }, index=range(len(test)))
            compare_df.index.name = "Ngày"
            compare_df_long = compare_df.reset_index().melt(id_vars='Ngày',value_vars=['Actual Prices', 'Predicted Prices'],
                                                            var_name='Price',value_name='Giá dự đoán')

            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"MAE: {mae:.4f}")


            fig_compare = px.line(compare_df_long,x='Ngày', y='Giá dự đoán', color='Price', title="📈 So sánh giá thực tế và giá dự đoán", color_discrete_map={'Actual Prices': 'blue','Predicted Prices': 'orange'})
            fig_compare.update_layout(xaxis_title='Ngày', yaxis_title='Giá dự đoán')
            st.plotly_chart(fig_compare, use_container_width=True)
