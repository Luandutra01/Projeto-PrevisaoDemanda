import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from utils import (
    moving_average_filter,   # com min_periods=1
    train_test_split,
    calcular_erros,
    plot_historical_data,
)

# =========================================================
# Funções auxiliares
# =========================================================

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_and_forecast_lstm(df, window_size=12, epochs=50):
    # Garantia contra NaN
    df = df.dropna(subset=['QUANT']).reset_index(drop=True)

    values = df['QUANT'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    X, y = create_sequences(values_scaled, window_size)

    if len(X) == 0:
        return None, pd.DataFrame(columns=['DATA', 'yhat1'])

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = create_lstm_model((window_size, 1))
    model.fit(X, y, epochs=epochs, verbose=0)

    y_pred = model.predict(X, verbose=0)
    y_pred = scaler.inverse_transform(y_pred)

    forecast = pd.DataFrame({
        'DATA': df['DATA'].iloc[window_size:].values,
        'yhat1': y_pred.flatten()
    })

    return model, forecast


# =========================================================
# Tela principal
# =========================================================

def analiseRede(df, nome_tabela, selected_graficos):

    st.title("📊 Análise – Rede Neural (LSTM)")
    st.caption(
        "O modelo é treinado com dados suavizados por média móvel, "
        "mas avaliado com base nos dados originais."
    )

    with st.sidebar:
        ordem_filtro = st.slider("Ordem da média móvel (semanas)", 1, 52, 1)
        window_size = st.slider("Janela temporal (semanas)", 4, 100, 12)
        epochs = st.slider("Épocas de treino", 10, 200, 50, 10)

    # =====================================================
    # Dados com e sem média móvel
    # =====================================================
    df_mm = moving_average_filter(df.copy(), ordem_filtro)
    df_original = df.copy().reset_index(drop=True)

    col1, col2 = st.columns(2)

    # =====================================================
    # Gráfico série histórica (média móvel)
    # =====================================================

    selected_product = moving_average_filter(df, ordem_filtro)
    df_fit = selected_product.rename(columns={'DATA': 'ds', 'QUANT': 'y'}).dropna()
    title = f"{selected_graficos} - Ordem do Filtro: {ordem_filtro} (semanas)"
    with col1:
        plot_historical_data(selected_product, title)

    # =====================================================
    # Split treino / teste (baseado no df_mm)
    # =====================================================
    train_mm, test_mm, _ = train_test_split(df_mm)

    # Alinha dados originais ao mesmo período do teste
    test_original = df_original.iloc[
        -len(test_mm):
    ].reset_index(drop=True)

    test_mm = test_mm.reset_index(drop=True)

    # =====================================================
    # Treinamento LSTM
    # =====================================================
    model, forecast_train = train_and_forecast_lstm(
        train_mm,
        window_size=window_size,
        epochs=epochs
    )

    if forecast_train.empty:
        st.warning("Série muito curta para a janela temporal escolhida.")
        return

    # Alinha previsão ao período do teste ORIGINAL
    forecast_test = forecast_train.tail(len(test_original)).reset_index(drop=True)

    # =====================================================
    # Gráfico previsão vs real (DADOS ORIGINAIS)
    # =====================================================
    with col2:
        st.subheader("Previsão vs Real (dados originais)")
        fig, ax = plt.subplots()

        ax.plot(
            test_original['DATA'],
            test_original['QUANT'],
            label='Real (original)'
        )

        ax.plot(
            test_original['DATA'],
            forecast_test['yhat1'],
            '--',
            label='Previsão LSTM (treinada com MM)'
        )

        ax.set_title("LSTM – Treino com MM / Avaliação Original")
        ax.set_xlabel("Data")
        ax.set_ylabel("Quantidade")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.legend()

        st.pyplot(fig)

    # =====================================================
    # Detalhes
    # =====================================================
    option2 = st.checkbox("Mostrar tabela de previsão")
    if option2:
        st.subheader("Tabela de previsão")
        st.write(
            pd.concat(
                [test_original[['DATA', 'QUANT']], forecast_test['yhat1']],
                axis=1
            )
        )

    st.subheader("Métricas de erro (dados originais)")
    calcular_erros(
        test_original,
        forecast_test['yhat1'],
        st,
        option2
    )
