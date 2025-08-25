import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils import (
    moving_average_filter,
    calcular_erros,
    plot_historical_data,
    train_test_split
)

# ==== Função cacheada para treinar SARIMA =====
@st.cache_data
def fit_sarima(df, p, d, q, P, D, Q, m):
    model = SARIMAX(df['y'], order=(p, d, q), seasonal_order=(P, D, Q, m))
    return model.fit(disp=False)

# ==== Página SARIMA no modelo do ARIMA ====
def analiseSarima(df, nome_tabela, selected_graficos):
    st.title("📊 Medidas de análise de precisão - SARIMA")
    
    # ===== Sugestão automática limitada =====
    series = df["QUANT"].astype(float)
    suggestion = auto_arima(
        series,
        seasonal=True,
        m=52,  # ajuste conforme frequência (semanal/anual)
        max_p=3, max_q=3, max_P=2, max_Q=2,  # limita combinações para reduzir memória
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )

    p_recommend, d_recommend, q_recommend = suggestion.order
    P_recommend, D_recommend, Q_recommend, m_recommend = suggestion.seasonal_order

    # ===== Sidebar =====
    with st.sidebar:
        ordem_filtro = st.slider("Ordem do filtro (semanas)", 1, 52, q_recommend if q_recommend > 0 else 1)
        p = st.slider("Parâmetro p (AR)", 0, 20, p_recommend)
        d = st.slider("Parâmetro d (I)", 0, 2, d_recommend)
        q = st.slider("Parâmetro q (MA)", 0, 20, q_recommend)
        P = st.slider("Parâmetro P (Sazonal AR)", 0, 5, P_recommend)
        D = st.slider("Parâmetro D (Sazonal I)", 0, 2, D_recommend)
        Q = st.slider("Parâmetro Q (Sazonal MA)", 0, 5, Q_recommend)
        m = st.slider("Período sazonal (m)", 1, 60, m_recommend)

    st.info(f"🔹 Parâmetros recomendados pelo auto_arima: "
            f"(p,d,q)=({p_recommend},{d_recommend},{q_recommend}), "
            f"(P,D,Q,m)=({P_recommend},{D_recommend},{Q_recommend},{m_recommend})")

    # ===== Gráficos =====
    col1, col2 = st.columns(2)
    
    # Série sem filtro
    selected_productS = moving_average_filter(df, 1)
    df_fitS = selected_productS.rename(columns={'DATA':'ds','QUANT':'y'}).dropna()
    with col1:
        plot_historical_data(selected_productS, selected_graficos + " - Ordem do Filtro: 1 (semanas)")

    # Série com filtro
    selected_product = moving_average_filter(df, ordem_filtro)
    df_fit = selected_product.rename(columns={'DATA':'ds','QUANT':'y'}).dropna()
    with col2:
        plot_historical_data(selected_product, selected_graficos + f" - Ordem do Filtro: {ordem_filtro} (semanas)")

    # ===== Divisão treino/teste =====
    train_data, test_data, tamanhoPrevisao = train_test_split(selected_product)
    tamanhoPrevisao += 1

    # ===== Treino SARIMA com caching =====
    prof_train = fit_sarima(train_data.rename(columns={'DATA':'ds','QUANT':'y'}), p,d,q,P,D,Q,m)
    future_dates = pd.date_range(train_data['DATA'].iloc[-1] + pd.Timedelta(weeks=1), periods=tamanhoPrevisao, freq='W')
    forecast_train = prof_train.forecast(steps=tamanhoPrevisao)
    
    # ===== Gráfico treino =====
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(train_data['DATA'], train_data['QUANT'], label='Histórico')
    ax.plot(future_dates, forecast_train, linestyle='--', color='orange', label='Previsão')
    ax.set_title('Treino SARIMA')
    ax.set_xlabel('Data')
    ax.set_ylabel('Quantidade')
    ax.legend()
    st.pyplot(fig)

    # ===== Comparação teste =====
    prof_test = fit_sarima(test_data.rename(columns={'DATA':'ds','QUANT':'y'}), p,d,q,P,D,Q,m)
    future_dates_test = pd.date_range(test_data['DATA'].iloc[0], periods=len(test_data), freq='W')
    forecast_test = prof_test.forecast(steps=len(test_data))

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(test_data['DATA'], test_data['QUANT'], label='Real')
    ax.plot(future_dates_test, forecast_test, linestyle='--', color='orange', label='Previsão')
    ax.set_title('Previsão vs Real (Teste)')
    ax.set_xlabel('Data')
    ax.set_ylabel('Quantidade')
    ax.legend()
    st.pyplot(fig)

    # ===== Mostrar detalhes =====
    if st.checkbox('Mostrar detalhes dos gráficos'):
        st.write(forecast_train)

    # ===== Mostrar métricas =====
    if st.checkbox('Mostrar dados de análise de precisão'):
        test_data.reset_index(drop=True, inplace=True)
        forecast_train.reset_index(drop=True, inplace=True)
        calcular_erros(test_data, forecast_train, st, True)
