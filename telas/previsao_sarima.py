import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from utils import moving_average_filter, plot_historical_data, download_link_excel
import pandas as pd
from pmdarima import auto_arima

def previsaoSarima(df, name, selected_graficos):
    st.title('Previsão com SARIMA')

    # ====== Sugestão automática dos parâmetros com auto_arima ======
    series = df["QUANT"].astype(float)
    suggestion = auto_arima(series, seasonal=True, m=52, stepwise=True, suppress_warnings=True)
    p_recommend, d_recommend, q_recommend = suggestion.order
    P_recommend, D_recommend, Q_recommend, s_recommend = suggestion.seasonal_order
    
    # ====== Obtenção dos dados da interface do streamlit ======
    with st.sidebar:
        ordem_filtro = st.slider("Ordem do filtro(semanas)", 1, 52, q_recommend if q_recommend > 0 else 1)
        periodo = st.slider("Intervalo a ser previsto(semanas)", 1, 24, 12)
        p = st.slider("Parâmetro p (AR)", 0, 20, p_recommend)
        d = st.slider("Parâmetro d (I)", 0, 2, d_recommend)
        q = ordem_filtro  # usamos ordem do filtro
        P = st.slider("Parâmetro P (AR sazonal)", 0, 5, P_recommend)
        D = st.slider("Parâmetro D (I sazonal)", 0, 2, D_recommend)
        Q = st.slider("Parâmetro Q (MA sazonal)", 0, 5, Q_recommend)
        s = st.slider("Periodicidade sazonal (s)", 2, 104, s_recommend)  # semanal, anual, etc.

    st.info(f"🔹 Parâmetros recomendados automaticamente pelo auto_arima: "
            f"p={p_recommend}, d={d_recommend}, q={q_recommend}, "
            f"P={P_recommend}, D={D_recommend}, Q={Q_recommend}, s={s_recommend}")

    # ====== Preparação dos dados ======
    selected_product = moving_average_filter(df, ordem_filtro)
    df_fit = selected_product.rename(columns={'DATA': 'ds', 'QUANT': 'y'}).dropna()
    title = f"{selected_graficos} - Ordem do Filtro: {ordem_filtro} (semanas)"
    
    col1, col2 = st.columns(2)
    with col1:
        plot_historical_data(selected_product, title)

    with col2:
        # Treino do modelo SARIMA
        model = SARIMAX(df_fit['y'], order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)
        future_dates = pd.date_range(df_fit['ds'].iloc[-1], periods=periodo+1, freq='W')[1:]
        forecast = model_fit.forecast(steps=periodo)
    
        # Plotando histórico e previsão
        fig, ax = plt.subplots()
        ax.plot(df_fit['ds'], df_fit['y'], label='Histórico')
        ax.plot(future_dates, forecast, label='Previsão', linestyle='--')
        ax.legend()
        st.pyplot(fig)
    
    # Exibição da tabela de previsão e opção de download
    forecast_df = pd.DataFrame({'Data': future_dates, 'Previsão média': forecast})
    st.write(forecast_df)
    st.markdown(download_link_excel(forecast_df, 'forecast_sarima'), unsafe_allow_html=True)
