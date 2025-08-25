import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from utils import moving_average_filter, plot_historical_data, download_link_excel
import pandas as pd

def previsaoArima(df, name, selected_graficos):
    st.title('Previsão com ARIMA')
    with st.sidebar:
        ordem_filtro = st.slider("Ordem do filtro (semanas)", 1, 52)
        periodo = st.slider("Intervalo a ser previsto(semanas)", 1, 24, 12)
        p = st.slider('Parâmetro p (AutoRegressivo)', 0, 20, 1)       
        d = st.slider('Parâmetro d (Integração)', 0, 2, 1)
        q = 0  

    selected_product = moving_average_filter(df, ordem_filtro)
    df_fit = selected_product.rename(columns={'DATA': 'ds', 'QUANT': 'y'}).dropna()
    title = f"{selected_graficos} - Ordem do Filtro: {ordem_filtro} (semanas)"
    
    col1, col2 = st.columns(2)
    with col1:
        plot_historical_data(selected_product, title)

    with col2:
        model = ARIMA(df_fit['y'], order=(p, d, q))
        model_fit = model.fit()
        future_dates = pd.date_range(df_fit['ds'].iloc[-1], periods=periodo+1, freq='W')[1:]
        forecast = model_fit.forecast(steps=periodo)
    
    
        fig, ax = plt.subplots()
        ax.plot(df_fit['ds'], df_fit['y'], label='Histórico')
        ax.plot(future_dates, forecast, label='Previsão', linestyle='--')
        ax.legend()
        st.pyplot(fig)
    
    forecast_df = pd.DataFrame({'Data': future_dates, 'Previsão média': forecast})
    st.write(forecast_df)
    st.markdown(download_link_excel(forecast_df, 'forecast_arima'), unsafe_allow_html=True)
