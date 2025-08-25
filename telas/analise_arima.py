import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima  # <- para sugerir p,d,q automaticamente

from utils import (
    moving_average_filter,
    calcular_erros,
    plot_historical_data,
    train_test_split,
    create_arima_object,
    get_model_aic_bic
)

def analiseArima(df, nome_tabela, selected_graficos):
    st.title("📊 Medidas de análise de precisão - ARIMA")
    
    # ====== Sugestão automática dos parâmetros com auto_arima ======
    series = df["QUANT"].astype(float)
    suggestion = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
    p_recommend, d_recommend, q_recommend = suggestion.order

    # ====== Obtenção dos dados da interface do streamlit ======
    with st.sidebar:
        ordem_filtro = st.slider("Ordem do filtro(semanas)", 1, 52, q_recommend if q_recommend > 0 else 1)
        p = st.slider("Parâmetro p (AutoRegressivo)", 0, 20, p_recommend)
        d = st.slider("Parâmetro d (Integração)", 0, 2, d_recommend)
        q = ordem_filtro  # <- no seu caso, q = ordem_filtro
    
    st.info(f"🔹 Parâmetros iniciais recomendados automaticamente pelo método **auto_arima**: p={p_recommend}, d={d_recommend}, q={q_recommend}")

    # ====== Continuação do seu código ======
    col1, col2 = st.columns(2)
    
    ##grafico sem média móvel
    selected_productS = df
    selected_productS = moving_average_filter(selected_productS, 1)
    selected_product_titleS = selected_graficos + " - Ordem do Filtro: 1 (semanas)"
    with col1:
        plot_historical_data(selected_productS, selected_product_titleS) 
    
    #primeiro gráfico
    selected_product = df
    selected_product = moving_average_filter(selected_product, ordem_filtro)
    selected_product_title = selected_graficos + f" - Ordem do Filtro: {ordem_filtro} (semanas)"
    with col2:
        plot_historical_data(selected_product, selected_product_title) 

    #dividindo dados para treino e teste
    train_data, test_data, tamanhoPrevisao = train_test_split(selected_product)
    tamanhoPrevisao += 1

    #treino do ARIMA
    prof_train, forecast_train, df_train, future_train, fig, future_dates = create_arima_object(train_data, tamanhoPrevisao, p, d, q)
    forecast_train = forecast_train.iloc[0:]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(fig)

    with col2:   
        prof_test, forecast_test, df_test, future_test, fig, future_dates = create_arima_object(test_data, 1, p, d, q)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_data['DATA'], forecast_train, color='orange', linestyle='--', label='Previsão')
        ax.plot(test_data['DATA'], test_data['QUANT'], color='blue', label='Real')
        ax.set_title('Previsão vs Real')
        ax.set_xlabel('Data')
        ax.set_ylabel('Quantidade')
        ax.legend()
        st.pyplot(fig)
        
    with col3:
        train_dataS, test_dataS, tamanhoPrevisao = train_test_split(selected_productS)
        prof_testS, forecast_testS, df_testS, future_testS, figS, future_datesS = create_arima_object(test_dataS, 1, p, d, q)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_data['DATA'], forecast_train, color='orange', linestyle='--', label='Previsão')
        ax.plot(test_data['DATA'], test_dataS['QUANT'], color='blue', label='Real')
        ax.set_title('Previsão vs Real (sem média móvel)')
        ax.set_xlabel('Data')
        ax.set_ylabel('Quantidade')
        ax.legend()
        st.pyplot(fig)

    option2 = st.checkbox('Mostrar detalhes dos gráficos')
    if option2:
        st.write(forecast_train)

    # === Cálculo de AIC e BIC ===
    aic, bic = get_model_aic_bic(train_data, p, d, q)
    
    st.subheader("📌 Critérios de Informação")
    col1, col2= st.columns(2)
    with col1: 
        st.write(f"**AIC:** {aic:.2f}")
    with col2: 
        st.write(f"**BIC:** {bic:.2f}")


    option = 'Mostrar dados de análise de precisão'
    if option:
        test_data.reset_index(drop=True, inplace=True)
        forecast_train.reset_index(drop=True, inplace=True)
        calcular_erros(test_data, forecast_train, st, option2)
        