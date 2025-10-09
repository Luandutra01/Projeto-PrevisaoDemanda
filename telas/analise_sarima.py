import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import auto_arima  
import matplotlib.dates as mdates

from utils import (
    moving_average_filter,
    calcular_erros,
    plot_historical_data,
    train_test_split,
    create_sarima_object,
    get_sarima_aic_bic
)

def analiseSarima(df, nome_tabela, selected_graficos):
    st.title("📊 Medidas de análise de precisão - SARIMA")
    
    # Série base
    series = df["QUANT"].astype(float)

    # ====== Checkbox para decidir se roda o auto_arima ======
    usar_auto = st.checkbox("Rodar auto_arima para sugerir parâmetros?", value=False)

    if usar_auto:
        st.info("⏳ Rodando auto_arima... isso pode demorar um pouco.")
        suggestion = auto_arima(series, seasonal=True, m=52, stepwise=True, suppress_warnings=True)  
        p_recommend, d_recommend, q_recommend = suggestion.order
        P_recommend, D_recommend, Q_recommend, m_recommend = suggestion.seasonal_order
    else:
        # Valores padrão caso não rode o auto_arima
        p_recommend, d_recommend, q_recommend = 1, 0, 1
        P_recommend, D_recommend, Q_recommend, m_recommend = 0, 0, 0, 52

    # ====== Obtenção dos dados da interface do streamlit ======
    with st.sidebar:
        ordem_filtro = st.slider("Ordem do filtro(semanas)", 1, 52, q_recommend if q_recommend > 0 else 1)
        p = st.slider("Parâmetro p (AutoRegressivo)", 0, 20, p_recommend)
        d = st.slider("Parâmetro d (Integração)", 0, 2, d_recommend)
        q = ordem_filtro

        P = st.slider("Parâmetro P (Sazonal AR)", 0, 5, P_recommend)
        D = st.slider("Parâmetro D (Sazonal I)", 0, 2, D_recommend)
        Q = st.slider("Parâmetro Q (Sazonal MA)", 0, 5, Q_recommend)
        m = st.slider("Período sazonal (m)", 1, 60, m_recommend)

    if usar_auto:
        st.success(f"🔹 Parâmetros sugeridos: "
                   f"(p,d,q)=({p_recommend},{d_recommend},{q_recommend}), "
                   f"(P,D,Q,m)=({P_recommend},{D_recommend},{Q_recommend},{m_recommend})")
    else:
        st.warning("⚠️ Auto_arima não rodou. Ajuste manualmente os parâmetros nos sliders.")

    # ====== Gráficos ======
    col1, col2 = st.columns(2)
    
    # Série sem filtro
    selected_productS = moving_average_filter(df, 1)
    with col1:
        plot_historical_data(selected_productS, selected_graficos + " - Ordem do Filtro: 1 (semanas)") 
    
    # Série com filtro escolhido
    selected_product = moving_average_filter(df, ordem_filtro)
    with col2:
        plot_historical_data(selected_product, selected_graficos + f" - Ordem do Filtro: {ordem_filtro} (semanas)") 

    # Divisão treino/teste
    train_data, test_data, tamanhoPrevisao = train_test_split(selected_product)
    tamanhoPrevisao += 1

    # ====== treino do SARIMA ======
    prof_train, forecast_train, df_train, future_train, fig, future_dates = create_sarima_object(
        train_data, tamanhoPrevisao, p, d, q, P, D, Q, m
    )
    forecast_train = forecast_train.iloc[0:]

    # ====== Gráficos de comparação ======
    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(fig)

    with col2:   
        prof_test, forecast_test, df_test, future_test, fig, future_dates = create_sarima_object(
            test_data, 1, p, d, q, P, D, Q, m
        )
        fig, ax = plt.subplots()
        ax.plot(test_data['DATA'], forecast_train, color='orange', linestyle='--', label='Previsão')
        ax.plot(test_data['DATA'], test_data['QUANT'], color='blue', label='Real')
        ax.set_title('Previsão vs Real')
        ax.set_xlabel('Data', fontsize=14)
        ax.set_ylabel('Quantidade', fontsize=14)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.legend()
        st.pyplot(fig)
        
    with col3:
        train_dataS, test_dataS, tamanhoPrevisao = train_test_split(selected_productS)
        prof_testS, forecast_testS, df_testS, future_testS, figS, future_datesS = create_sarima_object(
            test_dataS, 1, p, d, q, P, D, Q, m
        )
        fig, ax = plt.subplots()
        ax.plot(test_data['DATA'], forecast_train, color='orange', linestyle='--', label='Previsão')
        ax.plot(test_data['DATA'], test_dataS['QUANT'], color='blue', label='Real')
        ax.set_title('Previsão vs Real (sem média móvel)')
        ax.set_xlabel('Data')
        ax.set_ylabel('Quantidade')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.legend()
        st.pyplot(fig)

    # Mostrar previsões detalhadas
    option2 = st.checkbox('Mostrar detalhes dos gráficos')
    if option2:
        st.write(forecast_train)

    # === Cálculo de AIC e BIC ===
    aic, bic = get_sarima_aic_bic(train_data, p, d, q, P, D, Q, m)
    
    st.subheader("📌 Critérios de Informação")
    col1, col2= st.columns(2)
    with col1: 
        st.write(f"**AIC:** {aic:.2f}")
    with col2: 
        st.write(f"**BIC:** {bic:.2f}")
    
    # Mostrar métricas
    option = 'Mostrar dados de análise de precisão'
    if option:
        test_data.reset_index(drop=True, inplace=True)
        forecast_train.reset_index(drop=True, inplace=True)
        calcular_erros(test_data, forecast_train, st, option2)
