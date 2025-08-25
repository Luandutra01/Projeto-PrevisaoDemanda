import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from utils import (
    moving_average_filter,
    calcular_erros,
    plot_historical_data,
    train_test_split,
    create_arima_object
)

def analiseArima(df, nome_tabela, selected_graficos):
    st.title("📊 Medidas de análise de precisão - ARIMA")
    
    # Obtenção dos dados da interface do streamlit
    st.title("Medidas de análise de precisão")
    with st.sidebar:
        ordem_filtro = st.slider("Ordem do filtro(semanas)", 1, 52)
        p = st.slider('Parâmetro p (AutoRegressivo)', 0, 20, 1)       
        d = st.slider('Parâmetro d (Integração)', 0, 2, 1)
        q = st.slider('Parâmetro q (Média móvel)', 0, 52, 1)      
    
    col1, col2 = st.columns(2)
    
    ##grafico sem média móvel
    selected_productS = df
    selected_productS = moving_average_filter(selected_productS, 1)
    selected_product_titleS = selected_graficos + " - Ordem do Filtro: " + str(1) + " (semanas)"
    with col1:
        plot_historical_data(selected_productS, selected_product_titleS) 
    
    
    #primeiro gráfico
    
    #Utilização dos dados do streamlit para a execução
    selected_product = df
    selected_product = moving_average_filter(selected_product, ordem_filtro)
    selected_product_title = selected_graficos + " - Ordem do Filtro: " + str(ordem_filtro) + " (semanas)"
    selected_product_title2 = "boxplot de " + selected_graficos + " - Ordem do Filtro: " + str(ordem_filtro) + " (semanas)"
    with col2:
        plot_historical_data(selected_product, selected_product_title) 
    
    
    #segundo gráfico
    #boxplot_historical_data(selected_product, selected_product_title2)
    
    #dividindo dados para treino e teste
    train_data, test_data, tamanhoPrevisao = train_test_split(selected_product)
    tamanhoPrevisao += 1
    
    
    #terceiro gráfico/previsão
    #periodo = st.slider("Intervalo a ser previsto(semanas)", 12, 38)      
    
    prof_train, forecast_train, df_train, future_train, fig, future_dates = create_arima_object(train_data, tamanhoPrevisao, p, d, q)
    #prof_train, forecast_train, df_train, future_train = create_profet_object(train_data, 38)
    forecast_train = forecast_train.iloc[0:]
    
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        #generate_forecast_report(prof_train, forecast_train, 'Previsão de demanda')
        st.pyplot(fig)
    with col2:   
        
        prof_test, forecast_test, df_test, future_test, fig, future_dates = create_arima_object(test_data, 1, p, d, q)
    
        # Criar o gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotar a linha de previsão
        ax.plot(test_data['DATA'], forecast_train, color='orange', linestyle='--', label='Previsão')
        
        # Plotar a linha de dados reais
        ax.plot(test_data['DATA'], test_data['QUANT'], color='blue', label='Real')
        
        # Adicionar títulos e legendas
        ax.set_title('Previsão vs Real')
        ax.set_xlabel('Data')
        ax.set_ylabel('Quantidade')
        ax.legend()
        
        # Exibir o gráfico no Streamlit
        st.pyplot(fig)
        
    with col3:
        #dividindo dados para treino e teste
        train_dataS, test_dataS, tamanhoPrevisao = train_test_split(selected_productS)
        prof_testS, forecast_testS, df_testS, future_testS, figS, future_datesS = create_arima_object(test_dataS, 1, p, d, q)
        
       # Criar o gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotar a linha de previsão
        ax.plot(test_data['DATA'], forecast_train, color='orange', linestyle='--', label='Previsão')
        
        # Plotar a linha de dados reais
        ax.plot(test_data['DATA'], test_dataS['QUANT'], color='blue', label='Real')
        
        # Adicionar títulos e legendas
        ax.set_title('Previsão vs Real (sem média móvel)')
        ax.set_xlabel('Data')
        ax.set_ylabel('Quantidade')
        ax.legend()
        
        # Exibir o gráfico no Streamlit
        st.pyplot(fig)
        
    option2 = st.checkbox('Mostrar detalhes dos gráficos')
    #option2 = 'Mostrar detalhes dos gráficos'
    #tabela previsão
    if option2:
        st.write(forecast_train)
    
    
    # Criar uma caixa de seleção
    #option = st.checkbox('Mostrar dados de análise de precisão')
    option = 'Mostrar dados de análise de precisão'
    
    
    # Verificar se a caixa de seleção está marcada
    
    if option:
        #if periodo != 0:
        #    forecast_table = forecast_table[:-periodo]
        #    test_data = test_data[:-periodo]
    
        # Redefinir os índices para garantir que estejam alinhados corretamente
        test_data.reset_index(drop=True, inplace=True)
        forecast_train.reset_index(drop=True, inplace=True)
        
        if option2:
            st.write("Dados para teste:")
            st.write(test_data)     
            st.write("Coluna previsão média")
            
            #st.write(forecast_train['predicted_mean']
            st.write(forecast_train)
            
            st.write("Coluna dados para teste")
            st.write(test_data['QUANT'])
    
        option3 = st.checkbox('Mostrar em relação aos dados sem média móvel')
        if option3:
            ############ Em relação a sem média móvel
            st.write('Em relação ao dado sem média móvel')
            selected_product = df
            selected_product = moving_average_filter(selected_product, 1)
            selected_product_title = selected_graficos + " - Ordem do Filtro: " + str(1) + " (semanas)"
            train_data, test_data, tamanhoPrevisao = train_test_split(selected_product)
            test_data.reset_index(drop=True, inplace=True)
            
            if option2:
                st.write("Dados para teste-sem média móvel:")
                st.write(test_data)
            
        ####função
        calcular_erros(test_data, forecast_train, st, option2)
