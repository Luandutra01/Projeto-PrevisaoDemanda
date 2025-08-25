import streamlit as st
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import (
    moving_average_filter, 
    plot_historical_data, 
    train_test_split, 
    create_profet_object, 
    generate_forecast_report,
    generate_forecast_report2,
    generate_forecast_table,
    calcular_erros,
    calcular_porcentagem_entre_min_max
)

def analiseProphet(df, name, selected_graficos):
    # Obtenção dos dados da interface do streamlit
    st.title("Medidas de análise de precisão")
    with st.sidebar:
        ordem_filtro = st.slider("Ordem do filtro(semanas)", 1, 52)

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
    
    prof_train, forecast_train, df_train, future_train = create_profet_object(train_data, tamanhoPrevisao)
    #prof_train, forecast_train, df_train, future_train = create_profet_object(train_data, 38)

    
    col1, col2, col3 = st.columns(3)
    with col1:
        generate_forecast_report(prof_train, forecast_train, 'Previsão de demanda')
    with col2:     
        prof_test, forecast_test, df_test, future_test = create_profet_object(test_data, 0)
        generate_forecast_report2(prof_test, forecast_train, 'Previsão vs valor', tamanhoPrevisao)
                 
        ##tabela previsão
        forecast_table = generate_forecast_table(forecast_train, tamanhoPrevisao)
        #forecast_table = generate_forecast_table(forecast_train, 'Previsão de Demanda - Tabela', 38)
    with col3:
        #dividindo dados para treino e teste
        train_dataS, test_dataS, tamanhoPrevisao = train_test_split(selected_productS)
        prof_testS, forecast_testS, df_testS, future_testS = create_profet_object(test_dataS, 0)
        generate_forecast_report2(prof_testS, forecast_train, 'VS Previsão sem média móvel', tamanhoPrevisao)           
            
       

    option2 = st.checkbox('Mostrar detalhes dos gráficos')
    #option2 = 'Mostrar detalhes dos gráficos'
    #tabela previsão
    if option2:
        st.write('Previsão de Demanda - Tabela')
        st.write(forecast_table)
    

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
        forecast_table.reset_index(drop=True, inplace=True)
        
        if option2:
            st.write("Dados para teste:")
            st.write(test_data)     
            st.write("Coluna previsão média")
            st.write(forecast_table['Previsão média'])
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
        calcular_erros(test_data, forecast_table['Previsão média'], st, option2)
