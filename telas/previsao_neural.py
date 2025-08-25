import streamlit as st
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
from utils import moving_average_filter, plot_historical_data, generate_forecast_table, download_link_excel, create_neural_object, generate_components_report

def previsaoNeural(df, name, selected_graficos):
    st.title('Previsão de demanda')
    with st.sidebar:
        ordem_filtro = st.slider("Ordem do filtro (semanas)", 1, 52)
        periodo = st.slider("Intervalo a ser previsto(semanas)", 1, 24, 12) + 1
    
    #Utilização dos dados do streamlit para a execução
    selected_product = df
    selected_product = moving_average_filter(selected_product, ordem_filtro)
    selected_product_title = selected_graficos + " - Ordem do Filtro: " + str(ordem_filtro) + " (semanas)"
    selected_product_title2 = "boxplot de " + selected_graficos + " - Ordem do Filtro: " + str(ordem_filtro) + " (semanas)"
    col1, col2 = st.columns(2)
    with col1:
        plot_historical_data(selected_product, selected_product_title)  
        
        
        prof, forecast, df, future, fig = create_neural_object(selected_product, periodo)
        st.write(fig)
        #generate_forecast_report(prof, forecast, 'Previsão de demanda')
        
        ##########
        #forecast_table = generate_forecast_table(forecast, 'Previsão de Demanda - Tabela', periodo)
        #st.write(forecast_table)
        #######

    with col2:
        generate_components_report(prof, forecast, 'Componentes', df, future)
    
        #fig = prof.plot_components(forecast)
            
        # Removendo o terceiro subplot que mostra os dias da semana
        #fig.delaxes(fig.axes[3])
        #fig.delaxes(fig.axes[2])
        #fig.delaxes(fig.axes[0])
        #st.pyplot(fig)
        
    #forecast_table = generate_forecast_table(forecast, 'Previsão de Demanda - Tabela', periodo)
    forecast_renamed = forecast.rename(columns={'ds': 'Data', 'yhat1': 'Previsão média'})
    forecast_renamed = forecast_renamed.drop(columns=['y', 'trend', 'season_weekly', 'season_yearly'])
    forecast_renamed = forecast_renamed.iloc[1:]
    st.write(forecast_renamed)
    st.markdown(download_link_excel(forecast_renamed, 'forecast'), unsafe_allow_html=True)
