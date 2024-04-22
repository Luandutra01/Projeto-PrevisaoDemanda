# Importando as bibliotecas necessárias

import streamlit as st
# Para trabalhar com datas
import datetime
# Pandas para o gerênciamento dos dados
import pandas as pd
# Numpy para trabalhar com arrays e funções matemáticas
import numpy as np
# Matplotlib e Seaborn para plotar gráficos 
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import plotly.graph_objects as go
from plotly.offline import iplot
import altair as alt

from io import BytesIO

# Prophet para a predição de vendas
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_plotly, plot_components_plotly

def main():
    # Verificar se o usuário está logado
    if not is_user_logged_in():
        show_login_page()
    else:
        run_main_program()

def is_user_logged_in():
    # Verificar se o usuário está logado
    return st.session_state.get('logged_in', False)

def show_login_page():
    st.title("Login")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    login_button = st.button("Login")

    if login_button:
        if username == "admin" and password == "admin":
            #st.success("Login successful!")
            st.session_state['logged_in'] = True
            st.experimental_rerun()  # Re-executar o aplicativo para atualizar a tela
        else:
            st.error("Invalid username or password")

def run_main_program():
    pd.options.mode.chained_assignment = None

    #name = 'D:/LuanD/Dados semanais com gráficos.xlsx'
    name = st.file_uploader("Escolha o arquivo Excel", type=['xlsx'])
    print(name)
    
    if name is not None:
    
    # Dica para resolver o problema das semanas duplicadas https://stackoverflow.com/questions/35403752/pandas-sum-over-duplicated-indices-with-sum

        @st.cache_data
        def read_sheet(sheet):
            df = pd.read_excel(name, sheet, header=0)
            df = df.rename(columns={'QUANTIDADE': 'QUANT'})
            df = df.rename(columns={'DataInicioSemana': 'DATA'})
        
            
            df['Year'] = (df['DATA']//10000)
            df['Month'] = df['DATA'] - (df['Year'] * 10000)
            df['Month'] = df['Month'] // 100
            df['Day'] = df['DATA'] - ( df['Year'] * 10000 + df['Month'] * 100 )
        
            # Corrige algumas linhas que o dia é definido como sendo zero
            df.loc[df['Day'] < 1,'Day'] = 1
            
            # Cria um indice baseado no formato ano-semana
            df['AnoSemanaIdx'] =  df['Year'] * 100 + df['NumeroSemana']
            df = df.set_index('AnoSemanaIdx')
            
            # Corrige os dados das semanas repetidas
            df = df.groupby(df.index).agg({'NumeroSemana': 'first',
                                           'DATA': 'first', 
                                           'QUANT': 'sum',
                                           'Year': 'first',
                                           'Month': 'first',
                                           'Day': 'first'
                                          })
            
        
            # Coloca a data no formato DateTime em vez de inteiro
            df['DATA'] = pd.to_datetime(df[['Year','Month','Day']])
        
            # Corrige a semana extra no final do ano
        
            # 1 - Soma o valor da semana 53 na 52
            #fixedLastWeek = df.loc[df['NumeroSemana']==52, 'QUANT'].reset_index(drop=True) + df.loc[df['NumeroSemana']==53]['QUANT'].reset_index(drop=True)
            #cremoso[cremoso['NumeroSemana']==52]['QUANT']  = fixedLastWeek
        
            fixedLastWeek = df.loc[df['NumeroSemana']==52, 'QUANT'].reset_index(drop=True) + df.loc[df['NumeroSemana']==53]['QUANT'].reset_index(drop=True)
            dfFixedLastWeek = fixedLastWeek.to_frame()
            dfFixedLastWeek['anosemana'] = df[df['NumeroSemana']==52].index
            dfFixedLastWeek.set_index('anosemana', inplace=True)
            df.loc[df['NumeroSemana']==52,'QUANT'] = dfFixedLastWeek
            
            # 2 - Exclui a semana 53
            df = df.drop(df[df['NumeroSemana']==53].index)
            
            return df
            
        @st.cache_data
        def moving_average_filter(df, order):
            filtred_df = df.copy()
            filtred_df['QUANT'] = filtred_df['QUANT'].rolling(order).mean()
            #filtred_df.dropna()
            #filtred_df['QUANT'] = filtred_df['QUANT'].fillna(df['QUANT'][order - 1])
            return filtred_df
        
        @st.cache_data
        def create_profet_object(df, periodo):
            price_increase = pd.DataFrame({
              'holiday': 'price_increase',
              'ds': pd.to_datetime(['2018-01-01', '2018-04-01', '2018-11-01',
                                    '2019-09-01', '2020-03-01', '2020-06-01',
                                    '2021-02-01', '2021-06-01', '2022-02-01',
                                    '2022-05-01', '2022-08-01', '2023-01-01']),
              'lower_window': 0,
              'upper_window': 15,
            })
            
            specific_changepoints=['2018-01-05', '2018-04-01', '2018-11-01',
                                    '2019-09-01', '2020-03-01', '2020-06-01',
                                    '2021-02-01', '2021-06-01', '2022-02-01',
                                    '2022-05-01', '2022-08-01', '2023-01-01']
            
            prof = Prophet()
            
            
            prof.add_country_holidays(country_name='BR')
            df = df.rename(columns={'DATA': 'ds', 'QUANT': 'y'})
            prof.fit(df)
            future = prof.make_future_dataframe(periods=periodo, freq='W')
            #future.tail()
        
            forecast = prof.predict(future)
            #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            return (prof, forecast, df, future)
            
        @st.cache_data    
        def plot_historical_data(df, title):
            st.write(title)
            #st.line_chart(df.set_index('DATA'))
            chart = alt.Chart(df.reset_index()).mark_line().encode(
                x='DATA',
                #y='QUANT'
                y=alt.Y('QUANT', scale=alt.Scale(domain=[0, 200000]))
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            
        @st.cache_data
        def boxplot_historical_data(df, title):
            st.write(title)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Year', y='QUANT', ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Ano')
            ax.set_ylabel('Quantidade')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        @st.cache_data
        def generate_forecast_report(_prof, forecast, title):
            st.write(title)
            fig1 = prof.plot(forecast)
            st.pyplot(fig1)
            
        @st.cache_data
        def generate_components_report(_prof, forecast, title, df, future):
            prof = Prophet(weekly_seasonality=False)
            prof.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            forecast = prof.fit(df).predict(future)
            fig2 = prof.plot_components(forecast)
            st.pyplot(fig2)
        
        @st.cache_data
        def ler_nomes_das_planilhas(caminho_arquivo):
            workbook = openpyxl.load_workbook(caminho_arquivo)
        
            # Obtém os nomes das planilhas
            nomes_planilhas = workbook.sheetnames
        
            # Fecha o arquivo Excel
            workbook.close()
            
            return nomes_planilhas
        
        @st.cache_data
        def generate_forecast_table(forecast, title, periodo):
            st.write(title)
            forecast_renamed = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periodo)
            forecast_renamed = forecast_renamed.rename(columns={'ds': 'Data', 'yhat': 'Previsão média', 'yhat_lower': 'Previsão mínima', 'yhat_upper': 'Previsão máxima'})
            return forecast_renamed
        
        #pegar os nomes presentes na planilha
        nomes_planilhas = ler_nomes_das_planilhas(name)
        
        # Obtenção dos dados da interface do streamlit
        st.title("Previsão de demanda")
        graficos = nomes_planilhas
        selected_graficos = st.selectbox("Produtos:", graficos)
        ordem_filtro = st.slider("Ordem do filtro(semanas)", 1, 52)
        
        #Utilização dos dados do streamlit para a execução
        selected_product = read_sheet(selected_graficos)
        selected_product = moving_average_filter(selected_product, ordem_filtro)
        selected_product_title = selected_graficos + " - Ordem do Filtro: " + str(ordem_filtro) + " (semanas)"
        selected_product_title2 = "boxplot de " + selected_graficos + " - Ordem do Filtro: " + str(ordem_filtro) + " (semanas)"
        
        plot_historical_data(selected_product, selected_product_title)  
        
        boxplot_historical_data(selected_product, selected_product_title2)
        
        periodo = st.slider("Intervalo a ser previsto(semanas)", 1, 24)
        
        prof, forecast, df, future = create_profet_object(selected_product, periodo)
        generate_forecast_report(prof, forecast, 'Previsão de demanda')
        
        ##########
        forecast_table = generate_forecast_table(forecast, 'Previsão de Demanda - Tabela', periodo)
        st.write(forecast_table)
        #######
    
        
        generate_components_report(prof, forecast, 'Componentes', df, future)
    
        fig = prof.plot_components(forecast)
            
        # Removendo o terceiro subplot que mostra os dias da semana
        fig.delaxes(fig.axes[3])
        fig.delaxes(fig.axes[2])
        fig.delaxes(fig.axes[0])
        st.pyplot(fig)

if __name__ == "__main__":
    main()
