import streamlit as st
import pandas as pd
import openpyxl
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from prophet import Prophet
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from neuralprophet import NeuralProphet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.dates as mdates

### ====== LEITURA DE DADOS ======
@st.cache_data
def ler_nomes_das_planilhas(caminho_arquivo):
    """Retorna a lista de nomes das abas do Excel"""
    workbook = openpyxl.load_workbook(caminho_arquivo)
    nomes_planilhas = workbook.sheetnames
    workbook.close()
    return nomes_planilhas
    
@st.cache_data
def read_sheet(name, sheet):
    """Lê a planilha Excel e formata colunas/datas"""
    df = pd.read_excel(name, sheet, header=0)
    df = df.rename(columns={'QUANTIDADE': 'QUANT'})
    df = df.rename(columns={'DataInicioSemana': 'DATA'})

    # Extrair ano/mes/dia
    df['Year'] = (df['DATA']//10000)
    df['Month'] = df['DATA'] - (df['Year'] * 10000)
    df['Month'] = df['Month'] // 100
    df['Day'] = df['DATA'] - (df['Year'] * 10000 + df['Month'] * 100)
    df.loc[df['Day'] < 1,'Day'] = 1

    df['AnoSemanaIdx'] = df['Year'] * 100 + df['NumeroSemana']
    df = df.set_index('AnoSemanaIdx')

    # Agrupamento para semanas repetidas
    df = df.groupby(df.index).agg({
        'NumeroSemana': 'first',
        'DATA': 'first',
        'QUANT': 'sum',
        'Year': 'first',
        'Month': 'first',
        'Day': 'first'
    })

    df['DATA'] = pd.to_datetime(df[['Year','Month','Day']])

    # Corrige semana 53
    fixedLastWeek = df.loc[df['NumeroSemana']==52, 'QUANT'].reset_index(drop=True) + \
                    df.loc[df['NumeroSemana']==53]['QUANT'].reset_index(drop=True)
    dfFixedLastWeek = fixedLastWeek.to_frame()
    dfFixedLastWeek['anosemana'] = df[df['NumeroSemana']==52].index
    dfFixedLastWeek.set_index('anosemana', inplace=True)
    df.loc[df['NumeroSemana']==52,'QUANT'] = dfFixedLastWeek
    df = df.drop(df[df['NumeroSemana']==53].index)

    return df

### ====== OUTLIERS ======
def _iqr_bounds(s, k=1.5):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k*iqr, q3 + k*iqr

def remove_outliers_iqr_by_year(df, value_col, date_col="DATA", k=1.5):
    """IQR por ano (combina melhor com o boxplot anual)."""
    if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    def _f(g):
        s = pd.to_numeric(g[value_col], errors="coerce")
        lower, upper = _iqr_bounds(s, k)
        return g[(s >= lower) & (s <= upper)]
    return df.groupby(df[date_col].dt.year if date_col in df.columns else 0, group_keys=False).apply(_f)

### ====== FILTRO MÓVEL ======
@st.cache_data
def moving_average_filter(df, order):
    filtred_df = df.copy()
    filtred_df['QUANT'] = filtred_df['QUANT'].rolling(order).mean()
    return filtred_df

### ====== PLOTS ======
def plot_historical_data(df, title):
    st.write(title)
    chart = alt.Chart(df.reset_index()).mark_line().encode(
        x='DATA',
        y='QUANT'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def boxplot_historical_data(df, title):
    st.write(title)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='Year', y='QUANT', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Ano')
    ax.set_ylabel('Quantidade')
    plt.xticks(rotation=45)
    st.pyplot(fig)

### ====== PROPHET ======
@st.cache_data
def create_profet_object(df, periodo):
    prof = Prophet() 
    prof.add_country_holidays(country_name='BR')
    df = df.rename(columns={'DATA': 'ds', 'QUANT': 'y'})
    prof.fit(df)
    future = prof.make_future_dataframe(periods=periodo, freq='W')
    #future.tail()

    forecast = prof.predict(future)
    #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    return (prof, forecast, df, future)

### ====== Neural ======
@st.cache_data
def create_neural_object(df, periodo):
    df = df.rename(columns={'DATA': 'ds', 'QUANT': 'y'})
    duplicate_ds = df[df.duplicated(subset=['ds'], keep=False)]
    #if not duplicate_ds.empty:
        #st.write("Duplicatas encontradas na coluna 'ds', só a primeira foi mantida:")
        #st.write(duplicate_ds)
    df = df.drop_duplicates(subset=['ds'], keep='first')
    # Seleção das colunas 'ds' e 'y'
    df = df.loc[:, ['ds', 'y']]
    #df = df.drop(columns=['AnoSemanaIdx'])

    # Criação do objeto NeuralProphet
    prof = NeuralProphet()
    
    # Ajuste do modelo com os dados históricos
    prof.fit(df)

    #future = prof.make_future_dataframe(periods=periodo)
    future = prof.make_future_dataframe(df=df, periods=periodo)
    
    #future = pd.DataFrame({
    #    'ds': pd.date_range(start=df['ds'].max(), periods=periodo, freq='W')
    #})
    
    # Previsão dos valores futuros
    forecast = prof.predict(future)

    ###########
    # Configurações para plotagem
    fig, ax = plt.subplots()
    ax.plot(df['ds'], df['y'], label='Histórico')
    ax.plot(forecast['ds'], forecast['yhat1'], label='Previsão', linestyle='--')

    # Formatação do eixo x para exibir datas
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Adiciona legenda e rótulos aos eixos
    ax.legend()
    ax.set_title('Previsão com NeuralProphet')
    ax.set_xlabel('Data')
    ax.set_ylabel('Valor')
    ax.grid(True)

    # Exibe o gráfico na interface do Streamlit
    #st.pyplot(fig)
    #############
    
    return (prof, forecast, df, future, fig)

### ====== ARIMA ======

@st.cache_data
def create_arima_object(df, periodo, p, d, q):
    df = df.rename(columns={'DATA': 'ds', 'QUANT': 'y'})
    duplicate_ds = df[df.duplicated(subset=['ds'], keep=False)]
    #if not duplicate_ds.empty:
        #st.write("Duplicatas encontradas na coluna 'ds', só a primeira foi mantida:")
        #st.write(duplicate_ds)
    df = df.drop_duplicates(subset=['ds'], keep='first')
    # Seleção das colunas 'ds' e 'y'
    df = df.loc[:, ['ds', 'y']]
    #df = df.drop(columns=['AnoSemanaIdx'])


    # Ajusta o modelo ARIMA
    prof = ARIMA(df['y'], order=(p, d, q))  # Componente MA definido como 0
    prof_fit = prof.fit()

    # Cria um DataFrame com datas futuras para previsão
    future_dates = pd.date_range(df['ds'].iloc[-1] + pd.Timedelta(weeks=1), periods=periodo, freq='W')
    forecast = prof_fit.forecast(steps=periodo)
    future = pd.DataFrame({'ds': future_dates, 'y': forecast})


    # Configurações para plotagem
    fig, ax = plt.subplots()
    ax.plot(df['ds'], df['y'], label='Histórico')
    ax.plot(future['ds'], future['y'], label='Previsão', linestyle='--')

    # Formatação do eixo x para exibir datas
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Adiciona legenda e rótulos aos eixos
    ax.legend()
    ax.set_title('Previsão com ARIMA')
    ax.set_xlabel('Data')
    ax.set_ylabel('Valor')
    ax.grid(True)

    return prof, forecast, df, future, fig, future_dates

### ====== SARIMA ======

@st.cache_data
def create_sarima_object(df, periodo, p, d, q, P, D, Q, s):
    df = df.rename(columns={'DATA': 'ds', 'QUANT': 'y'})
    duplicate_ds = df[df.duplicated(subset=['ds'], keep=False)]
    df = df.drop_duplicates(subset=['ds'], keep='first')
    
    # Seleção das colunas 'ds' e 'y'
    df = df.loc[:, ['ds', 'y']]

    # Ajusta o modelo SARIMA
    sarima = SARIMAX(df['y'], order=(p, d, q), seasonal_order=(P, D, Q, s))
    sarima_fit = sarima.fit(disp=False)

    # Cria datas futuras
    future_dates = pd.date_range(df['ds'].iloc[-1] + pd.Timedelta(weeks=1), periods=periodo, freq='W')
    forecast = sarima_fit.forecast(steps=periodo)
    future = pd.DataFrame({'ds': future_dates, 'y': forecast})

    # Configurações para plotagem
    fig, ax = plt.subplots()
    ax.plot(df['ds'], df['y'], label='Histórico')
    ax.plot(future['ds'], future['y'], label='Previsão', linestyle='--')

    # Formatação do eixo x para exibir anos
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Adiciona legenda e rótulos
    ax.legend()
    ax.set_title('Previsão com SARIMA')
    ax.set_xlabel('Data')
    ax.set_ylabel('Valor')
    ax.grid(True)

    return sarima, forecast, df, future, fig, future_dates

### ====== FORECAST TABLE ======
def generate_forecast_table(forecast, periodo):
    forecast_renamed = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periodo)
    forecast_renamed = forecast_renamed.rename(columns={
        'ds': 'Data',
        'yhat': 'Previsão média',
        'yhat_lower': 'Previsão mínima',
        'yhat_upper': 'Previsão máxima'
    })
    return forecast_renamed

@st.cache_data
def generate_forecast_report(_prof, forecast, title):
    st.write(title)
    fig1 = _prof.plot(forecast)
    st.pyplot(fig1)

@st.cache_data
def generate_forecast_report2(_prof, forecast, title, tamanhoPrevisao):
    #tamanhoPrevisao += 1
    st.write(title)
    # Filtra as últimas num_semanas semanas
    forecast_filtered = forecast.tail(tamanhoPrevisao)
    fig1 = _prof.plot(forecast_filtered)
    st.pyplot(fig1)
    
    
@st.cache_data
def generate_components_report(_prof, forecast, title, df, future):
    _prof = Prophet(weekly_seasonality=False)
    _prof.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    forecast = _prof.fit(df).predict(future)
    fig2 = _prof.plot_components(forecast)
    st.pyplot(fig2)

### ====== DOWNLOAD ======
def download_link_excel(df, filename):
    """Gera link para download de um DataFrame como Excel"""
    temp_file = f"{filename}.xlsx"
    df.to_excel(temp_file, index=False)

    with open(temp_file, 'rb') as f:
        excel_data = f.read()

    import os
    os.remove(temp_file)

    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Baixar tabela para o excel</a>'
    return href

### ======= METRICAS ====== 

@st.cache_data
def calcular_porcentagem_entre_min_max(prev_min, prev_max, test_data):
    total = len(test_data)
    dentro_intervalo = ((test_data >= prev_min) & (test_data <= prev_max)).sum()
    percentual = dentro_intervalo / total * 100
    return percentual


def calcular_erros(test_data, forecast_train, st, option2):
    col1, col2, col3 = st.columns(3)
    with col1:
        # Calcular o Erro Médio Absoluto (MAE)
        mae = mean_absolute_error(test_data['QUANT'], forecast_train)
        st.write("Erro Médio Absoluto (MAE):", int(mae))
    
    APE = mae / test_data['QUANT'] * 100
    
    # Calcular o Erro Percentual Absoluto Médio (MAPE)

    MAPE = APE.mean()
    with col2: 
        st.write("Erro Percentual Absoluto Médio (MAPE):", round(MAPE, 2), "%")
        #st.plotly_chart(fig)
    with col3:
        # Calcular o Root Mean Square Error (RMSE)
        RMSE = np.sqrt(mean_squared_error(test_data['QUANT'], forecast_train))
        st.write("Raiz do Erro Quadrático Médio (RMSE):", int(RMSE))

    
    # Calcular os erros
    errors = test_data['QUANT'] - forecast_train

    if option2:
        st.write("Diferenças entre os valores e as previsões:")
        st.write(errors)

    col1, col2 = st.columns(2)
      
    # Plotar o histograma dos erros
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=6, color='skyblue', edgecolor='black')
    plt.title('Histograma dos Erros')
    plt.xlabel('Erro')
    plt.ylabel('Frequência')
    plt.grid(True)
    with col1: 
        st.pyplot(plt)
    
    # Erros em porcentagem
    errors_percent = (test_data['QUANT'] - forecast_train) / test_data['QUANT'] * 100
    if option2:
        st.write("Diferenças entre os valores e as previsões em porcentagem")
        st.write(errors_percent)
    
    # Plotar o histograma dos erros percentuais
    plt.figure(figsize=(8, 6))
    plt.hist(errors_percent, bins=6, color='skyblue', edgecolor='black')
    plt.title('Histograma dos Erros percentuais')
    plt.xlabel('Erro %')
    plt.ylabel('Frequência')
    plt.grid(True)

    with col2: 
        st.pyplot(plt)

@st.cache_data
def train_test_split(df):
    # Ordena o DataFrame pela data
    df_sorted = df.sort_values(by='DATA')
    # Calcula o índice para separar os dados de treinamento e teste
    split_index = int(len(df_sorted) * 0.95)  # porcentagem para treinamento 0.9 = 90%
    
    tamanhoPrevisao = int(len(df_sorted) * 0.05)
    
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]
    
    return train_df, test_df, tamanhoPrevisao

@st.cache_data
def get_model_aic_bic(df, p, d, q):
    """Retorna o AIC e BIC de um ARIMA ajustado."""
    df = df.rename(columns={'DATA': 'ds', 'QUANT': 'y'})
    model = ARIMA(df['y'], order=(p, d, q))
    model_fit = model.fit()
    return model_fit.aic, model_fit.bic


@st.cache_data
def get_sarima_aic_bic(df, p, d, q, P, D, Q, m):
    """Retorna o AIC e BIC de um SARIMA ajustado."""
    df = df.rename(columns={'DATA': 'ds', 'QUANT': 'y'})
    model = SARIMAX(df['y'], order=(p, d, q), seasonal_order=(P, D, Q, m))
    model_fit = model.fit(disp=False)
    return model_fit.aic, model_fit.bic