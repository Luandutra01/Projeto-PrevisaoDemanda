import streamlit as st
from prophet import Prophet
from utils import moving_average_filter, plot_historical_data, generate_forecast_table, download_link_excel

def previsaoProphet(df, name, selected_graficos):
    st.title('Previsão de demanda')
    with st.sidebar:
        ordem_filtro = st.slider("Ordem do filtro (semanas)", 1, 52)
        periodo = st.slider("Intervalo a ser previsto(semanas)", 1, 24, 12)

    selected_product = moving_average_filter(df, ordem_filtro)
    title = f"{selected_graficos} - Ordem do Filtro: {ordem_filtro} (semanas)"

    col1, col2 = st.columns(2)
    with col1:
        plot_historical_data(selected_product, title)

        prof = Prophet()
        df_fit = selected_product.rename(columns={'DATA': 'ds', 'QUANT': 'y'})
        prof.fit(df_fit)
        future = prof.make_future_dataframe(periods=periodo, freq='W')
        forecast = prof.predict(future)

        st.pyplot(prof.plot(forecast))
        forecast_table = generate_forecast_table(forecast, periodo)
        st.write(forecast_table)
        st.markdown(download_link_excel(forecast_table, 'forecast'), unsafe_allow_html=True)

    with col2:
        st.pyplot(prof.plot_components(forecast))
