import streamlit as st
from utils import moving_average_filter, plot_historical_data, boxplot_historical_data

def boxplot(df, name, selected_graficos):
    st.title("Boxplot")
    with st.sidebar:
        ordem_filtro = st.slider("Ordem do filtro(semanas)", 1, 52)

    selected_product = moving_average_filter(df, ordem_filtro)
    title_line = f"{selected_graficos} - Ordem do Filtro: {ordem_filtro} (semanas)"
    title_box = f"Boxplot de {selected_graficos} - Ordem do Filtro: {ordem_filtro} (semanas)"

    col1, col2 = st.columns(2)
    with col1:
        plot_historical_data(selected_product, title_line)
    with col2:
        boxplot_historical_data(selected_product, title_box)
