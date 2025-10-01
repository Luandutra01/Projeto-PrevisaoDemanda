#1034 linhas total
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

from utils import (
    ler_nomes_das_planilhas,
    read_sheet,
    remove_outliers_iqr_by_year,
    _iqr_bounds
)

# Importa as telas
from telas.boxplot import boxplot
from telas.previsao_prophet import previsaoProphet
from telas.analise_prophet import analiseProphet
from telas.previsao_neural import previsaoNeural
from telas.analise_neural import analiseNeural
from telas.previsao_arima import previsaoArima
from telas.analise_arima import analiseArima

st.set_page_config(layout="wide")

def run_main_program():
    st.sidebar.title("📊 Configuração dos Dados")
    uploaded_file = st.sidebar.file_uploader("Arraste o Excel aqui", type=["xlsx"])

    if uploaded_file:
        # Listar abas do Excel
        planilhas = ler_nomes_das_planilhas(uploaded_file)
        sheet_name = st.sidebar.selectbox("Escolha a planilha", planilhas)

        # Carregar dados
        df = read_sheet(uploaded_file, sheet_name)


        #####################
        # Sliders de corte de início/fim (você já tinha)
        remove_start = st.slider("Apagar dados do início (%)", 0, 100, 0)
        remove_end   = st.slider("Apagar dados do fim (%)",   0, 100, 0)
            
        rows_to_remove_start = int(len(df) * (remove_start / 100))
        rows_to_remove_end   = int(len(df) * (remove_end   / 100))
            
        if rows_to_remove_start + rows_to_remove_end >= len(df):
            st.warning("A remoção total excede o número de linhas disponíveis. Ajuste os sliders.")
            df_filtered = pd.DataFrame()
        else:
            df_filtered = df.iloc[rows_to_remove_start: len(df) - rows_to_remove_end].copy()
            
        # === Checkbox e controles do filtro de outliers ===
        with st.sidebar:
            rm_out = st.checkbox("Remover outliers (IQR)", value=False)
            if rm_out and not df_filtered.empty:
                # lista de colunas numéricas
                num_cols = [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])]
                # tenta escolher QUANTIDADE por padrão, se existir
                default_idx = 0
                if "QUANTIDADE" in num_cols:
                    default_idx = num_cols.index("QUANTIDADE")
                        
                #col_iqr = st.selectbox("Coluna para IQR:", options=num_cols, index=default_idx)
                col_iqr = "QUANT"
                    
                
                k = st.slider("Fator k do IQR", 1.0, 3.0, 1.5, 0.1)
            
        
            if 'rm_out' in locals() and rm_out and not df_filtered.empty:
                df_filtered = remove_outliers_iqr_by_year(df_filtered, value_col=col_iqr, date_col="DATA", k=k)


        st.sidebar.success(f"✔ Planilha '{sheet_name}' carregada com {len(df_filtered)} linhas.")

        selected_graficos = f"Tabela: {sheet_name}"
        
        with st.sidebar:
            selecao = option_menu(
                "Menu",
                ["Boxplot", "Previsão Prophet", "Análise Prophet", "Previsão NeuralProphet", "Análise NeuralProphet", "Previsão Arima", "Análise Arima"],
                icons=['box', 'graph-up', 'bar-chart-line', 'graph-up', 'bar-chart-line', 'graph-up', 'bar-chart-line'],
                menu_icon="cast",
                default_index=0,
            )
        if selecao == 'Boxplot':
            boxplot(df_filtered, sheet_name, selected_graficos)
        elif selecao == 'Previsão Prophet':
            previsaoProphet(df_filtered, sheet_name, selected_graficos)
        elif selecao == 'Análise Prophet':
            analiseProphet(df_filtered, sheet_name, selected_graficos)
        elif selecao == 'Previsão NeuralProphet':
            previsaoNeural(df_filtered, sheet_name, selected_graficos)
        elif selecao == 'Análise NeuralProphet':
            analiseNeural(df_filtered, sheet_name, selected_graficos)
        elif selecao == 'Previsão Arima':
            previsaoArima(df_filtered, sheet_name, selected_graficos)
        elif selecao == 'Análise Arima':
            analiseArima(df_filtered, sheet_name, selected_graficos)

    else:
        st.info("📥 Faça upload de um arquivo Excel para continuar.")


if __name__ == "__main__":
    run_main_program()
