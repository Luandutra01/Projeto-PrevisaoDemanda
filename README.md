# 📈 Previsão de Demanda com Modelos Estatísticos e de Machine Learning

Aplicação web e projeto de pesquisa focados em previsão de demanda utilizando
modelos estatísticos clássicos e abordagens de deep learning.

Este projeto compara diferentes técnicas de previsão de séries temporais usando
cinco anos de dados agregados de vendas semanais.

🔗 **Aplicação online:** https://projeto-previsaodemanda.streamlit.app
🔗 **Autor:** Luan Dutra

---

## 📌 Visão Geral do Projeto

Este projeto foi desenvolvido como parte de uma iniciativa acadêmica em
Ciência de Dados. O objetivo foi avaliar e comparar modelos de previsão
para predição de demanda utilizando dados históricos reais de vendas.

O estudo inclui:

* Modelos estatísticos clássicos
* Abordagens de machine learning
* Modelos de deep learning
* Avaliação comparativa utilizando métricas de erro

---

## 📄 Artigo Científico

Um artigo acadêmico completo detalhando a base teórica, decisões de modelagem,
desenho experimental e resultados comparativos está incluído neste repositório.

📎 **Artigo completo em português disponível em:** `/docs/ARTIGO.pdf`

O artigo apresenta:

* Fundamentos estatísticos de ARIMA e SARIMA
* Abordagem de modelagem com Prophet e NeuralProphet
* Metodologia de avaliação (MAE, MAPE, RMSE)
* Discussão comparativa de desempenho

---

## 🧠 Modelos Implementados

### Modelos Estatísticos

* ARIMA
* SARIMA
* Prophet

### Machine Learning / Deep Learning

* NeuralProphet
* LSTM (Rede Neural de Memória de Curto e Longo Prazo)

---

## 📊 Métricas de Avaliação

* MAE (Erro Absoluto Médio)
* MAPE (Erro Percentual Absoluto Médio)
* RMSE (Raiz do Erro Quadrático Médio)

---

## ⚙️ Tecnologias Utilizadas

### Linguagem de Programação

* Python 3.11

### Ciência de Dados & Modelagem

* numpy
* pandas
* scipy
* statsmodels
* pmdarima
* scikit-learn
* prophet
* neuralprophet
* tensorflow
* torch

### Visualização

* matplotlib
* seaborn
* plotly
* altair

### Deploy

* streamlit
* streamlit-option-menu

---

## 🚀 Executando o Projeto Localmente

### 1️⃣ Clonar o repositório

```bash
git clone https://github.com/Luandutra01/Projeto-PrevisaoDemanda.git
cd Projeto-PrevisaoDemanda
```

### 2️⃣ Criar ambiente virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

### 4️⃣ Executar a aplicação

```bash
streamlit run main.py
```

---

## 📈 Principais Contribuições

* Estudo comparativo entre modelos estatísticos e de deep learning para previsão
* Implementação de redes neurais LSTM para séries temporais
* Pipeline completo de pré-processamento (limpeza, engenharia de atributos, normalização)
* Dashboard interativo para visualização e comparação de modelos
* Estrutura de pesquisa reproduzível

---

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como parte de uma iniciativa de pesquisa
de graduação em Ciência da Computação, com foco em previsão de séries
temporais e machine learning aplicado.

---
