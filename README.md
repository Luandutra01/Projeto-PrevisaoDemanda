# 📈 Demand Forecasting with Statistical and Machine Learning Models

Web application and research project focused on demand forecasting using
classical statistical models and deep learning approaches.

This project compares different time series forecasting techniques using
five years of aggregated weekly sales data.

🔗 **Live App:** https://projeto-previsaodemanda.streamlit.app/\
🔗 **Author:** Luan Dutra

------------------------------------------------------------------------

## 📌 Project Overview

This project was developed as part of an academic research initiative in
Data Science. The objective was to evaluate and compare forecasting
models for demand prediction using real historical sales data.

The study includes:

-   Classical statistical models\
-   Machine learning approaches\
-   Deep learning models\
-   Comparative evaluation using error metrics

All data used are aggregated historical records.

------------------------------------------------------------------------

## 📄 Research Paper

A complete academic paper detailing the theoretical background, modeling
decisions, experimental design, and comparative results is included in
this repository.

📎 **Full paper in portuguese available at:** `/docs/ARTIGO.pdf`

The paper presents:

-   Statistical foundations of ARIMA and SARIMA\
-   Prophet and NeuralProphet modeling approach\
-   LSTM architecture design\
-   Evaluation methodology (MAE, MAPE, RMSE)\
-   Comparative performance discussion

------------------------------------------------------------------------

## 🧠 Models Implemented

### Statistical Models

-   ARIMA\
-   SARIMA\
-   Prophet

### Machine Learning / Deep Learning

-   NeuralProphet\
-   LSTM (Long Short-Term Memory Neural Network)

------------------------------------------------------------------------

## 📊 Evaluation Metrics

-   MAE (Mean Absolute Error)\
-   MAPE (Mean Absolute Percentage Error)\
-   RMSE (Root Mean Squared Error)

------------------------------------------------------------------------

## ⚙️ Technologies Used

### Programming Language

-   Python 3.11

### Data Science & Modeling

-   numpy\
-   pandas\
-   scipy\
-   statsmodels\
-   pmdarima\
-   scikit-learn\
-   prophet\
-   neuralprophet\
-   tensorflow\
-   torch

### Visualization

-   matplotlib\
-   seaborn\
-   plotly\
-   altair

### Deployment

-   streamlit\
-   streamlit-option-menu

------------------------------------------------------------------------

## 🚀 Running the Project Locally

### 1️⃣ Clone the repository

``` bash
git clone https://github.com/Luandutra01/Projeto-PrevisaoDemanda.git
cd Projeto-PrevisaoDemanda
```

### 2️⃣ Create virtual environment (recommended)

``` bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

``` bash
streamlit run main.py
```

------------------------------------------------------------------------

## 📈 Key Contributions

-   Comparative study between statistical and deep learning forecasting
    models\
-   Implementation of LSTM neural networks for time series\
-   Full preprocessing pipeline (cleaning, feature engineering,
    scaling)\
-   Interactive dashboard for visualization and model comparison\
-   Reproducible research structure

------------------------------------------------------------------------

## 🎓 Academic Context

This project was developed as part of an undergraduate research
initiative in Computer Science, focusing on time series forecasting and
applied machine learning.

------------------------------------------------------------------------
