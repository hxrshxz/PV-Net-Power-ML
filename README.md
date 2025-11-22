# Net Power Forecasting for PV-Integrated Residential Buildings in Delhi

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This project forecasts net power (PV generation minus residential load) for Delhi's residential sector using:

- **Photovoltaic (PV) power generation** simulation
- **Load forecasting with LSTM and XGBoost models**
- **Time-series analysis** of net power balance

## Key Features

- Accurate, Delhi-specific PV simulation using `pvlib`
- Robust forecasting via LSTM and XGBOOST
- Clear visual comparisons of actual vs predicted load
- CSV output for further analysis
- Highly accurate

## Research Paper

This project is based on the research paper:

**"Net Power Forecasting for PV-Integrated Residential Buildings in Delhi"**

📄 **Full Paper**: [`Research_Paper_ML_PV net power.pdf`](Research_Paper_ML_PV%20net%20power.pdf)

### Abstract

The paper investigates the forecasting of net power demand in residential buildings integrated with photovoltaic (PV) systems in Delhi, India. By combining solar power generation modeling with machine learning-based load forecasting, this research provides accurate predictions of the net power balance (PV generation minus consumption) to support grid planning and energy management.

### Key Contributions

- **Delhi-specific PV modeling** using meteorological data and `pvlib` simulation
- **Machine learning forecasting** with LSTM and XGBoost algorithms
- **Time-series analysis** of residential energy consumption patterns
- **Net power prediction** for optimal grid integration and energy storage planning
