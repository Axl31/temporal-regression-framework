# Time-Series Regressor

A lightweight and modular framework for transforming raw datasets into supervised learning windows and training regression models on multivariate time series.

## Features
- Sliding-window generation for multivariate time series
- Clean preprocessing utilities (zero filtering, datetime normalization)
- Flexible data loader supporting CSV/XLSX
- Automatic feature/target mapping
- Easy integration with scikit-learn models

## Project Structure
```
project/
├── main.py
├── utils/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── time_series.py
├── models/
│   └── regressor.py
├── data/
│   └── infy_stock.csv
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage Example

### Dataset
This example uses historical stock price data from the **National Stock Exchange of India Ltd. (NSE)**, available on Kaggle:  
**National Stock Exchange Time Series** — <https://www.kaggle.com/datasets/atulanandjha/national-stock-exchange-time-series>

**Context:**  
The NSE, established in 1992 in Mumbai, is one of India's largest electronic stock exchanges. It pioneered screen-based trading in 1994 and later introduced index futures and online trading.

**Dataset Contents:**
- Daily **Open**, **High**, **Low**, **Close**, and **Volume** values for multiple NSE-listed companies.
- Clean historical time series suitable for forecasting and modeling.

### Goal of This Example
Using this dataset, the project demonstrates how to:
1. Transform raw stock market time series into supervised learning windows.
2. Train regression models (e.g., Linear Regression, Random Forest, RNN-like structures) on the windowed data.
3. Forecast future closing prices.
4. Evaluate performance using RMSE.
5. Visualize predicted vs. actual price movements.


```python
python main.py
```

![Figure](images/Figure.jpg "Prediction vs Actual")

## Application Example

An application of this framework can be seen in the study "*[Transforming personalized weight forecasting: From the Personalized Metabolic Avatar to the Generalized Metabolic Avatar](https://doi.org/10.1016/j.compbiomed.2025.109879)*". The model supports predictive analytics similar to those used in the PMA/GMA research pipeline, enabling generalized metabolic forecasting without individual calibration.

## License
MIT License
