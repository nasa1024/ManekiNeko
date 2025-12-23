# ManekiNeko（招财猫）
> this project is used to explore the possibility of lying to earn money in crypto currency  

[English](README.md) | [中文](README_zh.md)

including following parts:
- [x] `getTradeData` get trade data from binance and save to sqlite
- [x] `lstmTrain` using lstm to predict the price of crypto currency
- [x] `xGBoost` using XGBoost to predict the price of crypto currency
- [x] `randomForest` using random forest to predict the price of crypto currency
- [ ] `transformer` using Transformer to predict the price of crypto currency
- [ ] `informer` using Informer to predict the price of crypto currency
- [ ] `LightGBM / CatBoost` efficient gradient boosting variants
- [ ] `N-BEATS` deep neural architecture for interpretable time series forecasting
- [ ] `Reinforcement Learning (PPO/DQN)` training agents to make trading decisions directly

> **[Implementation Roadmap](docs/model_implementation_roadmap.md)**: Detailed plan for implementing the above advanced models.
<br>

# Feature Engineering & Signal Generation

## Technical Indicators
The project generates extensive technical indicators to capture market dynamics:
- **Momentum**: RSI (14-period)
- **Trend**: EMA (12/26), MACD (Signal, Histogram), SMA (3, 6, 12, 20)
- **Volatility**: Bollinger Bands (Upper, Middle, Lower), Volatility Std (5-period)
- **Price Differentials**: Distance from SMA and Bollinger Bands

## Signal Labeling Logic
The model treats trading as a multi-class classification problem with signals derived from local extrema detection:

### 1. Fixed Interval Extrema (Primary Signals)
Data is divided into non-overlapping windows (default size: 300).
- **Label 4 (Buy)**: Local Minimum (Entry point)
- **Label 6 (Sell)**: Local Maximum (Exit point)
*Note: The algorithm pairs minima with subsequent maxima to ensure valid trade cycles.*

### 2. Sliding Window Extrema ("Slip" Signals)
A rolling window scans the time series to capture more granular market turning points.
- **Label 2 (Slip Buy)**: Local Minimum in sliding window
- **Label 3 (Slip Sell)**: Local Maximum in sliding window

### 3. Default State
- **Label 1 (Hold)**: No specific signal detected

# `getTradeData` usage
copy `config_example.ini` to config.ini and change the api key and secret key

```bash
pip install requirements.txt
python getTradeData.py
```


# license
MIT

