import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
db_path = "../getTradeData/binance-15.db"
try:
    connection = sqlite3.connect(db_path)
    sql_query = "SELECT * FROM BTCUSDT"
    df = pd.read_sql_query(sql_query, connection)
    connection.close()
except Exception as e:
    print(f"Error loading database from {db_path}: {e}")
    # Create dummy data for structure validation if DB fails (optional, but good for script robustness)
    print("Creating dummy data for demonstration...")
    df = pd.DataFrame({
        'open': np.random.rand(1000), 'high': np.random.rand(1000), 
        'low': np.random.rand(1000), 'close': np.random.rand(1000),
        'volume': np.random.rand(1000), 'quote_asset_volume': np.random.rand(1000),
        'taker_buy_base_asset_volume': np.random.rand(1000), 'taker_buy_quote_asset_volume': np.random.rand(1000)
    })

# Convert numeric columns
numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
df[numeric_columns] = df[numeric_columns].astype(float)

# 2. Feature Engineering
# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

# EMA
ema_short = df['close'].ewm(span=12).mean()
ema_long = df['close'].ewm(span=26).mean()

# MACD
macd = ema_short - ema_long
signal_line = macd.ewm(span=9).mean()
histogram = macd - signal_line

df['rsi'] = rsi
df['ema_short'] = ema_short
df['ema_long'] = ema_long
df['macd'] = macd
df['signal_line'] = signal_line
df['histogram'] = histogram

df['sma_3'] = df['close'].rolling(window=3).mean()
df['sma_6'] = df['close'].rolling(window=6).mean()
df['sma_12'] = df['close'].rolling(window=12).mean()
df['volatility_std'] = df['close'].rolling(window=5).std()
df['pct_change'] = df['close'].pct_change()
df['sma_20'] = df['close'].rolling(window=20).mean()
df['std_20'] = df['close'].rolling(window=20).std()
df['bollinger_upper'] = df['sma_20'] + 2 * df['std_20']
df['bollinger_middle'] = df['sma_20']
df['bollinger_lower'] = df['sma_20'] - 2 * df['std_20']
df['diff_bollinger_upper'] = df['close'] - df['bollinger_upper']
df['diff_bollinger_lower'] = df['close'] - df['bollinger_lower']
df['diff_sma_3'] = df['close'] - df['sma_3']
df['diff_sma_6'] = df['close'] - df['sma_6']
df['diff_sma_12'] = df['close'] - df['sma_12']

# Drop NaNs
df = df.iloc[20:].reset_index(drop=True)

# 3. Label Generation
close = df['close'].tolist()

def find_extrema_indices_no_slip(close, window_size):
    max_indices = []
    min_indices = []
    for i in range(0, len(close) - window_size + 1, window_size):
        window = close[i:i+window_size]
        max_value = max(window)
        min_value = min(window)
        max_index = window.index(max_value) + i
        min_index = window.index(min_value) + i
        if max_index != i and max_index != i + window_size - 1 and max_index not in max_indices:
            max_indices.append(max_index)
        if min_index != i and min_index != i + window_size - 1 and min_index not in min_indices:
            min_indices.append(min_index)
    return max_indices, min_indices

min_indices, max_indices = find_extrema_indices_no_slip(close, 300)

def pair_max_min(max_indices, min_indices):
    paired_max_indices = []
    paired_min_indices = []
    all_indices = sorted(max_indices + min_indices)
    for i in range(len(all_indices) - 1):
        if all_indices[i] in max_indices and all_indices[i+1] in min_indices:
            paired_max_indices.append(all_indices[i])
            paired_min_indices.append(all_indices[i+1])
    return paired_max_indices, paired_min_indices

buy_indices, sell_indices = pair_max_min(max_indices, min_indices)

# Signal: 
# 1 - Hold
# 2 - Buy (Derived from Fixed Interval Extrema)
# 3 - Sell (Derived from Fixed Interval Extrema)
buy_sell_signal = [1] * len(close)
for buy in buy_indices:
    buy_sell_signal[buy] = 2
for sell in sell_indices:
    buy_sell_signal[sell] = 3

df['signal'] = buy_sell_signal

# 4. Scaling
scaler = MinMaxScaler()
# Note: transforming 'signal' with MinMaxScaler might not be ideal if it's categorical (1,2,3), 
# but following the notebook (it scales the whole df).
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 5. Split Data
def split_data(data, lookback, train_ratio=0.9):
    n = len(data)
    train_size = int(n * train_ratio)
    
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    X_train = np.array([train_data.iloc[i:i+lookback].values for i in range(len(train_data) - lookback)])
    # Note: notebook selected [['high', 'low', 'signal']] for y, effectively predicting future signal based on past
    # The notebook code was:
    # y_train = np.array([train_data.iloc[i+lookback][['high', 'low', 'signal']] for i in range(len(train_data) - lookback)])
    # This implies y is taken at step i+lookback.
    
    # We need to ensure we select columns by name correctly after scaling where column names are preserved
    target_cols = ['high', 'low', 'signal']
    target_indices = [df.columns.get_loc(c) for c in target_cols] # indices in scaled df
    
    y_train = np.array([train_data.iloc[i+lookback, target_indices].values for i in range(len(train_data) - lookback)])
    
    X_test = np.array([test_data.iloc[i:i+lookback].values for i in range(len(test_data) - lookback)])
    y_test = np.array([test_data.iloc[i+lookback, target_indices].values for i in range(len(test_data) - lookback)])
    
    return X_train, y_train, X_test, y_test

lookback = 211
X_train, y_train, X_test, y_test = split_data(df_scaled, lookback)

# 6. Random Forest Training
print("Reshaping data for Random Forest...")
# Flatten time dimension: (n_samples, lookback, n_features) -> (n_samples, lookback * n_features)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Prepare Labels
# signal is the 3rd column in y (index 2)
# Inverse transform if we want original 1, 2, 3? 
# Or just use the scaled values? Random Forest Classifier needs integer labels.
# Since 'signal' was scaled, it is no longer 1,2,3 integers.
# We must recover the class labels.
# Option A: Re-access unscaled DF for labels.
# Option B: Inverse transform the y part. (Cleaner)

# Let's get labels from unscaled DF for simplicity and correctness.
# Re-split helper for labels only.
def get_labels(data, lookback, train_ratio=0.9):
    n = len(data)
    train_size = int(n * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    y_train_sig = np.array([train_data.iloc[i+lookback]['signal'] for i in range(len(train_data) - lookback)])
    y_test_sig = np.array([test_data.iloc[i+lookback]['signal'] for i in range(len(test_data) - lookback)])
    return y_train_sig, y_test_sig

y_train_labels, y_test_labels = get_labels(df, lookback)
# Ensure they are integers
y_train_labels = y_train_labels.astype(int)
y_test_labels = y_test_labels.astype(int)

print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_reshaped, y_train_labels)

print("Predicting...")
y_pred = rf_model.predict(X_test_reshaped)

# 7. Evaluation
print("Evaluation Results:")
print("Accuracy:", accuracy_score(y_test_labels, y_pred))
print("Classification Report:\n", classification_report(y_test_labels, y_pred))

# Optional: Save model
# import joblib
# joblib.dump(rf_model, 'random_forest_model.pkl')
