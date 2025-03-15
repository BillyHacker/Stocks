import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for plotting
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import xgboost as xgb

# --------------------------------------
# Extended Technical Indicator Functions
# --------------------------------------
def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, macd_signal

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def compute_stochastic_oscillator(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    stoch_d = stoch_k.rolling(d_window).mean()
    return stoch_k, stoch_d

def compute_volume_feature(close, volume, window=20):
    cv = close * volume
    cv_roll = cv.rolling(window).sum()
    vol_roll = volume.rolling(window).sum()
    vwap = cv_roll / (vol_roll + 1e-9)
    return vwap

# --------------------------------------
# Utility: News Sentiment Analysis
# --------------------------------------
def get_news_sentiment(ticker):
    ticker_obj = yf.Ticker(ticker)
    news_items = ticker_obj.news  # List of news dictionaries
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    if news_items:
        for item in news_items:
            title = item.get('title', '')
            sentiment = sia.polarity_scores(title)
            sentiments.append(sentiment['compound'])
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    else:
        avg_sentiment = 0
    return avg_sentiment

# --------------------------------------
# XGBoost Prediction Function with Extended Indicators and Rolling Forecast
# --------------------------------------
def run_prediction_xgboost(ticker):
    """
    Downloads 12 years of data for the ticker, computes extended technical indicators,
    creates sliding window features (60 days flattened to 600 features), trains an XGBoost model,
    performs a rolling forecast for 30 future business days (with indicator recalculation),
    plots actual test prices, predicted test prices, and future predictions, and saves the plot.
    
    Returns:
        result_image (str): Filename of the saved plot (e.g., "result_xgb.png")
    """
    # A. Download historical data
    today = datetime.today()
    start_date = f"{today.year - 12}-01-01"
    end_date = today.strftime('%Y-%m-%d')
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        raise ValueError("No data found for ticker: " + ticker)
    
    # B. Compute Extended Indicators
    stock_data['MACD'], stock_data['MACD_signal'] = compute_macd(stock_data['Close'])
    stock_data['RSI'] = compute_rsi(stock_data['Close'])
    stock_data['SMA'], stock_data['BOLL_UP'], stock_data['BOLL_DOWN'] = compute_bollinger_bands(stock_data['Close'])
    if 'High' not in stock_data.columns or 'Low' not in stock_data.columns:
        stock_data['High'] = stock_data['Close']
        stock_data['Low'] = stock_data['Close']
    stock_data['STOCH_K'], stock_data['STOCH_D'] = compute_stochastic_oscillator(
        stock_data['High'], stock_data['Low'], stock_data['Close']
    )
    if 'Volume' not in stock_data:
        stock_data['Volume'] = 1
    stock_data['VWAP'] = compute_volume_feature(stock_data['Close'], stock_data['Volume'])
    stock_data.dropna(inplace=True)
    
    # C. Prepare Feature Set
    features = ['Close', 'MACD', 'MACD_signal', 'RSI', 'SMA',
                'BOLL_UP', 'BOLL_DOWN', 'STOCH_K', 'STOCH_D', 'VWAP']
    data = stock_data[features]
    
    # D. Scale Features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    look_back = 60
    if len(scaled_data) < (look_back + 1):
        raise ValueError("Not enough data after computing indicators.")
    
    # E. Create Sliding Window Features
    X = []
    y_vals = []
    for i in range(look_back, len(scaled_data)):
        window_data = scaled_data[i - look_back:i, :]
        X.append(window_data.flatten())
        y_vals.append(scaled_data[i, 0])
    X, y_vals = np.array(X), np.array(y_vals)
    
    # F. Split Data into Train/Test (80/20)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_vals[:split_index], y_vals[split_index:]
    
    # G. Train XGBoost Model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 7,
        'eta': 0.005,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    num_rounds = 300
    model_xgb = xgb.train(params, dtrain, num_rounds)
    
    # H. Test Set Predictions
    y_pred = model_xgb.predict(dtest)
    y_pred_10d = np.zeros((len(y_pred), len(features)))
    y_pred_10d[:, 0] = y_pred
    y_pred_inv = scaler.inverse_transform(y_pred_10d)[:, 0]
    y_test_10d = np.zeros((len(y_test), len(features)))
    y_test_10d[:, 0] = y_test
    y_test_inv = scaler.inverse_transform(y_test_10d)[:, 0]
    
    # I. Rolling Forecast for Future Predictions (30 business days)
    last_data = scaled_data[-look_back:].copy()
    forecast_df = stock_data.copy()
    last_index = forecast_df.index[-1]
    future_days = 30
    future_preds = []
    for day in range(future_days):
        sample = last_data.flatten().reshape(1, -1)
        dsample = xgb.DMatrix(sample)
        pred_scaled_close = model_xgb.predict(dsample)[0]
        row_10d = last_data[-1].copy()
        row_10d[0] = pred_scaled_close
        row_inv = scaler.inverse_transform(row_10d.reshape(1, -1))[0]
        row_dict = {feat: row_inv[i] for i, feat in enumerate(features)}
        new_index = last_index + pd.Timedelta(days=1)
        while new_index.weekday() >= 5:
            new_index += pd.Timedelta(days=1)
        row_series = pd.Series(row_dict, name=new_index)
        forecast_df = pd.concat([forecast_df, row_series.to_frame().T])
        last_index = new_index
        future_preds.append(row_inv[0])
        # Recompute indicators on the last 60 rows
        last_60 = forecast_df.iloc[-look_back:].copy()
        last_60['MACD'], last_60['MACD_signal'] = compute_macd(last_60['Close'])
        last_60['RSI'] = compute_rsi(last_60['Close'])
        last_60['SMA'], last_60['BOLL_UP'], last_60['BOLL_DOWN'] = compute_bollinger_bands(last_60['Close'])
        if 'High' not in last_60.columns:
            last_60['High'] = last_60['Close']
        if 'Low' not in last_60.columns:
            last_60['Low'] = last_60['Close']
        last_60['STOCH_K'], last_60['STOCH_D'] = compute_stochastic_oscillator(
            last_60['High'], last_60['Low'], last_60['Close']
        )
        if 'Volume' in forecast_df.columns:
            last_60['VWAP'] = compute_volume_feature(last_60['Close'], last_60['Volume'])
        last_60 = last_60.ffill()
        scaled_60 = scaler.transform(last_60[features].to_numpy())
        last_data = scaled_60
    future_predictions = future_preds
    
    # J. Get News Sentiment
    news_sentiment = get_news_sentiment(ticker)
    
    # K. Plot the Results
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(10, 6))
    data_idx = stock_data.index[look_back:]
    dates_test = data_idx[split_index:]
    ax.plot(dates_test, y_test_inv, color='red', label='Actual Prices')
    ax.plot(dates_test, y_pred_inv, color='green', label='Predicted Prices')
    future_dates = forecast_df.index[-future_days:]
    ax.plot(future_dates, future_predictions, color='blue', linestyle='--', label='Predicted Future Prices')
    ax.text(0.05, 0.95, f"News Sentiment: {news_sentiment:.2f}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})
    ax.set_title(f"{ticker} Price Prediction (Extended Indicators) - XGBoost")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    result_path = os.path.join("static", "result_xgb.png")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    plt.savefig(result_path)
    plt.close()
    
    print("Prediction complete. Plot saved as result_xgb.png")
    training_progress = 100
    return "result_xgb.png"

# --------------------------------------
# LightGBM Prediction Function (Simplified Demo)
# --------------------------------------
def run_prediction_lightgbm(ticker):
    import time
    time.sleep(2)
    return "result_lgb.png"

# --------------------------------------
# LSTM Prediction Function (Simplified Demo)
# --------------------------------------
def run_prediction_lstm(ticker):
    # Placeholder for LSTM prediction logic
    import time
    print("Starting LSTM prediction for", ticker)
    time.sleep(2)
    print("LSTM prediction complete.")
    return "result_lstm.png"
