from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Global variables for training status and results
training_status = "Idle"
training_progress = 0  # progress percentage
result_image = None
current_ticker = None

# --------------------------------------
# Utility: Technical Indicator Functions
# --------------------------------------
def compute_macd(series, short_window=12, long_window=26):
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    return macd

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=period).mean()
    roll_down = down.rolling(window=period).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    band_width = upper_band - lower_band
    return band_width

# --------------------------------------
# Utility: News Sentiment Analysis
# --------------------------------------
def get_news_sentiment(ticker):
    ticker_obj = yf.Ticker(ticker)
    news_items = ticker_obj.news  # List of news dictionaries
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
# Utility: Get Trending Stocks Data
# --------------------------------------
def get_trending_stocks():
    trending = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL"]
    result = {}
    for symbol in trending:
        try:
            ticker_obj = yf.Ticker(symbol)
            info = ticker_obj.info
            price = info.get("regularMarketPrice", None)
            result[symbol] = price
        except Exception as e:
            result[symbol] = None
    return result

# --------------------------------------
# Set random seeds for reproducibility
# --------------------------------------
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# --------------------------------------
# Updated LSTM Model Definition with input_size=6 (Close, Volume, MA200, MACD, RSI, Bollinger)
# --------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=100, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]  # Use output from the last time step
        out = self.fc(out)
        return out

# --------------------------------------
# Main Function: run_prediction updated with epochs parameter
# --------------------------------------
def run_prediction(ticker, epochs):
    global training_status, training_progress, result_image, current_ticker
    current_ticker = ticker
    training_status = "Starting training..."
    training_progress = 0

    # Download historical data for the past 12 years
    today = datetime.today()
    start_date = f"{today.year - 12}-01-01"
    end_date = today.strftime('%Y-%m-%d')
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        training_status = "Error: No data found."
        training_progress = 100
        return

    # Compute technical indicators:
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['MACD'] = compute_macd(stock_data['Close'])
    stock_data['RSI'] = compute_rsi(stock_data['Close'])
    stock_data['Bollinger'] = compute_bollinger_bands(stock_data['Close'])
    stock_data.dropna(inplace=True)

    # Preprocess data: Use 'Close', 'Volume', 'MA200', 'MACD', 'RSI', 'Bollinger'
    data = stock_data[['Close', 'Volume', 'MA200', 'MACD', 'RSI', 'Bollinger']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    min_required = 61  # look_back (60) + 1 data point
    if len(scaled_data) < min_required:
        training_status = "Error: Not enough data to create training samples."
        training_progress = 100
        return

    # Prepare data for LSTM with a 60-day look-back window
    look_back = 60
    X = []
    y_vals = []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, :])
        y_vals.append(scaled_data[i, 0])
    X, y_vals = np.array(X), np.array(y_vals)

    # Split into training and test sets (80% training, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_vals[:train_size], y_vals[train_size:]

    # Further split training data into training and validation sets (80/20 split)
    val_size = int(len(X_train) * 0.2)
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_final, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)
    validation_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=12, pin_memory=True)

    # Build model, loss, and optimizer (input_size now 6)
    model = LSTMModel(input_size=6, hidden_size=100, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with early stopping
    total_epochs = int(epochs)  # Use the epoch value provided by the user
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch_X, val_batch_y in validation_loader:
                val_outputs = model(val_batch_X)
                loss = criterion(val_outputs, val_batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(validation_loader)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            model.load_state_dict(best_model_state)
            break
        
        training_status = f"Epoch {epoch+1}/{total_epochs} - Loss: {avg_train_loss:.6f}"
        training_progress = int(((epoch+1) / total_epochs) * 100)

    training_status = "Training complete! Generating predictions..."
    training_progress = 100

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        predicted = model(X_test_tensor).detach().numpy()
    # Inverse transform predictions
    dummy_pred = np.zeros((predicted.shape[0], 6))
    dummy_pred[:, 0] = predicted[:, 0]
    predicted_stock_price = scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_test = np.zeros((y_test.shape[0], 6))
    dummy_test[:, 0] = y_test
    y_test_actual = scaler.inverse_transform(dummy_test)[:, 0]

    # Predict future prices (30 business days ahead)
    def predict_future(scaled_data, model, look_back, scaler, future_days=30):
        last_data = scaled_data[-look_back:]
        future_predictions = []
        model.eval()
        for _ in range(future_days):
            input_data = last_data.reshape((1, look_back, 6))
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            with torch.no_grad():
                predicted_value = model(input_tensor).item()
            future_predictions.append(predicted_value)
            new_row = np.array([predicted_value, last_data[-1, 1], last_data[-1, 2],
                                last_data[-1, 3], last_data[-1, 4], last_data[-1, 5]])
            last_data = np.concatenate((last_data[1:], new_row.reshape(1, -1)), axis=0)
        dummy_future = np.zeros((len(future_predictions), 6))
        dummy_future[:, 0] = future_predictions
        future_predictions = scaler.inverse_transform(dummy_future)[:, 0]
        return future_predictions

    future_predictions = predict_future(scaled_data, model, look_back, scaler)
    news_sentiment = get_news_sentiment(ticker)
    training_status = "Prediction complete!"

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stock_data.index[train_size + look_back:], y_test_actual, color='red', label='Actual Prices')
    ax.plot(stock_data.index[train_size + look_back:], predicted_stock_price, color='green', label='Predicted Prices')
    current_date = pd.Timestamp.today()
    future_dates = pd.date_range(current_date + pd.Timedelta(days=1), periods=30, freq='B')
    ax.plot(future_dates, future_predictions, color='blue', linestyle='--', label='Predicted Future Prices')
    ax.text(0.05, 0.95, f"News Sentiment: {news_sentiment:.2f}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})
    ax.set_title(f"{ticker} Stock Price Prediction (12 Years Data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()

    result_path = os.path.join("static", "result_lstm.png")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    plt.savefig(result_path)
    plt.close()

    result_image = "result_lstm.png"

# --------------------------------------
# Flask Routes
# --------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/start_prediction', methods=['POST'])
def start_prediction():
    global training_status, training_progress
    ticker = request.form.get('ticker')
    if not ticker:
        ticker = 'CVX'
    # Retrieve the epoch value from the form; default to 50 if not provided.
    epochs = request.form.get('epochs', 50)
    training_status = "Starting training..."
    training_progress = 0
    # Start prediction in a separate thread, passing the ticker and epochs value.
    thread = threading.Thread(target=run_prediction, args=(ticker, epochs), daemon=True)
    thread.start()
    return redirect(url_for('status_page'))

@app.route('/status')
def status():
    global training_status, training_progress
    done = (training_status == "Prediction complete!")
    return jsonify({"status": training_status, "progress": training_progress, "done": done})

@app.route('/status_page')
def status_page():
    return render_template('status.html')

@app.route('/results')
def results():
    global result_image, current_ticker
    if result_image is None:
        return redirect(url_for('status_page'))
    return render_template('results.html', result_image=result_image, ticker=current_ticker)

@app.route('/trending')
def trending():
    trending_data = get_trending_stocks()
    return jsonify(trending_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
