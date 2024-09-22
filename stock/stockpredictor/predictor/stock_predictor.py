import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def calculate_metrics(df):
    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()

    # Calculate moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # Calculate volatility (standard deviation of returns)
    df['Volatility'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)

    return df

def predict_future_price(df, days_to_predict=30):
    df['Prediction'] = df['Close'].shift(-days_to_predict)
    df = df.dropna()

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA50', 'MA200', 'Volatility']
    X = df[features]
    Y = df['Prediction']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    last_30_days = df[features].tail(days_to_predict)
    prediction = model.predict(last_30_days)

    return prediction[-1]

def analyze_stock(ticker, start_date, end_date):
    df = fetch_stock_data(ticker, start_date, end_date)
    if df.empty:  # Assuming `data` is a DataFrame
        raise ValueError("No data found for the given ticker.")

    df = calculate_metrics(df)
    current_price = int(df['Close'].iloc[-1])
    future_price = round(predict_future_price(df))

    context = {
        'ticker': ticker,
        'current_price': current_price,
        'average_price': df['Close'].mean(),
        'volatility': df['Volatility'].iloc[-1],
        'returns': df['Returns'].mean() * 252,
        'ma50': df['MA50'].iloc[-1],
        'ma200': df['MA200'].iloc[-1],
        'predicted_price': future_price,
        'trend': "Bullish" if current_price > df['MA50'].iloc[-1] > df['MA200'].iloc[-1] else "Bearish",
        'prediction': "rise" if future_price > current_price else "fall",
    }
    return context
