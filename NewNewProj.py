import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
from flask import Flask, render_template

app = Flask(__name__)

# Major Currencies with Yahoo Finance Symbols
currencies = ['EURUSD=X', 'JPY=X', 'GBPUSD=X', 'CHF=X', 'CNY=X', 'CAD=X', 'NZD=X', 'KRW=X', 'INR=X']
currency_names = {
    'EURUSD=X': 'Euro',
    'JPY=X': 'Japanese Yen',
    'GBPUSD=X': 'British Pound Sterling',
    'CHF=X': 'Swiss Franc',
    'CNY=X': 'Chinese Yuan',
    'CAD=X': 'Canadian Dollar',
    'NZD=X': 'New Zealand Dollar',
    'KRW=X': 'South Korean Won',
    'INR=X': 'Indian Rupee'
}

# Store current exchange rates in a dictionary
exchange_rates = {}

# Fetch and print exchange rates for each currency
for currency in currencies:
    try:
        currency_data = yf.Ticker(currency)
        current_rate = currency_data.history(period='1d')['Close'].iloc[0]
        exchange_rates[currency] = current_rate
        print(f"Current USD to {currency} rate: {current_rate}")
    except Exception as e:
        print(f"Error fetching data for {currency}: {e}")

# Store historical exchange rate data in dictionaries
dfs = {}

# Fetch data for each currency
for currency in currencies:
    try:
        df = yf.download(currency, start='2022-01-01', end='2023-11-14')
        dfs[currency] = df
        print(f"Fetched data for {currency}")
    except Exception as e:
        print(f"Error fetching data for {currency}: {e}")

# Data Cleaning
for currency, df in dfs.items():
    # Convert the index to datetime if it's not already
    df.index = pd.to_datetime(df.index)

    # Sort the data by date
    df.sort_index(inplace=True)

    # Handling missing values
    # Forward fill
    df.fillna(method='ffill', inplace=True)

    # Checking for duplicates and removing them
    df = df[~df.index.duplicated(keep='first')]

    # Update the dictionary with the cleaned dataframe
    dfs[currency] = df

for currency, df in dfs.items():
    print(f"Data for {currency}:")
    print("Data Types:\n", df.dtypes)
    print("Missing Values:\n", df.isnull().sum())
    print("Data Range: ", df.index.min(), "to", df.index.max())
    print("Sample Data:\n", df.head())

# Enhancing the static time series plots
plt.figure(figsize=(15, 10))
for currency, df in dfs.items():
    plt.plot(df.index, df['Close'], label=currency_names[currency])

plt.title('Exchange Rate Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.xticks(rotation=45)
plt.grid(True)  # Adding gridlines for better readability
plt.legend()
plt.tight_layout()
plt.show()

# Enhancing interactive line charts
for currency, df in dfs.items():
    fig = px.line(df, x=df.index, y='Close', title=f'{currency_names[currency]} Exchange Rate Over Time')
    fig.update_traces(mode='lines+markers')  # Adding markers for better visibility
    fig.update_layout(hovermode='x')  # Enhancing hover interaction
    fig.show()

# Predictive Modeling
# Use EUR/USD data for predictive modeling
X = dfs['EURUSD=X'][['Open', 'High', 'Low']]
y = dfs['EURUSD=X']['Close']

# Train a linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance
score = model.score(X_test, y_test)
print("Model score:", score)

# Backtesting
# Backtest the linear regression model
historical_predictions = model.predict(X)
historical_errors = np.abs(historical_predictions - y)
average_error = np.mean(historical_errors)
print("Average error:", average_error)

# Calculate maximum potential loss
max_loss = historical_errors.max()
print("Maximum potential loss:", max_loss)

# Alert System for significant movement
threshold = 0.01  # 1% change

for currency, df in dfs.items():
    latest_rate = df['Close'].iloc[-1]
    previous_rate = df['Close'].iloc[-2]

    change = abs(latest_rate - previous_rate) / previous_rate

    if change > threshold:
        print(f"ALERT: Significant change in {currency_names[currency]}: {change*100:.2f}%")

# Function to calculate percent change
def calculate_percent_change(df):
    df['pct_change'] = df['Close'].pct_change() * 100
    return df

# Fetch and process data for each currency
dfs = {}
for currency in currencies:
    df = yf.download(currency, start='2022-01-01', end='2023-11-14')
    df = calculate_percent_change(df)  # Calculate percent changes
    dfs[currency] = df

# Generate individual graphs for each currency
for currency, df in dfs.items():
    fig = go.Figure()

    # Exchange Rate Trace
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Exchange Rate'))
    fig.update_layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Exchange Rate'),
        title=f'{currency_names[currency]} Exchange Rate and Percent Change Over Time'
    )

    # Percent Change Trace (Secondary Y-axis)
    fig.add_trace(go.Scatter(x=df.index, y=df['pct_change'], mode='lines', name='Percent Change', yaxis='y2'))
    fig.update_layout(
        yaxis2=dict(title='Percent Change (%)', overlaying='y', side='right'),
        showlegend=True,
        hovermode='x'
    )

    fig.show()


# Create individual graphs for percent change
for currency, df in dfs.items():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['pct_change'], label=f'{currency_names[currency]} Percent Change')
    ax.set_title(f'{currency_names[currency]} Percent Change Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Percent Change (%)')
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
