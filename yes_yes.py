import pandas as pd
import yfinance as yf
import plotly
import plotly.graph_objs as go
import json
from flask import Flask, render_template, request

# Define your currency symbols and names
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

app = Flask(__name__)

# Function to fetch current exchange rates
def fetch_exchange_rates():
    exchange_rates = {}
    for currency in currencies:
        try:
            currency_data = yf.Ticker(currency)
            data = currency_data.history(period='1d')
            if not data.empty:
                current_rate = data['Close'].iloc[0]
                exchange_rates[currency] = round(current_rate, 2)  # Rounded to 2 decimal places
            else:
                print(f"No price data found for {currency}")
        except Exception as e:
            print(f"Error fetching data for {currency}: {e}")
    return exchange_rates

# Function to calculate daily changes
def calculate_daily_changes():
    daily_changes = {}
    for currency in currencies:
        try:
            currency_data = yf.Ticker(currency)
            data = currency_data.history(period='2d')
            if not data.empty and len(data) > 1:
                prev_close = data['Close'].iloc[-2]
                current_close = data['Close'].iloc[-1]
                change = ((current_close - prev_close) / prev_close) * 100
                daily_changes[currency] = round(change, 2)  # Rounded to 2 decimal places
            else:
                print(f"Not enough data to calculate daily changes for {currency}")
        except Exception as e:
            print(f"Error fetching data for {currency}: {e}")
    return daily_changes

# Function to identify significant movements (changes over 1%)
def identify_significant_movements():
    daily_changes = calculate_daily_changes()
    significant_movements = {currency: change for currency, change in daily_changes.items() if abs(change) > 1}
    return significant_movements

# Function to generate Plotly graph JSON for a currency, with exchange rate and percentage change
def generate_plotly_graph_json(currency, df):
    # Calculate the percentage change from the start for each point
    start_rate = df['Close'].iloc[0]
    df['Percent Change'] = df['Close'].apply(lambda x: ((x - start_rate) / start_rate) * 100)

    # Create a line plot for the exchange rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Exchange Rate'))

    # Add a secondary y-axis for the percentage change
    fig.update_layout(
        yaxis_title='Exchange Rate',
        yaxis2=dict(
            title='Percent Change',
            overlaying='y',
            side='right',
            showgrid=False,
        )
    )

    # Add a line plot for the percentage change on the secondary y-axis
    fig.add_trace(go.Scatter(x=df.index, y=df['Percent Change'], name='Percent Change', yaxis='y2', mode='lines'))

    # Add annotation for the final percentage change value
    final_pct_change = df['Percent Change'].iloc[-1]
    fig.add_annotation(x=df.index[-1], y=final_pct_change, text=f'{final_pct_change:.2f}%', showarrow=True, arrowhead=1)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Store historical exchange rate data in dictionaries
dfs = {}
for currency in currencies:
    try:
        df = yf.download(currency, start='2022-01-01', end='2023-11-14')
        dfs[currency] = df
        print(f"Fetched data for {currency}")
    except Exception as e:
        print(f"Error fetching data for {currency}: {e}")

# Flask route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    exchange_rates = fetch_exchange_rates()
    daily_changes = calculate_daily_changes()
    significant_movements = identify_significant_movements()
    graphs_json = {}  # Initialize graphs_json as an empty dictionary

    if request.method == 'POST':
        selected_currencies = request.form.getlist('currencies')
        print("Selected Currencies:", selected_currencies)  # Print selected currencies for debugging

        for currency in selected_currencies:
            if currency in dfs:
                df = dfs[currency]
                graph_json = generate_plotly_graph_json(currency, df)
                graphs_json[currency] = graph_json
                print(f"Graph JSON for {currency}:", graph_json)  # Print graph JSON for debugging
    else:
        selected_currencies = []  # Initialize selected_currencies as an empty list for GET requests

    return render_template('index.html', exchange_rates=exchange_rates,
                           daily_changes=daily_changes, significant_movements=significant_movements,
                           currency_names=currency_names, currency_options=currencies, 
                           selected_currencies=selected_currencies, graphs_json=graphs_json)

if __name__ == '__main__':
    app.run(debug=True)

