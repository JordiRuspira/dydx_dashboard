import streamlit as st
import pandas as pd
import requests

# Helper function for preparing data for Streamlit charts
def prepare_chart_data(x, y):
    return pd.DataFrame({'Date': x, 'Value': y}).set_index('Date')

# Function to load and preprocess MEV data
def load_mev_data():
    mev_df = pd.read_csv('filtered_mev_data_with_dates.csv')
    mev_df['date'] = pd.to_datetime(mev_df['date'], utc=True).dt.date
    mev_df = mev_df.dropna(subset=['date'])
    daily_aggregated = mev_df.groupby('date')['Order Book Discrepancy ($)'].sum().reset_index()
    return daily_aggregated

# Function to load and preprocess volume data
def load_volume_data():
    volume_df = pd.read_csv('dydx_daily_volume.csv')
    volume_df['labels'] = (pd.to_datetime(volume_df['labels'], utc=True) + pd.Timedelta(days=1)).dt.date
    volume_df['total_volume'] = pd.to_numeric(volume_df['total_volume'], errors='coerce')
    return volume_df

# Function to merge MEV and volume data
def merge_mev_volume_data(mev_data, volume_data):
    combined_df = pd.merge(
        mev_data, volume_data,
        left_on='date', right_on='labels',
        how='inner'
    )
    combined_df['ratio'] = 100 * (combined_df['Order Book Discrepancy ($)'] / combined_df['total_volume'])
    return combined_df

# Function to load and preprocess open orders data
def load_open_orders():
    open_orders_df = pd.read_csv('open_orders.csv')
    open_orders_df['order_submitted_time'] = pd.to_datetime(open_orders_df['order_submitted_time'], errors='coerce')
    open_orders_df['date'] = open_orders_df['order_submitted_time'].dt.date
    daily_orders = open_orders_df.groupby('date')['NUM_ORDERS'].sum().reset_index()
    return daily_orders

# Streamlit setup
st.set_page_config(page_title="MEV Data Dashboard", layout="wide")
st.title("MEV Data Analysis Dashboard")

# Load data
mev_data = load_mev_data()
volume_data = load_volume_data()
combined_data = merge_mev_volume_data(mev_data, volume_data)
open_orders_data = load_open_orders()

# Create charts using Streamlit
st.subheader("Daily Discrepancy")
st.bar_chart(prepare_chart_data(combined_data['date'], combined_data['Order Book Discrepancy ($)']))

st.subheader("Daily Orders Created")
st.bar_chart(prepare_chart_data(open_orders_data['date'], open_orders_data['NUM_ORDERS']))

st.subheader("Daily Volume")
st.bar_chart(prepare_chart_data(combined_data['date'], combined_data['total_volume']))

st.subheader("Ratio: Discrepancy / Volume")
st.line_chart(prepare_chart_data(combined_data['date'], combined_data['ratio']))

# Function to load validator data
def get_validator_data():
    validator_api_url = "https://dydx.observatory.zone/api/v1/validator"
    validator_response = requests.get(validator_api_url)
    validator_data = validator_response.json()
    return pd.DataFrame(validator_data.get('validators', []))

# Visualize 7-day rolling average of empty block percentages for validators
st.subheader("Top Validators: 7-Day Rolling Average Empty Block Percentage")
df = pd.read_csv('empty_blocks_2.csv')
validator_df = get_validator_data()
df = df.merge(validator_df[['pubkey', 'moniker']], left_on='validator_moniker', right_on='pubkey', how='left')
df['block_date'] = pd.to_datetime(df['block_date'])
df['rolling_7day_avg_empty_block_pct'] = df.groupby('validator_moniker')['empty_block_pct'].transform(lambda x: x.rolling(window=7).mean() * 100)

validator_chart_data = df[['block_date', 'rolling_7day_avg_empty_block_pct', 'moniker']].dropna()
plot_data = df.pivot_table(
    index='block_date', 
    columns='moniker', 
    values='rolling_7day_avg_empty_block_pct'
)


st.subheader("7-Day Rolling Average Empty Block Percentage by Validator")
st.line_chart(plot_data)
