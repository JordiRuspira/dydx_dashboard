import streamlit as st
import pandas as pd
import requests
import gdown
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.ticker as ticker
import numpy as np
import plotly.express as px



# Helper function for preparing data for Streamlit charts
def prepare_chart_data(x, y):
    return pd.DataFrame({'Date': x, 'Value': y}).set_index('Date')

# Function to load and preprocess MEV data
def load_mev_data():  
    #url = "https://drive.google.com/file/d/1f1pfLkNqbwiypDbyWZkHZO12D-_znYq0/view?usp=drive_link"
    url = "https://drive.google.com/uc?id=1f1pfLkNqbwiypDbyWZkHZO12D-_znYq0"
    output = 'filtered_mev_data_with_dates.csv'
    gdown.download(url, output, quiet=False)
    # Load into DataFrame
    mev_df = pd.read_csv(output)
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
st.text("")
st.markdown("This dashboard is maintained by the dYdX MEV Committee and the data is refereshed weekly, the reason being that most of it is both extracted from Numia and calculations are made on top of that ad-hoc (we will list the exact scripts for the community to scrutinize), and if it was to be done on each dashboard refresh, it would take too long to load.")
st.text("")
st.markdown("As part of our work in the MEV Committee, we believe that having a weekly refreshed dashboard can help users keep track of the work and see insights as well. Individual block discrepancies are analyzed using methods mentioned in previous reports. The goal here is to have also a historical view of what's been happening so far.")
st.text("")


# Load data
mev_data = load_mev_data()
volume_data = load_volume_data()
combined_data = merge_mev_volume_data(mev_data, volume_data)
open_orders_data = load_open_orders()

# Create charts using Streamlit
st.subheader("Daily Discrepancy")
st.text("")
st.markdown("A first metric that we can show is the total daily order book discrepancy.")
st.text("")
st.bar_chart(prepare_chart_data(combined_data['date'], combined_data['Order Book Discrepancy ($)']))

fig1 = px.bar(combined_data, x="date", y="Order Book Discrepancy ($)'", color_discrete_sequence=px.colors.qualitative.Pastel2)
    fig1.update_layout(
    title="Daily Discrepancy",
    xaxis_title="Date",
    yaxis_title="Order book discrepancy ($)", 
    xaxis_tickfont_size=14,
    yaxis_tickfont_size=14,
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
   


st.subheader("Daily Volume")
st.text("")
st.markdown("If we compare discrepancy to total daily volume, we can already see that it matches, which might make sense after all. For reference, this data is pulled directly from Numia`s data lenses tool.")
st.text("")
st.bar_chart(prepare_chart_data(combined_data['date'], combined_data['total_volume']))


st.subheader("Ratio: Discrepancy / Volume (in percentage)")
st.text("")
st.markdown("Now that we have both daily volume and daily discrepancy, we can calculate the ratio discrepancy over volume in a daily basis, and we see that it doesn't fluctuate much, except for some higher peaks.")
st.text("")
st.line_chart(prepare_chart_data(combined_data['date'], combined_data['ratio']))


st.subheader("Daily Orders Created")
st.text("")
st.markdown("In an attempt to look at additional insights, we have also decided to plot daily number of orders created.")
st.text("")
st.bar_chart(prepare_chart_data(open_orders_data['date'], open_orders_data['NUM_ORDERS']))
 


# Function to load validator data
def get_validator_data():
    validator_api_url = "https://dydx.observatory.zone/api/v1/validator"
    validator_response = requests.get(validator_api_url)
    validator_data = validator_response.json()
    return pd.DataFrame(validator_data.get('validators', []))

# Visualize 7-day rolling average of empty block percentages for validators
st.subheader("Top Validators: 7-Day Rolling Average Empty Block Percentage")
st.text("")
st.markdown("As the committee has highlighted in the last reports, we believe that a key indicator so far to look at performance of nodes is the number of empty blocks that they have proposed (i.e. number of blocks with no matches) compared to the otehr nodes.")
st.text("")
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



