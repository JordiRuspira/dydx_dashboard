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


blue_pastel_palette = ['#A2C3E6', '#B8D4EF', '#CDE3F5', '#E0F1FB', '#8FB3D9']

# Helper function for preparing data for Streamlit charts
def prepare_chart_data(x, y):
    return pd.DataFrame({'Date': x, 'Value': y}).set_index('Date')

# Function to load and preprocess MEV data
def load_mev_data():  
    #url = "https://drive.google.com/uc?id=1f1pfLkNqbwiypDbyWZkHZO12D-_znYq0" 
    #url = "https://drive.google.com/uc?id=1V3eGS5dDcgWeaJuFe7LfhkCtsA19TUXP"
    url = "https://drive.google.com/uc?id=1wUjaA2DkKWVTAGHTU7Lrg0vF3S2ukBcx"
    output = 'filtered_mev_data_with_dates_20250121.csv'
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
fig1 = px.bar(combined_data, x="date", y="Order Book Discrepancy ($)", color_discrete_sequence=blue_pastel_palette)
fig1.update_layout(
    title="Daily Discrepancy",
    xaxis_title="Date",
    yaxis_title="Order book discrepancy ($)", 
    xaxis_tickfont_size=14,
    yaxis_tickfont_size=14,
    bargap=0.15,
    bargroupgap=0.1
)
st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

st.subheader("Daily Volume")
st.text("")
st.markdown("If we compare discrepancy to total daily volume, we can already see that it matches, which might make sense after all. For reference, this data is pulled directly from Numia`s data lenses tool.")
st.text("") 
fig2 = px.bar(combined_data, x="date", y="total_volume", color_discrete_sequence=px.colors.qualitative.Pastel2)
fig2.update_layout(
    title="Daily Volume",
    xaxis_title="Date",
    yaxis_title="Volume ($)", 
    xaxis_tickfont_size=14,
    yaxis_tickfont_size=14,
    bargap=0.15,
    bargroupgap=0.1
)
st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

st.subheader("Ratio: Discrepancy / Volume (in percentage)")
st.text("")
st.markdown("Now that we have both daily volume and daily discrepancy, we can calculate the ratio discrepancy over volume in a daily basis, and we see that it doesn't fluctuate much, except for some higher peaks.")
st.text("") 
fig3 = px.bar(combined_data, x="date", y="ratio", color_discrete_sequence=px.colors.qualitative.Pastel2)
fig3.update_layout(
    title="Daily ratio discrepancy over volume (percentage)",
    xaxis_title="Date",
    yaxis_title="Ratio discrepancy/volume (percentage)", 
    xaxis_tickfont_size=14,
    yaxis_tickfont_size=14,
    bargap=0.15,
    bargroupgap=0.1
)
st.plotly_chart(fig3, theme="streamlit", use_container_width=True)

st.subheader("Daily Orders Created")
st.text("")
st.markdown("In an attempt to look at additional insights, we have also decided to plot daily number of orders created.")
st.text("") 
fig4 = px.bar(open_orders_data, x="date", y="NUM_ORDERS", color_discrete_sequence=px.colors.qualitative.Pastel2)
fig4.update_layout(
    title="Daily number of orders created",
    xaxis_title="Date",
    yaxis_title="Number of orders created", 
    xaxis_tickfont_size=14,
    yaxis_tickfont_size=14,
    bargap=0.15,
    bargroupgap=0.1
)
st.plotly_chart(fig4, theme="streamlit", use_container_width=True)

# Function to load validator data
def get_validator_data():
    validator_api_url = "https://dydx.observatory.zone/api/v1/validator"
    validator_response = requests.get(validator_api_url)
    validator_data = validator_response.json()
    return pd.DataFrame(validator_data.get('validators', []))

# Visualize validator data
st.subheader("Top Validators: 7-Day Rolling Average Empty Block Percentage")
st.text("")
st.markdown("As the committee has highlighted in the last reports, we believe that a key indicator so far to look at performance of nodes is the number of empty blocks that they have proposed (i.e. number of blocks with no matches) compared to the other nodes.")
st.text("")

# Load and process data
df = pd.read_csv('empty_blocks.csv')
validator_df = get_validator_data()
df = df.merge(validator_df[['pubkey', 'moniker']], left_on='validator_moniker', right_on='pubkey', how='left')
df['block_date'] = pd.to_datetime(df['block_date'])
df['rolling_7day_avg_empty_block_pct'] = df.groupby('validator_moniker')['empty_block_pct'].transform(lambda x: x.rolling(window=7).mean() * 100)

# Sort validators by their latest empty block percentage
latest_empty_blocks = df.groupby('moniker').agg({
    'block_date': 'max',
    'rolling_7day_avg_empty_block_pct': 'last'
}).sort_values('rolling_7day_avg_empty_block_pct', ascending=False)

# Get top 10 validators by empty block percentage
top_10_validators_empty = latest_empty_blocks.head(10).index.tolist()

# Create the line chart for empty blocks
fig5 = px.line(
    df[df['moniker'].isin(top_10_validators_empty)], 
    x="block_date", 
    y="rolling_7day_avg_empty_block_pct", 
    color="moniker",
    title="7-Day Rolling Average Empty Block Percentage by Validator (Top 10)",
    labels={
        "block_date": "Block Date", 
        "rolling_7day_avg_empty_block_pct": "Empty Block Percentage (%)",
        "moniker": "Validator"
    },
    color_discrete_sequence=px.colors.qualitative.Pastel,
    category_orders={"moniker": top_10_validators_empty}  # Order the legend based on sorted validators
)

fig5.update_layout(
    xaxis_title="Date",
    yaxis_title="7-Day Rolling Average (%)",
    legend_title="Validator (Sorted by Latest %)",
    xaxis_tickfont_size=12,
    yaxis_tickfont_size=12,
    margin=dict(l=40, r=40, t=40, b=40),
    hovermode="x unified"
)

st.plotly_chart(fig5, theme="streamlit", use_container_width=True)

# Load and process MEV data
df_movingavg = pd.read_csv('mev_filtered_blocks_with_vp_date_new.csv')

st.subheader("Average 7-day Moving Average Order Book Discrepancy")
st.text("")
st.markdown("This chart shows the 7-day moving average of order book discrepancy per block for each validator over time.")
st.text("")

# Sort validators by their latest MEV discrepancy
latest_mev = df_movingavg.groupby('moniker').agg({
    'date': 'max',
    'mev_7day': 'last'
}).sort_values('mev_7day', ascending=False)

# Get top 10 validators by MEV discrepancy
top_10_validators_mev = latest_mev.head(10).index.tolist()

# Create the line chart for MEV data
fig7 = px.line(
    df_movingavg[df_movingavg['moniker'].isin(top_10_validators_mev)], 
    x="date", 
    y="mev_7day",
    color="moniker",
    title="7-day Moving Average Order Book Discrepancy per Block by Validator (Top 10)",
    labels={
        "date": "Date",
        "mev_7day": "Order Book Discrepancy ($)",
        "moniker": "Validator"
    },
    color_discrete_sequence=px.colors.qualitative.Pastel,
    category_orders={"moniker": top_10_validators_mev}  # Order the legend based on sorted validators
)

fig7.update_layout(
    xaxis_title="Date",
    yaxis_title="Order Book Discrepancy ($)",
    legend_title="Validator (Sorted by Latest Discrepancy)",
    xaxis_tickfont_size=14,
    yaxis_tickfont_size=14,
    margin=dict(l=40, r=40, t=40, b=40),
    hovermode="x unified"
)

st.plotly_chart(fig7, theme="streamlit", use_container_width=True)


# Add validator comparison section 
st.subheader("Validator Comparison")
st.text("")
st.markdown("Compare selected validators' performance against the average of all other validators.")
st.text("")

# Calculate latest day's performance for each validator
latest_date = df_movingavg['date'].max()
latest_performance = df_movingavg[df_movingavg['date'] == latest_date]
latest_avg = latest_performance['mev_7day'].mean()

# Get validators performing above average on the latest day
above_avg_validators = latest_performance[latest_performance['mev_7day'] > latest_avg]['moniker'].tolist()

# Get unique validator names for the selector
validators = sorted(df['moniker'].unique())

# Create validator multiselector with above-average validators as default
selected_validators = st.multiselect(
    "Select validators to compare",
    validators,
    default=above_avg_validators
)

# Process empty blocks data for comparison
def prepare_empty_blocks_comparison(df, selected_validators):
    # Selected validators data
    selected_data = df[df['moniker'].isin(selected_validators)]
    
    # Others average data
    others_data = df[~df['moniker'].isin(selected_validators)].groupby('block_date')['rolling_7day_avg_empty_block_pct'].mean().reset_index()
    
    return selected_data, others_data

# Process MEV data for comparison
def prepare_mev_comparison(df, selected_validators):
    # Selected validators data
    selected_data = df[df['moniker'].isin(selected_validators)]
    
    # Others average data
    others_data = df[~df['moniker'].isin(selected_validators)].groupby('date')['mev_7day'].mean().reset_index()
    
    return selected_data, others_data

# Prepare comparison data
empty_blocks_selected, empty_blocks_others = prepare_empty_blocks_comparison(df, selected_validators)
mev_selected, mev_others = prepare_mev_comparison(df_movingavg, selected_validators)

# Create empty blocks comparison chart
fig_empty_compare = px.line(
    title="Empty Blocks: Selected Validators vs Average of Others"
)

# Add lines for each selected validator
for validator in selected_validators:
    validator_data = empty_blocks_selected[empty_blocks_selected['moniker'] == validator]
    fig_empty_compare.add_scatter(
        x=validator_data['block_date'],
        y=validator_data['rolling_7day_avg_empty_block_pct'],
        name=validator,
        line=dict(width=3)
    )

# Add others average line
fig_empty_compare.add_scatter(
    x=empty_blocks_others['block_date'],
    y=empty_blocks_others['rolling_7day_avg_empty_block_pct'],
    name='Average of Other Validators',
    line=dict(color='#CDE3F5', width=2, dash='dash')
)

fig_empty_compare.update_layout(
    xaxis_title="Date",
    yaxis_title="Empty Block Percentage (%)",
    xaxis_tickfont_size=14,
    yaxis_tickfont_size=14,
    legend_title="Validator",
    hovermode="x unified"
)

st.plotly_chart(fig_empty_compare, theme="streamlit", use_container_width=True)

# Create MEV comparison chart
fig_mev_compare = px.line(
    title="Order Book Discrepancy: Selected Validators vs Average of Others"
)

# Add lines for each selected validator
for validator in selected_validators:
    validator_data = mev_selected[mev_selected['moniker'] == validator]
    fig_mev_compare.add_scatter(
        x=validator_data['date'],
        y=validator_data['mev_7day'],
        name=validator,
        line=dict(width=3)
    )

# Add others average line
fig_mev_compare.add_scatter(
    x=mev_others['date'],
    y=mev_others['mev_7day'],
    name='Average of Other Validators',
    line=dict(color='#CDE3F5', width=2, dash='dash')
)

fig_mev_compare.update_layout(
    xaxis_title="Date",
    yaxis_title="Order Book Discrepancy ($)",
    xaxis_tickfont_size=14,
    yaxis_tickfont_size=14,
    legend_title="Validator",
    hovermode="x unified"
)

st.plotly_chart(fig_mev_compare, theme="streamlit", use_container_width=True)
