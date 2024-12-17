import streamlit as st
import pandas as pd
import requests
import gdown

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

def get_sorted_validators(df, column):
        last_values = df.groupby('moniker').apply(lambda x: x.sort_values('time').iloc[-1][column])
        sorted_validators = last_values.sort_values(ascending=False).index.tolist()
        return sorted_validators

url = "https://drive.google.com/uc?id=1kv6gzzwd4tKz-PMfdvyadJWgYv5vnl4r"
output = 'mev_filtered_blocks_with_vp.csv'
gdown.download(url, output, quiet=False) 
df = pd.read_csv(output)

for category in ['2.5-100%', '1-2.5%', '0-1%']:
    df_category = df[df['Voting Power Category'] == category]

    # Sort validators by their last 7-day MA value
    sorted_validators_7day = get_sorted_validators(df_category, 'mev_7day')

    # Create a figure for the 7-day moving average
    fig_7day = go.Figure()

    for validator in sorted_validators_7day:
        df_val = df_category[df_category['moniker'] == validator]

        fig_7day.add_trace(go.Scatter(
            x=df_val.index, 
            y=df_val['mev_7day'], 
            mode='lines', 
            name=f'{validator} 7-day MA',
            hoverinfo='x+y+name',  # Show only the date, discrepancy value, and validator name on hover
            line=dict(width=2),
            text=[f'{validator}<br>Discrepancy: {val:.2f}' for val in df_val['mev_7day']]
        ))

        # Update layout for the 7-day moving average
    fig_7day.update_layout(
        title=f'Average 7-day MA order book discrepancy per block for validators with voting power between {category}',
        xaxis_title='Time',
        yaxis_title='Order Book Discrepancy ($)',
        hovermode="closest",  # Show info for the nearest line only
        legend_title='Validators',
        template='plotly_white',
        height=800
    )

    # Show the interactive plot for 7-day MA
    st.plotly_chart(fig_7day, theme="streamlit", use_container_width=True)





