import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Function to get MEV data (including blocks with no discrepancies)
def get_mev_data(initial_block_height, final_block_height):
    all_data = []
    step = 50000
    for start in range(initial_block_height, final_block_height, step):
        end = min(start + step - 1, final_block_height)
        mev_api_url = f"https://dydx.observatory.zone/api/v1/raw_mev?limit=500000&from_height={start}&to_height={end}&with_block_info=True"
        try:
            mev_response = requests.get(mev_api_url, timeout=30)
            mev_response.raise_for_status()
            mev_data = mev_response.json()
            mev_datapoints = mev_data.get('datapoints', [])
            all_data.extend(mev_datapoints)
        except requests.RequestException as e:
            print(f"Error fetching MEV data for block range {start} to {end}: {e}")
            continue
    # Convert to DataFrame and ensure missing discrepancy is treated as 0
    df = pd.DataFrame(all_data)
    if not df.empty:
        df['value'] = df['value'].fillna(0).astype(float)
    return df

# Function to get validator data
def get_validator_data():
    validator_api_url = "https://dydx.observatory.zone/api/v1/validator"
    try:
        validator_response = requests.get(validator_api_url, timeout=30)
        validator_response.raise_for_status()
        validator_data = validator_response.json()
        return pd.DataFrame(validator_data.get('validators', []))
    except requests.RequestException as e:
        print(f"Error fetching validator data: {e}")
        return pd.DataFrame()

# Function to process and filter MEV data
def process_data(mev_df, validator_df, final_block_height):
    mev_df['value'] = mev_df['value'].astype(float)
    mev_df['height'] = mev_df['height'].astype(int)
    mev_df['Order Book Discrepancy ($)'] = np.where(
    mev_df['probability'] == 1, 
    mev_df['value'] / 10**6, 
    0
    )
    #mev_df = mev_df[(mev_df['probability'] != 0) | (mev_df['value'] <= 0)]

    # Estimate timestamp based on block height difference
    current_time = datetime.utcnow()
 
    # Merge MEV data with validator data
    merged_df = pd.merge(mev_df, validator_df, left_on='proposer', right_on='pubkey', how='left')

    # Filter data for the last 30 days using timestamps 
    return merged_df

# Main function to check MEV values and generate plots
def check_mev_values_and_generate_plots():
    print("Starting MEV value check")

    # Input the initial and final block heights
    try:
        initial_block_height = int(input("Enter the initial block height: ")) 
        final_block_height = int(input("Enter the final block height: ")) 
    except ValueError:
        print("Invalid block height input. Please enter valid integers.")
        return

    mev_df = get_mev_data(initial_block_height, final_block_height)
    validator_df = get_validator_data()
    
    if mev_df.empty:
        print("No MEV data found")
        return

    filtered_df = process_data(mev_df, validator_df, final_block_height)
    
    if filtered_df.empty:
        print("No blocks found in the last 30 days")
        return
 
    # Join the data on moniker and validator name
    df = filtered_df
    # df.loc[df['probability'] == 0, 'Order Book Discrepancy ($)'] = 0
    daily_block_ranges = pd.read_csv('daily_block_ranges.csv')

    # Ensure 'block_date' in daily_block_ranges is in datetime format
    daily_block_ranges['block_date'] = pd.to_datetime(daily_block_ranges['block_date'])
    
    # Assign dates to blocks based on the block ranges
    df['date'] = df['height'].apply(
        lambda block: daily_block_ranges.loc[
            (daily_block_ranges['first_block_height'] <= block) &
            (daily_block_ranges['last_block_height'] >= block),
            'block_date'
        ].values[0] if not daily_block_ranges.loc[
            (daily_block_ranges['first_block_height'] <= block) &
            (daily_block_ranges['last_block_height'] >= block)
        ].empty else None
    ) 
    # Aggregate Order Book Discrepancy by date and moniker
    df_aggregated = df.groupby(['date', 'moniker'], as_index=False).agg({
        'Order Book Discrepancy ($)': 'sum',
        'moniker': 'first',
        'height': 'nunique'  # Count unique heights
    })

    # Rename the new column for clarity
    df_aggregated.rename(columns={'height': 'Unique Heights'}, inplace=True)
    df_aggregated['ratio'] = df_aggregated['Order Book Discrepancy ($)']/df_aggregated['Unique Heights']
    # Sort by moniker and date
    df_aggregated = df_aggregated.sort_values(by=['moniker', 'date'])
    # Set 'date' as the index for the rolling operation
    df_aggregated.set_index('date', inplace=True)
    # Calculate the 7-day moving average for the aggregated data
    df_aggregated['mev_7day'] = df_aggregated.groupby('moniker')['ratio'].apply(
        lambda x: x.rolling('7D').mean()).reset_index(level=0, drop=True)
   
    # Save the results to CSV
    df_aggregated.to_csv('mev_filtered_blocks_with_vp_date_new.csv', index=True)


    # Function to get sorted validators by their last moving average value
    def get_sorted_validators(df, column):
        last_values = df.groupby('moniker').apply(lambda x: x.sort_values('date').iloc[-1][column])
        sorted_validators = last_values.sort_values(ascending=False).index.tolist()
        return sorted_validators

    # Sort validators by their last 7-day MA value
    sorted_validators_7day = get_sorted_validators(df_aggregated, 'mev_7day') 
    # Create a figure for the 7-day moving average
    fig_7day = go.Figure()

    for validator in sorted_validators_7day: 
        df_val = df_aggregated[df_aggregated['moniker'] == validator]

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
        title=f'Average 7-day MA order book discrepancy per block for validators',
        xaxis_title='Time',
        yaxis_title='Order Book Discrepancy ($)',
        hovermode="closest",  # Show info for the nearest line only
        legend_title='Validators',
        template='plotly_white',
        height=800
    )

    # Show the interactive plot for 7-day MA
    fig_7day.show()
 
       

check_mev_values_and_generate_plots()
