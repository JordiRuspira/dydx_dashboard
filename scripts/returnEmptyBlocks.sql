WITH blocks AS (
  SELECT proposer_address AS validator_moniker,
         block_height,
         DATE(block_timestamp) AS block_date -- Add block_date for daily calculations
  FROM `numia-data.dydx_mainnet.dydx_blocks` 
  WHERE block_timestamp > '2024-06-01'
),
per_block AS (
  SELECT validator_moniker,
         b.block_height,
         b.block_date,
         CASE 
            WHEN f.block_height IS NULL THEN 'empty block' 
            ELSE 'non_empty' 
         END AS is_empty_block
  FROM blocks b
  LEFT JOIN `numia-data.dydx_mainnet.dydx_match` f
    ON f.block_height = b.block_height
),
daily_data AS (
  SELECT validator_moniker,
         block_date,
         COUNT(DISTINCT block_height) AS num_blocks_proposed,
         COUNT(DISTINCT CASE WHEN is_empty_block = 'empty block' THEN block_height ELSE NULL END) / COUNT(DISTINCT block_height) AS empty_block_pct
  FROM per_block
  GROUP BY validator_moniker, block_date
)
SELECT validator_moniker,
       block_date,
       num_blocks_proposed,
       empty_block_pct,
       -- Rolling 7-day average using window function
       AVG(empty_block_pct) OVER (PARTITION BY validator_moniker ORDER BY block_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7day_avg_empty_block_pct
FROM daily_data
ORDER BY validator_moniker, block_date;
