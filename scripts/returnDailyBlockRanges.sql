WITH daily_blocks AS (
  SELECT 
    block_height,
    DATE(block_timestamp) AS block_date,
    ROW_NUMBER() OVER (PARTITION BY DATE(block_timestamp) ORDER BY block_height ASC) AS rn_asc,
    ROW_NUMBER() OVER (PARTITION BY DATE(block_timestamp) ORDER BY block_height DESC) AS rn_desc
  FROM 
    `numia-data.dydx_mainnet.dydx_blocks`
  WHERE 
    block_timestamp >= '2024-07-01 00:00:00'  
)
SELECT 
  block_date,
  MAX(CASE WHEN rn_asc = 1 THEN block_height END) AS first_block_height,
  MAX(CASE WHEN rn_desc = 1 THEN block_height END) AS last_block_height
FROM 
  daily_blocks
GROUP BY 
  block_date
ORDER BY 
  block_date;
