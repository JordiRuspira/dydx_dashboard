SELECT
  date(publish_time) AS order_submitted_time,
  COUNT(*) AS NUM_ORDERS
FROM
  `numia-data.dydx_mainnet.dydx_mempool_transactions` m
WHERE
  JSON_EXTRACT_SCALAR(m.attributes, '$.tx_msg_type') = '/dydxprotocol.clob.MsgPlaceOrder'
  AND date(publish_time)  >= '2024-08-01'
GROUP BY
  date(publish_time) 
