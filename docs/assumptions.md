
# Modeling Assumptions

| Assumption | Description | Justification | Impact of Violation |
| :--- | :--- | :--- | :--- |
| **Stationarity** | Market regimes persist for > 4 hours. | Allows for short-term trend following. | Alpha decay; false signals. |
| **Slippage** | Follows a square-root law of volume. | Standard institutional model. | Underestimation of txn costs. |
| **Liquidity** | Top 20 crypto assets have infinite liquidity for < $10k orders. | Prevents market impact modeling complexity for small size. | High slippage on thin alts. |
| **Fee Structure** | Binance VIP 0 Tier (Maker 0.02% / Taker 0.04%). | Conservative baseline. | Overestimation of costs (conservative). |
| **Latency** | Network latency < 100ms. | Cloud co-location assumption. | Execution failure in HFT mode. |
