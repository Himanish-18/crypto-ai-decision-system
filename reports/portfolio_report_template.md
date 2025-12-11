# Daily Portfolio Report: {{ date }}

## 1. Executive Summary
- **Total PnL**: ${{ total_pnl_usd }} ({{ total_pnl_pct }}%)
- **Sharpe Ratio**: {{ sharpe_ratio }}
- **Max Drawdown**: {{ max_drawdown_pct }}%

## 2. Asset Performance
| Asset | Alloc % | PnL ($) | PnL (%) |
|-------|---------|---------|---------|
{% for asset in assets %}
| {{ asset.symbol }} | {{ asset.alloc }} | {{ asset.pnl_usd }} | {{ asset.pnl_pct }} |
{% endfor %}

## 3. Risk Metrics
- **Portfolio Beta**: {{ portfolio_beta }}
- **Value at Risk (95%)**: ${{ var_95 }}

## 4. Execution Attribution
- **Total Slippage**: ${{ total_slippage_usd }}
- **Avg Spread Paid**: {{ avg_spread_bps }} bps
