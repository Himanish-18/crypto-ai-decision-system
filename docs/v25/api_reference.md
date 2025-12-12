
# API Reference

## Science Module
### `src.science.formulation`
- `MathematicalFormulation`: Constants and equation definitions.
- `ScienceLossFunctions`: Custom loss functions (Sharpe, Sortino).

### `src.science.validation`
- `StatisticalValidator.bootstrap_confidence_interval(data, func, n=1000)`: Returns 95% CI.
- `StatisticalValidator.diebold_mariano_test(real, p1, p2)`: Compares two models.

## Backtesting Module
### `src.backtesting.hf_backtester`
- `HedgeFundBacktester(data, signal, config)`: Main class.
- `HedgeFundBacktester.run()`: Executes simulation.
- `PerformanceMetrics`: Static methods for Sharpe, Omega, cVaR.

## Risk Module
### `src.risk.institutional_v25.formal_risk`
- `SensitivityAnalysis.calculate_probability_of_ruin(win_rate, reward_risk, risk_per_trade)`

## Execution Module
### `src.execution.quant_execution_v25`
- `QuantExecutorV25.execute_order(symbol, qty, side)`: Async execution entry point.
- `MarketMicrostructure.calculate_microprice(bid, ask, b_qty, a_qty)`
