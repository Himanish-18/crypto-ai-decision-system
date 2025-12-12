
# Institutional Core v25: Quantitative Trading System

**Status**: Institutional Grade Candidate  
**Version**: 25.0.0  
**License**: Proprietary

## 🚀 Overview

The Institutional Core v25 is a state-of-the-art quantitative trading engine designed for robustness, scientific rigor, and institutional-grade risk management. It supersedes previous versions by eliminating reliance on retail-grade logic and introducing formal mathematical verification, advanced machine learning reproducibility, and adaptive execution algorithms.

## ✨ Key Features

### 1. Scientific Foundation
- **Formal Verification**: Loss functions derived from financial utility theory (Sharpe, Sortino).
- **Validation Suite**: Bootstrap confidence intervals, Monte Carlo permutation tests, Diebold-Mariano tests.
- **Benchmark Pipeline**: Automated comparison against ARIMA, GARCH, and Logistic Regression baselines.

### 2. Machine Learning & Reproducibility
- **Traceability**: Full experiment tracking via MLflow/W&B.
- **Reproducibility**: Deterministic seeding and dataset versioning (DVC).
- **Explainability**: SHAP (Shapley Additive Explanations) for feature attribution.

### 3. Hedge-Fund Grade Backtesting
- **Advanced Metrics**: cVaR (95%), Omega Ratio, Calmar Ratio.
- **Realistic Simulation**: Regime-dependent slippage, variable transaction costs, market impact models.
- **Stress Testing**: Historical scenario replay (COVID-19, 2008 Crash, FTX Collapse).

### 4. Smart Execution (No-HFT)
- **Algorithms**: Adaptive TWAP, VWAP, and Liquidity Slicing.
- **Microstructure**: Orderbook imbalance and microprice inference for passive filling.
- **Urgency**: Dynamic urgency scaling based on alpha decay.

### 5. Risk Management
- **Formal Gates**: Probability of Ruin calculations.
- **Portfolio Guard**: Real-time exposure limits and leverage checks.
- **Kill Switch**: Automated "Hard-Veto" system.

## 🛠️ Architecture

The system follows a modular microservices architecture:

- **Data Service**: Ingestion and feature engineering.
- **Strategy Service**: ML inference and signal generation.
- **Execution Service**: Smart order routing and algo execution.
- **Risk Service**: Pre-trade and post-trade risk validation.
- **Monitoring**: Prometheus/Grafana observability stack.

## 📦 Installation & Usage

### Prerequisites
- Python 3.10+
- Docker

### Setup
```bash
# Clone repository
git clone <repo_url>

# Build Docker environment
docker build -t v25-core .

# Run Unit Tests
pytest tests/unit

# Run Backtest
python src/backtesting/hf_backtester.py
```

## 📚 Documentation
- [Architecture Details](ARCHITECTURE.md)
- [Research Paper](RESEARCH_PAPER.md)
- [API Reference](api_reference.md)
