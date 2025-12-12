
# Feature Justification & Academic Provenance

This document justifies the inclusion of each feature family used in the v25 Model, citing relevant academic literature.

| Feature Family | Specific Features | Academic Justification | Provenance |
| :--- | :--- | :--- | :--- |
| **Momentum** | RSI, MACD, ROC | *Moskowitz, T. J., et al. (2012).* "Time series momentum." Momentum is a persistent anomaly across asset classes. | `src/features/momentum.py` |
| **Volatility** | ATR, Bollinger Bands, GARCH Variance | *Bollerslev, T. (1986).* "Generalized autoregressive conditional heteroskedasticity." Volatility clustering is a key property of financial time series. | `src/features/volatility.py` |
| **Volume/Liquidity** | OBV, VWAP, Orderbook Imbalance | *Cartea, A., et al. (2015).* Algorithmic and High-Frequency Trading. Order flow imbalance is a primary driver of short-term price discovery. | `src/features/liquidity.py` |
| **On-Chain** | Hashrate, Active Addresses | *Liu, Y., & Tsyvinski, A. (2021).* "Risks and Returns of Cryptocurrency." Network factors significantly predict crypto returns. | `src/features/onchain.py` |
| **Sentiment** | Twitter Sentiment, Funding Rate | *Bollen, J., et al. (2011).* "Twitter mood predicts the stock market." Social sentiment is a leading indicator for retail-driven assets. | `src/features/sentiment.py` |

## Model Derivations

### Regime Filter (Hidden Markov Model)
We assume market returns $r_t$ follow a mixture distribution driven by a hidden state $S_t \in \{1, \dots, K\}$:
$$ r_t | S_t = k \sim N(\mu_k, \sigma_k^2) $$
The transition probability is $P(S_t=j | S_{t-1}=i) = A_{ij}$.
We use Gaussian Mixture Models (GMM) as a proxy for HMM states to classify regimes (Bull, Bear, Flat).

### Risk: 5-Factor PCA
We decompose the covariance matrix $\Sigma$ of returns into 5 principal components:
$$ R = \beta \cdot F + \epsilon $$
Where $F$ are the latent risk factors accounting for >85% of variance.
