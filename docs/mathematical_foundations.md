
# Mathematical Foundations

## 1. Alpha Formulation
The expected return $E[R]$ is modeled as:
$$ E[R_t] = f(X_t) - C(V_t, \sigma_t) $$
Where $f(X)$ is the ensemble prediction and $C$ is the cost function.

## 2. Risk Methodology
We employ **Conditional Value at Risk (cVaR)** at 95% confidence:
$$ cVaR_{\alpha} = \frac{1}{1-\alpha} \int_{-\infty}^{VaR_{\alpha}} x \cdot p(x) dx $$
This captures tail risk better than standard VaR.

## 3. Probability of Ruin
For a strategy with win rate $w$ and payoff ratio $R$:
$$ P(Ruin) \approx e^{-2 \cdot \frac{\mu}{\sigma^2} \cdot \text{Capital}} $$
Ensuring $\mu > 0$ and managing $\sigma$ via position sizing is critical.
