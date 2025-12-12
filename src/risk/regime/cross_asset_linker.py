
import numpy as np
import pandas as pd

# v39 Cross-Asset Macro Scanner
# Detects co-movement between BTC, ETH, Nasdaq, and Gold.

class CrossAssetLinker:
    def __init__(self, tickers=["BTC", "ETH", "NDX", "GOLD"]):
        self.tickers = tickers
        self.returns_buffer = {t: [] for t in tickers}
        self.window_size = 60
        
    def update(self, ticker, price):
        if ticker in self.returns_buffer:
            self.returns_buffer[ticker].append(price)
            if len(self.returns_buffer[ticker]) > self.window_size + 1:
                self.returns_buffer[ticker].pop(0)

    def compute_correlation_matrix(self):
        # Align series (Simplification: assuming sync updates for now)
        data = {}
        min_len = min([len(v) for v in self.returns_buffer.values()])
        if min_len < 30: return None
        
        for t in self.tickers:
            series = pd.Series(self.returns_buffer[t][-min_len:])
            data[t] = series.pct_change().dropna()
            
        df = pd.DataFrame(data)
        return df.corr()

    def detect_coupling(self):
        corr = self.compute_correlation_matrix()
        if corr is None: return "UNCERTAIN"
        
        # Check BTC vs Nasdaq
        if "BTC" in corr and "NDX" in corr:
            btc_ndx = corr.loc["BTC", "NDX"]
            if btc_ndx > 0.8: return "RISK_ON_COUPLING"
            if btc_ndx < -0.5: return "DECOUPLING"
            
        return "INDEPENDENT"
