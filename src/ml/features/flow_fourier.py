import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

class NoiseRegimeClassifier:
    """
    Uses Fast Fourier Transform (FFT) to analyze the frequency domain of price returns.
    High High-Frequency Energy = Choppy/Noise.
    High Low-Frequency Energy = Trending/Signal.
    """
    def __init__(self, window_size: int = 64):
        self.window_size = window_size
        
    def calculate_regime(self, prices: np.ndarray) -> float:
        """
        Returns a 'Noise Score' (0.0 to 1.0).
        1.0 = Pure White Noise
        0.0 = Smooth Trend
        """
        if len(prices) < self.window_size:
            return 0.5
            
        # Normalize: Returns or Detrended Prices?
        # Detrended Prices are better for simple spectral analysis of residuals
        # But Returns are stationary. Let's use Returns.
        # Actually, let's use Log Returns.
        prices = np.array(prices)
        returns = np.diff(np.log(prices))
        
        # Take snippet
        snippet = returns[-self.window_size:]
        
        # Apply Hanning Window to reduce spectral leakage
        windowed = snippet * np.hanning(len(snippet))
        
        # FFT
        yf = fft(windowed)
        xf = fftfreq(len(snippet), 1)[:len(snippet)//2]
        
        # Power Spectrum
        power = 2.0/len(snippet) * np.abs(yf[0:len(snippet)//2])
        
        # Energy Ratio: High Freq / Total Energy
        # Split point: Lower 1/4 vs Upper 3/4?
        # Or simply Entropy of spectrum?
        
        # Metric 1: High Freq Ratio
        mid_idx = len(power) // 2
        low_energy = np.sum(power[:mid_idx])
        high_energy = np.sum(power[mid_idx:])
        total_energy = low_energy + high_energy + 1e-9
        
        noise_ratio = high_energy / total_energy
        
        return noise_ratio

    def enrich_dataframe(self, df: pd.DataFrame, col="close") -> pd.DataFrame:
        """
        Rolling calculation of noise regime.
        """
        if col not in df.columns: return df
        
        prices = df[col].values
        noise_scores = np.zeros(len(prices))
        noise_scores[:] = np.nan
        
        # Optimization: Don't do true rolling apply (slow), do blocked or strided?
        # For simplicity in v1, we do a loop or rolling apply.
        # Rolling apply is slow in pandas.
        
        # Vectorized heuristic?
        # NO, FFT is complex.
        # Let's implementation a simplified rolling variance ratio first as proxy, 
        # but User explicitly asked for FFT.
        
        # We will iterate for the last few hundred rows only if running live.
        # If backtesting, this will be slow.
        
        # For LIVE usage (incremental), we usually only need the LAST value.
        # So we create a method `get_last_state`.
        
        scores = []
        for i in range(len(prices)):
            if i < self.window_size:
                scores.append(0.5)
                continue
            
            p_window = prices[i-self.window_size:i]
            score = self.calculate_regime(p_window)
            scores.append(score)
            
        df["feat_fft_noise"] = scores
        return df
