from typing import Tuple

import numpy as np
import pandas as pd


class NoiseFilter:
    """
    Detects market 'Chop' or 'Noise' regimes using low-frequency power (FFT)
    and realized variance.
    """

    def __init__(
        self,
        window: int = 30,
        fft_threshold: float = 0.20,
        var_threshold: float = 0.0005,
    ):
        self.window = window
        self.fft_threshold = fft_threshold
        self.var_threshold = var_threshold

    def is_chop(self, closes: pd.Series) -> bool:
        """
        Check if the current window is dominated by noise (Chop).

        Logic:
           - Realized Variance < Threshold (Low movement amplitude)
           - AND FFT Low-Frequency Power < Threshold (Lack of strong trend component)

        Returns: True if Chop/Noise, False if Trend/Signal.
        """
        if len(closes) < self.window:
            return False

        recent = closes.iloc[-self.window :].values

        # 1. Realized Variance (of returns)
        returns = np.diff(np.log(recent))
        realized_var = np.var(returns)

        # 2. FFT Power Spectrum
        # Normalize to favor trend detection vs noise
        # Detrend first to analyze fluctuations?
        # Or analyze raw? Usually detrend linear to find cycles.
        # Simple approach applied: Percentage of power in lower frequencies.

        # Linear Detrend
        x = np.arange(len(recent))
        p = np.polyfit(x, recent, 1)
        trend = np.polyval(p, x)
        detrended = recent - trend

        # FFT
        fft_vals = np.fft.rfft(detrended)
        power = np.abs(fft_vals) ** 2
        total_power = np.sum(power)

        if total_power == 0:
            return True

        # Low Frequency (First 3 components usually trend/cycle)
        low_freq_power = np.sum(power[1:4])  # Skip 0 (DC)
        lf_ratio = low_freq_power / total_power

        # Decision
        # Chop = Low Variance AND Low Trend Power
        # Actually, Chop often has HIGH Variance (Volatile sideways).
        # User Spec: "Variance < Threshold AND FFT LF < 20%".
        # This implies "Quiet Noise" (Dead market).

        is_quiet = realized_var < self.var_threshold
        is_noisy_spectrum = (
            lf_ratio < self.fft_threshold
        )  # Energy is spread (White Noise)

        return is_quiet and is_noisy_spectrum
