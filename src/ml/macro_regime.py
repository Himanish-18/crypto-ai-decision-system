import logging
from typing import Any, Dict

from src.data.multi_feed import MultiFeedRouter

logger = logging.getLogger("macro_regime")


class MacroRegimeModel:
    """
    v17 Global Macro Intelligence.
    Classifies Market Regime based on Funding, OI, and Volatility.
    States: RISK_ON, RISK_OFF, LIQ_CRUNCH, EXPANSION, VOL_SHOCK, NEUTRAL.
    """

    def __init__(self):
        self.current_regime = "NEUTRAL"
        self.feed_router = MultiFeedRouter()

    def _fetch_live_metrics(self) -> Dict[str, float]:
        """
        Fetch Live Macro Data (GVOL, DXY, VIX, Funding) via Redundant Router.
        """
        return {
            "gvol": self.feed_router.get_redundant_gvol(),
            "funding_consensus": self.feed_router.get_redundant_funding(),
            "dxy_trend": 0.0,
            "vix": 15.0,
        }

    def analyze_regime(self, market_data: Dict[str, Any]) -> str:
        """
        Input:
            market_data with keys: 'funding_rate', 'open_interest', 'volatility', 'candles'
        Output:
            Regime String
        """
        try:
            # 1. Fetch External Macro Data
            macro_feed = self._fetch_live_metrics()

            # Extract Metrics (Merge passed data with live feed)
            funding = market_data.get("funding_rate", 0.0001)
            oi_change = market_data.get("oi_change_pct", 0.0)
            vol = market_data.get("volatility", 0.02)  # Daily Vol
            gvol = macro_feed.get("gvol", 0.02)

            # Simple Heuristic Logic for v1 APEX

            # 1. Liquidity Crunch Check
            # High Vol + Negative Funding + Price Crash (implied by arb spread widening usually)
            # v17 Update: Check GVOL spike too
            if (vol > 0.05 or gvol > 0.06) and funding < -0.0005:
                return "LIQ_CRUNCH"

            # 2. Risk-On
            # Positive Funding + Rising OI + Moderate Vol
            if funding > 0.0 and oi_change > 0.01:
                return "RISK_ON"

            # 3. Risk-Off
            # Negative Funding + Dropping OI
            if funding < 0.0 and oi_change < -0.01:
                return "RISK_OFF"

            # 4. Vol Shock
            if vol > 0.06:
                return "VOL_SHOCK"

            return "NEUTRAL"

        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            return "NEUTRAL"
