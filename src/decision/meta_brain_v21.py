from src.decision.meta_brain_v18 import MetaBrainV18
from src.models.regime_ensemble import RegimeEnsemble
from src.ingest.macro_feeds import MacroFeeds
import logging
import pandas as pd

logger = logging.getLogger("meta_brain_v21")
print("DEBUG: src.decision.meta_brain_v21 MODULE LOADED")
class MetaBrainV21(MetaBrainV18):
    """
    v21 MetaBrain.
    Adds Hard Regime Veto based on Ensemble Risk Score.
    """
    def __init__(self, regime_model_path: str = None):
        super().__init__()
        self.macro_feeds = MacroFeeds()
        self.regime_ensemble = None
        
        if regime_model_path:
            try:
                self.regime_ensemble = RegimeEnsemble.load(regime_model_path)
                logger.info(f"🧠 v21 Regime Ensemble Loaded from {regime_model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Regime Ensemble: {e}")
                self.regime_ensemble = RegimeEnsemble() # Empty init

    def think(self, market_data: dict) -> dict:
        print(f"DEBUG: v21 think called. market_data keys: {market_data.keys()}")
        """
        v21 Intelligence Cycle:
        1. Regine Check (Hard Veto)
        2. v18 Logic (MoE, signals)
        """
        # 1. Regime Veto
        if self.regime_ensemble:
            try:
                # Prepare Features
                # Extract candle data from market_data
                # Assuming market_data["candles"] is the DataFrame, or provided via some key
                candle_data = market_data.get("candles")
                if candle_data is None and isinstance(market_data, pd.DataFrame):
                    candle_data = market_data # Fallback if passed directly in tests
                    
                if candle_data is not None and not candle_data.empty:
                    macro_metrics = self.macro_feeds.fetch_live_metrics()
                    
                    # Helper: derive vol from candle
                    volatility = candle_data["close"].pct_change().std() * 100 
                    
                    # Mock Vector matching training expectation
                    features = pd.DataFrame([{
                        "volatility": volatility,
                        "funding_rate": macro_metrics.get("funding_rate", 0),
                        "iv_index": macro_metrics.get("iv_index", 50)
                    }])
                    
                    risk_score, details = self.regime_ensemble.predict_risk(features)
                    logger.info(f"🧠 v21 Regime Risk: {risk_score:.2f} ({details})")
                    
                    if risk_score > 0.85:
                        logger.warning(f"⛔ REGIME HARD VETO: Risk {risk_score:.2f} > 0.85. Forcing HOLD.")
                        return {
                            "action": "HOLD",
                            "confidence": 1.0,
                            "veto_reason": "REGIME_RISK",
                            "regime_info": details,
                            "agent": "V21_REGIME"
                        }
                    
            except Exception as e:
                logger.error(f"v21 Regime Check Error: {e}")
        
        # 2. Proceed to v18 Logic
        return super().think(market_data)
