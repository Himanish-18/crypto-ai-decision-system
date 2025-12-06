import logging

logger = logging.getLogger("iv_guard")

class IVGuard:
    """
    Risk Gatekeeper based on Implied Volatility.
    Blocks directional trades during extreme IV events (CRITICAL_VOL).
    Authorizes trades ONLY if hedged.
    """
    def __init__(self):
        pass

    def check_trade(self, vol_metrics: dict, is_hedged: bool) -> bool:
        """
        Returns True (Allow) or False (Block).
        """
        if not vol_metrics:
            return True # No data, assume safe (or fail open? usually fail safe, but for sim fail open)
            
        is_critical = vol_metrics.get("is_critical", False)
        crash_prem = vol_metrics.get("crash_premium", 0.0)
        
        if is_critical:
            if is_hedged:
                logger.info(f"🛡️ IV Guard: Critical Vol ({vol_metrics['current_iv']:.2f}) but HEDGED. Allowing trade.")
                return True
            else:
                logger.warning(f"⛔ IV Guard: Critical Vol ({vol_metrics['current_iv']:.2f}) & Unhedged. BLOCKING Trade.")
                return False
                
        # Check Skew Panic (High Crash Premium)
        # If Puts are massively expensive (> 10% skew), block Longs?
        if crash_prem > 0.10: # 10 vol points skew
             logger.warning(f"⛔ IV Guard: Extreme Skew ({crash_prem:.2f}). Blocking Unhedged Directional.")
             return False if not is_hedged else True
             
        return True
