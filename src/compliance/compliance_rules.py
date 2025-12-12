from typing import Dict, List


class ComplianceRules:
    """
    Hard-coded compliance rules for institutional trading.
    """

    MAX_LEVERAGE = 3.0
    MAX_DAILY_LOSS_PCT = 0.05
    RESTRICTED_JURISDICTIONS = ["North Korea", "Iran", "Gaza Strip", "Syria"]
    MAX_EXPOSURE_PER_ASSET_PCT = 0.40

    @staticmethod
    def check_jurisdiction(user_country: str) -> bool:
        if user_country in ComplianceRules.RESTRICTED_JURISDICTIONS:
            return False
        return True

    @staticmethod
    def check_new_order(
        current_leverage: float, order_value_impact: float, equity: float
    ) -> bool:
        projected_leverage = current_leverage + (order_value_impact / equity)
        if projected_leverage > ComplianceRules.MAX_LEVERAGE:
            return False
        return True

    @staticmethod
    def check_daily_loss(current_pnl_pct: float) -> bool:
        if current_pnl_pct < -ComplianceRules.MAX_DAILY_LOSS_PCT:
            return False  # Halt Trading
        return True
