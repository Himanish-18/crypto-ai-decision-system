import logging
from src.risk_engine.risk_module import RiskEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("risk_test")

def test_risk_engine():
    risk = RiskEngine(account_size=10000)
    
    # 1. Test Volatility Scaling
    # Low Volatility (1%) -> Larger Size
    size_low_vol = risk.calculate_position_size(win_rate=0.6, entry_price=100, volatility=0.01)
    logger.info(f"Size (Low Vol 1%): {size_low_vol:.2f} units")
    
    # High Volatility (15%) -> Smaller Size
    # Target Risk 1% / Vol 15% = 6.67% Allocation (Below 10% Cap)
    size_high_vol = risk.calculate_position_size(win_rate=0.6, entry_price=100, volatility=0.15)
    logger.info(f"Size (High Vol 15%): {size_high_vol:.2f} units")
    
    assert size_high_vol < size_low_vol, f"High volatility should reduce position size. Low: {size_low_vol}, High: {size_high_vol}"
    
    # 2. Test VaR Limit
    # Portfolio Value = 10000, Vol = 2% -> VaR = 10000 * 1.65 * 0.02 = 330
    # Max VaR = 10000 * 0.05 = 500
    # Should Pass
    assert risk.check_var_limit(10000, 0.02) == True
    logger.info("VaR Check (Low Vol): Passed")
    
    # Portfolio Value = 10000, Vol = 10% -> VaR = 1650
    # Should Fail
    assert risk.check_var_limit(10000, 0.10) == False
    logger.info("VaR Check (High Vol): Passed (Rejected)")
    
    logger.info("✅ Risk Engine Hardening Verified.")

if __name__ == "__main__":
    test_risk_engine()
