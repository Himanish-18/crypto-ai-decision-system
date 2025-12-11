
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger("secrets_loader")

def load_secrets():
    """
    Load trading secrets from .env file or environment variables.
    Returns a dictionary with keys:
    - BINANCE_API_KEY
    - BINANCE_API_SECRET
    - BINANCE_TESTNET (bool)
    """
    # Load .env (override=True so .env takes precedence for safety)
    load_dotenv(override=True)
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    # Default to TRUE (Testnet) if not explicitly set to False
    testnet_str = os.getenv("BINANCE_TESTNET", "True").lower()
    is_testnet = testnet_str in ["true", "1", "yes"]
    
    # Validation: Check for placeholders or empty strings
    if api_key and "your_api_key" in api_key:
        api_key = None
    if api_secret and "your_secret_key" in api_secret:
        api_secret = None
    
    if not api_key or not api_secret:
        logger.warning("⚠️ Binance API Keys missing or placeholders detected. Bot will run in Mock/Simulation mode.")
        return {
            "BINANCE_API_KEY": None,
            "BINANCE_API_SECRET": None,
            "BINANCE_TESTNET": True 
        }
        
    return {
        "BINANCE_API_KEY": api_key,
        "BINANCE_API_SECRET": api_secret,
        "BINANCE_TESTNET": is_testnet
    }
