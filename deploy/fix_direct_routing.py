
import yaml

class FIXRouter:
    """
    v34 Direct FIX Routing Logic.
    Selects the fastest path based on Colo Config.
    """
    def __init__(self, config_path="deploy/colo_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
    def get_route(self, exchange: str):
        # Check NY4
        if exchange in self.config["location"]["primary"]["cross_connects"][0]["exchanges"]:
            return "NY4_DIRECT_FIBER"
            
        # Check LD4
        if exchange in self.config["location"]["secondary"]["cross_connects"][0]["exchanges"]:
            return "LD4_DIRECT_FIBER"
            
        return "INTERNET_VPN_FALLBACK"

if __name__ == "__main__":
    router = FIXRouter()
    print(f"Coinbase Route: {router.get_route('Coinbase_Prime')}")
    print(f"LMAX Route: {router.get_route('LMAX')}")
