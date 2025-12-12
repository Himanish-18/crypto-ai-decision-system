class TransactionCostModelV2:
    """
    Advanced TCM.
    Calculates Break-Even requirements and Net Expected Value.
    """

    def __init__(self, maker_fee=0.0002, taker_fee=0.0005):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    def estimate_cost(
        self,
        side: str,
        size: float,
        volatility: float,
        spread: float,
        mode: str = "TAKER",
    ):
        """
        Returns estimated cost in percentage points (e.g., 0.001 for 0.1%).
        """
        # 1. Exchange Fees
        fee = self.maker_fee if mode == "MAKER" else self.taker_fee

        # 2. Slippage / Market Impact
        # Impact ~ k * sqrt(size) * vol
        # Simplified linear approximation for small sizes
        impact = 0.0
        if mode == "TAKER":
            # Taker pays half spread immediately + impact
            impact = (spread / 2.0) + (0.1 * volatility * size)  # Dummy coefficient
        else:
            # Maker earns spread capture but risks adverse selection
            # Cost is primarily "Adverse Selection" risk, not explicit slippage.
            # We model this as a smaller cost probability.
            impact = 0.0

        total_cost = fee + impact
        return total_cost

    def is_profitable(
        self, expected_return: float, cost: float, min_margin: float = 0.0005
    ) -> bool:
        """
        True if ExpReturn > Cost + Margin
        """
        return (expected_return - cost) > min_margin
