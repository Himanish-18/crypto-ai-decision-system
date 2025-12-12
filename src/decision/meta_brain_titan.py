import logging
from typing import Any, Dict

from src.decision.meta_brain_v17 import MetaBrainV17
from src.ml.meta_label_v2 import LossProbabilityModelV2

logger = logging.getLogger("meta_brain_titan")


class MetaBrainHardVeto(MetaBrainV17):
    """
    v18 TITAN Hard Veto Layer.
    Enforces absolute safety overrides above all agent logic.
    """

    def __init__(self):
        super().__init__()
        self.meta_label_v2 = LossProbabilityModelV2()

    def think(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        TITAN Decision Pipeline:
        1. HARD VETO (Macro)
        2. META-LABEL SUPREMACY (Prob Loss)
        3. AGENT MESH (Standard)
        """
        # 1. HARD VETO CHECK
        regime = self.macro_model.analyze_regime(market_data)
        market_data["regime"] = regime  # Inject for downstream

        if regime in ["LIQ_CRUNCH", "VOL_SHOCK"]:
            logger.critical(
                f"🛑 TITAN HARD VETO ACTIVATED. Regime: {regime}. FORCING CASH."
            )
            return {
                "action": "HOLD",
                "size": 0.0,
                "agent": "TITAN_VETO",
                "reason": f"Hard Veto: {regime}",
            }

        # 2. RUN STANDARD AGENT MESH & GUARDS
        # (This runs ManipulationGuard, ArbEngine, AgentMesh via v17 logic)
        decision = super().think(market_data)

        if decision["action"] == "HOLD":
            return decision

        # 3. META-LABEL SUPREMACY v2
        # Calculate Probability of Loss for the proposed action
        p_loss = self.meta_label_v2.predict_loss_prob(market_data, decision)

        if p_loss > 0.60:
            logger.warning(
                f"🛡️ TITAN META-LABEL VETO. P(Loss) {p_loss:.2f} > 0.60. Blocking Agent {decision['agent']}."
            )
            return {
                "action": "HOLD",
                "size": 0.0,
                "agent": "TITAN_META_LABEL",
                "reason": f"High P(Loss): {p_loss:.2f}",
            }

        return decision
