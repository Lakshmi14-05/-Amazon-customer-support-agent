"""
Phase 9: Auto-Handle vs. Escalate Decision System.
Evaluates:
1. Intent confidence & ambiguity.
2. Grounded retrieval evidence coverage signal.
3. High-risk/sensitive domain triggers (stolen goods, physical damage, fraud).
4. Unsupported action claims in draft reply (hallucinated refund/cancellation promises).
Outputs a binary decision ('auto_handle' or 'escalate') with a clear, specific stated reason.
"""

import re
from typing import Dict, Any, Tuple


# High-risk intents that mandate human intervention by policy
MANDATORY_ESCALATION_INTENTS = {
    "missing_stolen_delivery",
    "damaged_defective_wrong_item",
    "other_miscellaneous_feedback"
}

# Regex to detect dangerous untaken action claims in draft replies
UNSUPPORTED_ACTION_CLAIMS = re.compile(
    r"\b(i have (issued|processed|cancelled|refunded|credited)|refund has been (issued|processed)|money has been credited|replacement order has been created|replacement has been sent)\b",
    re.IGNORECASE
)


def evaluate_escalation(
    customer_message: str,
    predicted_intent: str,
    intent_confidence: float,
    retrieval_result: Dict[str, Any],
    draft_reply: str
) -> Tuple[str, str]:
    """
    Evaluates whether an incoming message should be auto-handled or escalated.
    Returns (decision, decision_reason).
    """
    # 1. Check for Mandatory High-Risk Domains
    if predicted_intent in MANDATORY_ESCALATION_INTENTS:
        if predicted_intent == "missing_stolen_delivery":
            return (
                "escalate",
                "High-risk physical claim regarding missing/stolen package requiring internal carrier GPS verification and account lookup."
            )
        elif predicted_intent == "damaged_defective_wrong_item":
            return (
                "escalate",
                "Physical damage/wrong item dispute requiring damage inspection, photo verification, and manual replacement approval."
            )
        else:
            return (
                "escalate",
                "Unstructured feedback, high negative sentiment, or complaint lacking standard automated resolution workflow."
            )

    # 2. Check Intent Confidence Threshold
    if intent_confidence < 0.60:
        return (
            "escalate",
            f"Low intent classification confidence ({intent_confidence:.2f} < 0.60); query is ambiguous."
        )

    # 3. Check Retrieval Evidence Coverage
    coverage_tier = retrieval_result.get("coverage_tier", "THIN_OR_ABSENT")
    if coverage_tier == "THIN_OR_ABSENT":
        top_sim = retrieval_result.get("top_similarity", 0.0)
        return (
            "escalate",
            f"Insufficient historical evidence (top similarity: {top_sim:.2f}); lack of grounded resolution cases to guarantee answer correctness."
        )

    # 4. Check for Unsupported Action Claims in Draft Reply
    if UNSUPPORTED_ACTION_CLAIMS.search(draft_reply):
        return (
            "escalate",
            "Proposed reply contains an unsupported claim of completed financial action (e.g. promised refund/cancellation)."
        )

    # 5. Passed all guardrails -> Safe to Auto-Handle
    top_sim = retrieval_result.get("top_similarity", 0.0)
    return (
        "auto_handle",
        f"Confident intent ({predicted_intent}, conf: {intent_confidence:.2f}) with verified historical procedural guidance (similarity: {top_sim:.2f})."
    )
