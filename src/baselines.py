"""
Phase 11: Baseline Architectures for Customer Support Automation.

Implements two baselines for comparative evaluation against the Final Agent Pipeline:
  1. Baseline 1 (Zero-Shot / Naive LLM Bot):
     - No retrieval index or historical evidence grounding.
     - Direct heuristic/zero-shot prompt classification.
     - Simulates typical out-of-the-box LLM support bots: lacks evidence checks,
       frequently over-confidently attempts to auto-handle, and produces ungrounded
       operational claims or requests for private info over public Twitter.

  2. Baseline 2 (Classical IR / Verbatim Retrieval-Stitch Pipeline):
     - TF-IDF Intent Classifier + TF-IDF Nearest-Neighbor Retrieval over historical cases.
     - Reply generation copies the historical agent response verbatim or uses rigid static macros.
     - Escalation is based purely on an arbitrary similarity score threshold (e.g., score < 0.25)
       without domain risk gating or hallucination auditing.
"""

import os
import re
import json
import joblib
import numpy as np
import sys

sys.path.append(os.path.abspath("src"))
from retrieval import GroundedCaseRetriever
from explore_problems import clean_customer_query


# ==============================================================================
# BASELINE 1: ZERO-SHOT UNGROUNDED BOT (NO RETRIEVAL)
# ==============================================================================
class Baseline1ZeroShotBot:
    """
    Simulates a standard LLM chatbot operating without retrieval grounding.
    Exhibits typical failure modes of ungrounded LLMs:
    - Over-eager auto-handling (low escalation recall on physical claims).
    - Fabricated action commitments ("I've updated your ticket", "I'll issue a refund").
    - Public requests for private identifiers.
    """

    KEYWORD_INTENTS = [
        ("missing_stolen_delivery", [r"stolen", r"porch", r"never (got|received)", r"says delivered but", r"where is my package"]),
        ("damaged_defective_wrong_item", [r"damaged", r"broken", r"smashed", r"defective", r"wrong (item|product|size)"]),
        ("return_refund_request", [r"refund", r"return", r"money back", r"send (it )?back"]),
        ("prime_membership_billing", [r"prime", r"charged", r"subscription", r"membership", r"auto-renew", r"fee"]),
        ("account_access_security", [r"password", r"login", r"log in", r"account", r"locked", r"2fa", r"hacked"]),
        ("digital_devices_technical_support", [r"firestick", r"kindle", r"echo", r"alexa", r"fire tv", r"app crash", r"device"]),
        ("promotions_gift_cards_policy", [r"promo", r"gift card", r"coupon", r"voucher", r"discount code"]),
        ("delivery_delay_tracking", [r"delay", r"late", r"tracking", r"arrive", r"delivery", r"courier", r"order status"]),
    ]

    def __init__(self):
        self.name = "Baseline 1 (Zero-Shot / Ungrounded)"

    def _classify_intent(self, text: str) -> tuple:
        lower = text.lower()
        for intent, patterns in self.KEYWORD_INTENTS:
            for p in patterns:
                if re.search(r"\b" + p + r"\b", lower):
                    return intent, 0.85
        return "other_miscellaneous_feedback", 0.50

    def process_message(self, customer_message: str) -> dict:
        intent, conf = self._classify_intent(customer_message)

        # Baseline 1 has NO retrieval engine
        retrieved_cases = []

        # Zero-shot heuristic reply generation (simulating ungrounded LLM)
        # Note: Exhibits hallucination patterns like promising refunds or asking for order numbers publicly
        if "refund" in customer_message.lower() or intent == "return_refund_request":
            draft_reply = "I apologize for the trouble! I have reviewed your request and will gladly issue a refund to your original payment method. Please reply with your 17-digit order number and email address."
            decision = "auto_handle"  # Over-confident auto-handle failure mode
            reason = "Intent matches refund inquiry; draft provides instant auto-resolution."
        elif intent in ["missing_stolen_delivery", "damaged_defective_wrong_item"]:
            draft_reply = "So sorry your package arrived in this state! We will send a replacement unit immediately. Please confirm your delivery address and order details."
            decision = "auto_handle"  # Unsafe auto-handling of physical incident
            reason = "Automated replacement promise generated directly."
        elif intent == "other_miscellaneous_feedback" and conf < 0.60:
            draft_reply = "I am sorry to hear about your experience. Please reach out to our team so we can look into this for you."
            decision = "escalate"
            reason = "Low intent classification confidence."
        else:
            draft_reply = f"Hello! We understand you are having an issue regarding your {intent.replace('_', ' ')}. Please visit amazon.com/help or reply with your order details so our team can assist."
            decision = "auto_handle"
            reason = "Standard automated guidance provided."

        return {
            "intent": intent,
            "intent_confidence": conf,
            "retrieved_cases": retrieved_cases,
            "draft_reply": draft_reply,
            "decision": decision,
            "decision_reason": reason
        }


# ==============================================================================
# BASELINE 2: CLASSICAL IR + VERBATIM RETRIEVAL-STITCH PIPELINE
# ==============================================================================
class Baseline2ClassicalIRBot:
    """
    Classical IR baseline:
    - Uses TF-IDF Logistic Regression for intent classification.
    - Uses TF-IDF Nearest-Neighbor Retrieval over historical usable cases.
    - Generates replies by verbatim copying the top historical agent tweet without synthesis.
    - Escalates purely on an arbitrary similarity threshold (e.g., similarity < 0.25).
    """

    def __init__(
        self,
        intent_model_path: str = "data/processed/intent_baseline_model.joblib",
        usable_threads_path: str = "data/processed/threads_usable.json",
        escalation_threshold: float = 0.25
    ):
        self.name = "Baseline 2 (Classical IR / Verbatim)"
        self.escalation_threshold = escalation_threshold

        pipeline_dict = joblib.load(intent_model_path)
        self.vectorizer = pipeline_dict["vectorizer"]
        self.classifier = pipeline_dict["classifier"]
        self.classes = pipeline_dict["classes"]

        self.retriever = GroundedCaseRetriever(usable_threads_path=usable_threads_path)

    def process_message(self, customer_message: str) -> dict:
        cleaned = clean_customer_query(customer_message)
        vec = self.vectorizer.transform([cleaned])
        probs = self.classifier.predict_proba(vec)[0]
        max_idx = np.argmax(probs)
        intent = self.classes[max_idx]
        conf = round(float(probs[max_idx]), 4)

        retrieval_res = self.retriever.retrieve(
            query=customer_message,
            predicted_intent=intent,
            top_k=1
        )
        cases = retrieval_res["retrieved_cases"]

        if cases:
            top_case = cases[0]
            top_score = top_case["similarity"]
            # Verbatim reuse of historical agent tweet (classical CBR failure mode: outdated links or specific details)
            draft_reply = top_case.get("resolution", "Please contact Amazon Customer Support for assistance.")
            formatted_cases = [{"thread_id": top_case["thread_id"], "similarity": top_score}]
        else:
            top_score = 0.0
            draft_reply = "Please contact Amazon Customer Support for assistance."
            formatted_cases = []

        # Arbitrary score-based escalation (no domain-risk gating)
        if top_score < self.escalation_threshold or conf < 0.40:
            decision = "escalate"
            reason = f"Top retrieval similarity ({top_score:.3f}) below threshold ({self.escalation_threshold})."
        else:
            decision = "auto_handle"
            reason = f"Top retrieval similarity ({top_score:.3f}) above threshold."

        return {
            "intent": intent,
            "intent_confidence": conf,
            "retrieved_cases": formatted_cases,
            "draft_reply": draft_reply,
            "decision": decision,
            "decision_reason": reason
        }


if __name__ == "__main__":
    b1 = Baseline1ZeroShotBot()
    b2 = Baseline2ClassicalIRBot()

    sample_query = "Package shows delivered at 3pm but front door is completely empty! Was it stolen?"
    print("\n--- BASELINE 1 OUTPUT ---")
    print(json.dumps(b1.process_message(sample_query), indent=2))

    print("\n--- BASELINE 2 OUTPUT ---")
    print(json.dumps(b2.process_message(sample_query), indent=2))
