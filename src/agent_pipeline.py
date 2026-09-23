"""
End-to-End AI Support Agent Pipeline.
Connects:
- Phase 6 Intent Classifier
- Phase 7 Grounded Case Retriever
- Phase 8 Grounded Reply Generator
- Phase 9 Escalation Decision Engine
Outputs the exact required Hiver per-message JSON schema.
"""

import os
import json
import io
import sys
import numpy as np
import joblib

sys.path.append(os.path.abspath("src"))
from retrieval import GroundedCaseRetriever
from reply_generator import generate_grounded_reply
from escalation import evaluate_escalation
from explore_problems import clean_customer_query


class AmazonSupportAgent:
    def __init__(
        self,
        intent_model_path: str = "data/processed/intent_baseline_model.joblib",
        usable_threads_path: str = "data/processed/threads_usable.json"
    ):
        print("Initializing AmazonSupportAgent...")
        if not os.path.exists(intent_model_path):
            raise FileNotFoundError(f"Intent model not found at {intent_model_path}. Run intents.py first.")

        pipeline_dict = joblib.load(intent_model_path)
        self.vectorizer = pipeline_dict["vectorizer"]
        self.classifier = pipeline_dict["classifier"]
        self.classes = pipeline_dict["classes"]

        self.retriever = GroundedCaseRetriever(usable_threads_path=usable_threads_path)
        print("AmazonSupportAgent ready for inference.")

    def process_message(self, customer_message: str) -> dict:
        """
        Executes end-to-end inference on an incoming customer tweet.
        """
        # 1. Intent Classification
        cleaned = clean_customer_query(customer_message)
        vec = self.vectorizer.transform([cleaned])
        probs = self.classifier.predict_proba(vec)[0]
        max_idx = np.argmax(probs)
        predicted_intent = self.classes[max_idx]
        intent_confidence = round(float(probs[max_idx]), 4)

        # 2. Historical Retrieval over 231 usable cases
        retrieval_res = self.retriever.retrieve(
            query=customer_message,
            predicted_intent=predicted_intent,
            top_k=2
        )

        # 3. Grounded Reply Generation
        draft_reply = generate_grounded_reply(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            retrieved_cases=retrieval_res["retrieved_cases"],
            coverage_tier=retrieval_res["coverage_tier"]
        )

        # 4. Auto-Handle vs Escalate Decision
        decision, decision_reason = evaluate_escalation(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            retrieval_result=retrieval_res,
            draft_reply=draft_reply
        )

        # 5. Format to exact required schema
        formatted_cases = [
            {
                "thread_id": c["thread_id"],
                "similarity": c["similarity"]
            }
            for c in retrieval_res["retrieved_cases"]
        ]

        output = {
            "intent": predicted_intent,
            "intent_confidence": intent_confidence,
            "retrieved_cases": formatted_cases,
            "draft_reply": draft_reply,
            "decision": decision,
            "decision_reason": decision_reason
        }
        return output


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    agent = AmazonSupportAgent()

    test_queries = [
        "Where do I send my package now that it is ready for return and refund? No return instructions in email.",
        "Tracking says delivered to resident 2 hours ago, but nobody was home and porch is empty! Was it stolen??",
        "The ceramic coffee mug arrived in a thousand broken pieces completely smashed in the envelope!",
        "How do I cancel auto-renew on my Prime membership so I do not get charged next month?",
        "Your customer service is completely useless and rude. No one helps me!"
    ]

    print("\n=== PIPELINE INFERENCE DEMO (EXACT HIVER OUTPUT SCHEMA) ===\n")
    results = []
    for q in test_queries:
        res = agent.process_message(q)
        results.append({"query": q, "response": res})
        print(f"Customer Message: \"{q}\"")
        print(json.dumps(res, indent=2))
        print("-" * 60)

    # Save demo results
    demo_path = "reports/pipeline_inference_demo.json"
    with open(demo_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved demo outputs to {demo_path}")
