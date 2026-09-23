"""
Phase 6: Baseline Intent Classifier.
Implements a fast, reproducible, and transparent TF-IDF + LogisticRegression classifier
trained on customer queries categorized under the 9-intent taxonomy.
Saves model artifacts, per-intent metrics, and confusion matrix to disk.
"""

import os
import re
import json
import io
import sys
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

sys.path.append(os.path.abspath("src"))
from explore_problems import load_all_customer_queries

# Official 9-intent taxonomy
INTENT_TAXONOMY = {
    "delivery_delay_tracking": {
        "description": "Inquiries regarding late, un-shipped, or in-transit packages.",
        "pattern": re.compile(
            r"\b(where is my order|where is my package|where's my order|late|delay|delayed|transit|dispatch|dispatched|courier|carrier|track|tracking|arriving|estimate|supposed to arrive|still not arrived|one day shipping|two day shipping|same day shipping)\b",
            re.IGNORECASE
        )
    },
    "missing_stolen_delivery": {
        "description": "Packages marked delivered by carrier but missing, handed to wrong person, or stolen.",
        "pattern": re.compile(
            r"\b(handed to resident|says delivered|marked delivered|delivered today but|not delivered|never received|empty box|stolen|porch|where did it go|nowhere to be found)\b",
            re.IGNORECASE
        )
    },
    "damaged_defective_wrong_item": {
        "description": "Damaged, broken, wet, crushed, or completely wrong items delivered.",
        "pattern": re.compile(
            r"\b(damage|damaged|broken|smashed|wet|ruined|defective|faulty|cracked|shattered|torn|wrong item|not what i ordered|sent me)\b",
            re.IGNORECASE
        )
    },
    "return_refund_request": {
        "description": "Return shipping labels, drop-off locations, refund timelines, and status.",
        "pattern": re.compile(
            r"\b(refund|return|returning|exchange|replace|replacement|return label|pass my parcel|send back|drop off|money back)\b",
            re.IGNORECASE
        )
    },
    "prime_membership_billing": {
        "description": "Prime membership charges, renewals, student discounts, cancellation, and fee disputes.",
        "pattern": re.compile(
            r"\b(prime membership|prime member|cancel prime|subscription|charged for prime|annual fee|monthly fee|auto-renew|double charge|unauthorized charge)\b",
            re.IGNORECASE
        )
    },
    "account_access_security": {
        "description": "Login problems, 2FA codes, password resets, compromised/hacked accounts, and spoofing.",
        "pattern": re.compile(
            r"\b(log in|login|sign in|password|2step|two factor|2fa|hacked|compromised|email changed|security|account locked|phishing|spoof)\b",
            re.IGNORECASE
        )
    },
    "digital_devices_technical_support": {
        "description": "Technical support for Echo, Kindle, Fire TV, Prime Video, and Amazon mobile app bugs.",
        "pattern": re.compile(
            r"\b(echo|alexa|kindle|fire tv|firestick|prime video|app|dot|ring|device|streaming|error code|bricked|reboot|unmute)\b",
            re.IGNORECASE
        )
    },
    "promotions_gift_cards_policy": {
        "description": "Beta codes, gift card redemption, promo codes, price guarantee, and affiliate rules.",
        "pattern": re.compile(
            r"\b(gift card|beta code|pre-order|preorder|promo code|promotion|affiliate|cash back|price match|voucher|code)\b",
            re.IGNORECASE
        )
    },
    "other_miscellaneous_feedback": {
        "description": "General customer service complaints, app quizzes, social banter, and out-of-scope feedback.",
        "pattern": re.compile(
            r"\b(customer service|worst customer|apathy|rude|useless|quiz|winner|contest|hate|complaint)\b",
            re.IGNORECASE
        )
    }
}


def label_queries_by_taxonomy(queries):
    """
    Labels queries into the 9-intent taxonomy based on unambiguous pattern matching,
    placing ambiguous or unmatched queries into other_miscellaneous_feedback.
    """
    labeled = []
    for q in queries:
        text = q["cleaned_text"]
        matched_intents = []
        for intent_name, meta in INTENT_TAXONOMY.items():
            if meta["pattern"].search(text):
                matched_intents.append(intent_name)

        if len(matched_intents) == 1:
            q["intent"] = matched_intents[0]
            q["label_confidence"] = 1.0
        elif len(matched_intents) > 1:
            # Priority tie-breaking based on specificity:
            # Physical incident > Missing > Damaged > Account > Billing > Devices > Promo > Delay > Other
            priority_order = [
                "damaged_defective_wrong_item",
                "missing_stolen_delivery",
                "account_access_security",
                "prime_membership_billing",
                "digital_devices_technical_support",
                "return_refund_request",
                "promotions_gift_cards_policy",
                "delivery_delay_tracking",
                "other_miscellaneous_feedback"
            ]
            assigned = next((it for it in priority_order if it in matched_intents), matched_intents[0])
            q["intent"] = assigned
            q["label_confidence"] = 0.8
        else:
            q["intent"] = "other_miscellaneous_feedback"
            q["label_confidence"] = 0.5

        labeled.append(q)

    return labeled


def train_baseline_intent_classifier(queries, model_save_path="data/processed/intent_baseline_model.joblib"):
    """
    Trains a TF-IDF + LogisticRegression baseline classifier.
    Saves model artifacts and returns detailed evaluation metrics.
    """
    corpus = [q["cleaned_text"] for q in queries]
    labels = [q["intent"] for q in queries]

    X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(
        corpus, labels, queries, test_size=0.20, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)

    # Attach prediction to test queries for error inspection
    for q, pred, prob in zip(q_test, y_pred, y_prob):
        q["pred_intent"] = pred
        q["pred_confidence"] = float(np.max(prob))

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    unique_classes = sorted(list(set(labels)))
    report_dict = classification_report(y_test, y_pred, target_names=unique_classes, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred, labels=unique_classes).tolist()

    # Save model pipeline
    pipeline = {"vectorizer": vectorizer, "classifier": clf, "classes": unique_classes}
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)

    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "classes": unique_classes,
        "per_intent_report": report_dict,
        "confusion_matrix": conf_mat,
        "test_size": len(y_test),
        "train_size": len(y_train)
    }

    return pipeline, metrics, q_test


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    usable_path = "data/processed/threads_usable.json"
    partially_path = "data/processed/threads_partially_usable.json"
    queries = load_all_customer_queries(usable_path, partially_path)

    print(f"--- Phase 6: Training Baseline Intent Classifier ({len(queries):,} queries) ---")
    labeled_queries = label_queries_by_taxonomy(queries)

    # Label distribution
    counts = Counter(q["intent"] for q in labeled_queries)
    print("\nTaxonomy Dataset Distribution:")
    for intent, count in counts.most_common():
        pct = (count / len(labeled_queries)) * 100
        print(f"  • {intent:<35}: {count:>5} ({pct:>5.1f}%)")

    model_path = "data/processed/intent_baseline_model.joblib"
    pipeline, metrics, test_samples = train_baseline_intent_classifier(labeled_queries, model_save_path=model_path)

    # Save metrics JSON
    metrics_path = "reports/intent_baseline_metrics.json"
    os.makedirs("reports", exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel pipeline saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"\nOverall Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Macro F1 Score       : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score    : {metrics['weighted_f1']:.4f}")

    print("\n--- Per-Intent Test Breakdown ---")
    for cls_name in metrics["classes"]:
        cls_data = metrics["per_intent_report"][cls_name]
        print(f"• {cls_name:<35} | Prec: {cls_data['precision']:.3f} | Rec: {cls_data['recall']:.3f} | F1: {cls_data['f1-score']:.3f} | Support: {int(cls_data['support'])}")
