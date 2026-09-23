"""
Phase 3: Data Quality & Usability Filtering for Reconstructed Threads (v2).
Upgraded with:
1. True language detection via `langdetect` applied to the customer's first message.
2. Tightened confirmation filter: requires explicit positive confirmation NOT followed
   by negation, dismissal, sarcasm, or abandonment words.
3. Classifies into 'usable', 'partially_usable', and 'unusable'.
"""

import os
import re
import json
from typing import Dict, List, Any, Tuple
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Enforce deterministic language detection
DetectorFactory.seed = 42

# Regex patterns
DM_DEFLECTION_PATTERN = re.compile(
    r"\b(dm|direct message|send us a dm|dm us|message us|reach out via dm|click here to dm|https://t\.co/\w+)\b",
    re.IGNORECASE
)

POSITIVE_CONFIRMATION_PATTERN = re.compile(
    r"\b(thank you|thanks|worked|sorted|fixed|appreciate it|awesome thanks|got it|perfect thanks|all good now|resolved now|that helped)\b",
    re.IGNORECASE
)

NEGATION_DISMISSAL_PATTERN = re.compile(
    r"\b(no thanks|no thank you|but|wont|won't|cannot|cant|can't|elsewhere|waste|done with|useless|still not|still have not|no resolution|not worth|never|unacceptable|fed up|ridiculous|worst|thanks for nothing|thanks a\s*lot|cancel|canceling|cancelling)\b",
    re.IGNORECASE
)

COMMON_ENGLISH_WORDS = {"why", "my", "is", "not", "yet", "the", "order", "what", "where", "how", "have", "you", "your", "can", "help", "please"}

COMPLETED_ACTION_PATTERN = re.compile(
    r"\b(refund processed|replacement sent|ticket #?\w+|issue resolved|credited your account|canceled your order|updated your shipping|fix has been deployed|re-ordered)\b",
    re.IGNORECASE
)

POLICY_INSTRUCTION_PATTERN = re.compile(
    r"\b(go to your account|navigate to|you can return this within|manage your content and devices|turn off auto-renew|change payment method|unmute the microphone|settings >|your orders >|select 'return for refund'|check out our echo help pages)\b",
    re.IGNORECASE
)

SPAM_OR_TRIVIAL_PATTERN = re.compile(
    r"^(@\w+\s*)*(hi|hello|hey|help|\?+|dm|\.|\!+|please help me)$",
    re.IGNORECASE
)


def is_english_text(text: str) -> bool:
    """
    Detects whether the message is English using langdetect.
    Pre-cleans user handles and URLs to avoid false detections.
    """
    cleaned = re.sub(r"@\w+", "", text)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned).strip()

    words = cleaned.split()
    if len(words) < 3:
        # If very short (1-2 words), accept only if pure ASCII
        return all(ord(c) < 128 for c in cleaned)

    try:
        lang = detect(cleaned)
        if lang == "en":
            return True
        # Guard against langdetect falsely calling short typo English text Afrikaans/Somali/Dutch
        cleaned_words = set(cleaned.lower().split())
        overlap = cleaned_words.intersection(COMMON_ENGLISH_WORDS)
        if len(overlap) >= 2:
            return True
        return False
    except LangDetectException:
        # Fallback to ASCII check
        return all(ord(c) < 128 for c in cleaned)


def check_tightened_confirmation(last_customer_text: str) -> bool:
    """
    Asserts an explicit positive confirmation that is NOT negated or dismissed.
    e.g. 'Thanks, it worked!' -> True
    e.g. 'Thanks for trying but I will order elsewhere' -> False
    e.g. 'No resolution provided... Thanks alot' -> False
    """
    text_lower = last_customer_text.lower()
    has_positive = bool(POSITIVE_CONFIRMATION_PATTERN.search(text_lower))
    has_negation = bool(NEGATION_DISMISSAL_PATTERN.search(text_lower))

    # Reject if negation/dismissal is present
    return has_positive and not has_negation


def classify_thread_quality(thread: Dict[str, Any]) -> Tuple[str, str]:
    """
    Classifies a conversation thread into 'usable', 'partially_usable', or 'unusable'
    with an explicit rationale.
    """
    messages = thread.get("messages", [])

    # 1. Structural Checks
    if len(messages) < 2:
        return "unusable", "Orphaned or single-turn tweet; lacks dialog exchange."

    customer_messages = [m for m in messages if m["speaker"] == "customer"]
    brand_messages = [m for m in messages if m["speaker"] == "brand"]

    if not customer_messages or not brand_messages:
        return "unusable", "Missing either customer problem statement or brand response."

    first_customer_text = customer_messages[0]["text"].strip()
    cleaned_problem = re.sub(r"@\w+", "", first_customer_text).strip()

    # 2. Proper Language Detection on Customer's First Message
    if not is_english_text(first_customer_text):
        return "unusable", "Non-English customer query detected via langdetect."

    # 3. Trivial Spam Filter
    if len(cleaned_problem) < 10 or SPAM_OR_TRIVIAL_PATTERN.match(cleaned_problem):
        return "unusable", "Customer query is trivial, spam, or lacks a coherent problem description."

    last_turn = messages[-1]
    last_brand_msg = brand_messages[-1]["text"].strip()
    last_customer_msg = customer_messages[-1]["text"].strip()

    # 4. Check Usable Criteria
    # Tightened confirmation check on customer's closing message
    customer_confirmed = (
        (last_turn["speaker"] == "customer")
        and check_tightened_confirmation(last_customer_msg)
    )

    brand_action_taken = bool(COMPLETED_ACTION_PATTERN.search(last_brand_msg))

    brand_instruction_complete = (
        bool(POLICY_INSTRUCTION_PATTERN.search(last_brand_msg))
        and not bool(DM_DEFLECTION_PATTERN.search(last_brand_msg))
        and len(last_brand_msg) > 50
    )

    is_final_turn_deflection = bool(DM_DEFLECTION_PATTERN.search(last_turn["text"]))

    if (customer_confirmed or brand_action_taken or brand_instruction_complete) and not is_final_turn_deflection:
        return "usable", "Clear customer problem with verified non-negated resolution, explicit confirmation, or completed action."

    # 5. Partially Usable Criteria
    # Clear customer problem with brand interaction, but deflected to DM or unresolved
    if len(cleaned_problem) >= 10:
        if is_final_turn_deflection or bool(DM_DEFLECTION_PATTERN.search(last_brand_msg)):
            return "partially_usable", "Substantive customer issue present, but brand redirected to private DM/link without public resolution."
        elif last_turn["speaker"] == "customer":
            return "partially_usable", "Customer follow-up was left unanswered by the brand."
        else:
            return "partially_usable", "Brand engaged with clarification or partial advice, but outcome remains unconfirmed."

    return "unusable", "Fails usability criteria."
