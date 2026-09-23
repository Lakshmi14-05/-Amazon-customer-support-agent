"""
Phase 8: Grounded Reply Generator.
Receives brand ('AmazonHelp'), incoming customer message, predicted intent, and retrieved usable cases.
Generates an accurate, grounded reply adhering strictly to 6 core safety constraints.
Supports Gemini/OpenAI API via environment variables with deterministic grounded fallback.
"""

import os
import json
import re
from typing import Dict, Any, List


SYSTEM_PROMPT = """You are the official AI customer support assistant for AmazonHelp on Twitter.
Your goal is to draft a helpful, accurate, polite customer support reply.

CRITICAL OPERATIONAL RULES:
1. Grounding: You MUST use the provided historical resolved cases as your sole factual evidence.
2. No Hallucinations: NEVER invent policies, return windows, refund amounts, or delivery dates not backed by the evidence.
3. No Untaken Actions: NEVER claim you have taken an action (e.g., NEVER say "I have issued your refund" or "I cancelled your order"). You do not have direct backend tool execution.
4. Professional & Privacy-Safe: Never ask for or expose sensitive customer information publicly. If an account lookup is required, instruct the customer to use official secure links or contact support through their Amazon account.
5. Adaptive Guidance: Adapt the solution rather than copying verbatim.
6. Evidence Deficit: If the retrieved cases do not contain enough information to safely resolve the customer's problem, explicitly state that you cannot verify the details and advise them on how to reach official support.
"""


def format_user_prompt(customer_message: str, predicted_intent: str, retrieved_cases: List[Dict[str, Any]]) -> str:
    cases_formatted = ""
    for c in retrieved_cases:
        cases_formatted += f"\n--- Reference Case (Thread {c['thread_id']} | Intent: {c['intent']}) ---\n"
        cases_formatted += f"Customer Problem: {c['customer_problem']}\n"
        cases_formatted += f"Verified Amazon Resolution: {c['resolution']}\n"

    prompt = f"""Incoming Customer Message:
\"{customer_message}\"

Predicted Intent: {predicted_intent}

Historical Grounding Evidence (from verified AmazonHelp resolutions):
{cases_formatted}

Draft a public Twitter support reply following all operational rules:"""
    return prompt


def generate_grounded_reply(
    customer_message: str,
    predicted_intent: str,
    retrieved_cases: List[Dict[str, Any]],
    coverage_tier: str = "HIGH"
) -> str:
    """
    Generates a grounded response using Gemini/OpenAI if configured,
    or a deterministic grounded adapter if running in an offline environment.
    """
    # 1. If coverage is THIN_OR_ABSENT, return evidence deficit disclosure
    if coverage_tier == "THIN_OR_ABSENT" or not retrieved_cases:
        return (
            "We apologize for the inconvenience with your order. Because this issue requires direct "
            "verification of your account and carrier details which are not available here, please "
            "connect securely with our Customer Support team via your Amazon account under 'Help' > 'Contact Us'."
        )

    # 2. Check for Gemini API key
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        try:
            import urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            user_prompt = format_user_prompt(customer_message, predicted_intent, retrieved_cases)
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + user_prompt}]}
                ],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200}
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            # Fall back to grounded adapter
            pass

    # 3. Deterministic Grounded Adapter: Adapt resolution from top retrieved case
    top_case = retrieved_cases[0]
    hist_res = top_case["resolution"]

    # Clean rep initials (e.g. ^JM, ^TE) and user handles from historical resolution
    cleaned_res = re.sub(r"\^[A-Z]{2,3}", "", hist_res)
    cleaned_res = re.sub(r"@\w+", "", cleaned_res).strip()

    # Prefix greeting
    if not cleaned_res.lower().startswith(("hi", "hello")):
        draft = f"Hello! {cleaned_res}"
    else:
        draft = cleaned_res

    return draft
