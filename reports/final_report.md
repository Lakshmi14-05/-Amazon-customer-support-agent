# Final Report — Hiver AI Support Agent (AmazonHelp)

## 1. Problem Framing

### What "Good" Means for AmazonHelp Support Automation
For an e-commerce support system operating on public social media, "good" does **not** mean answering every message with a plausible-sounding generative response. A support agent operating in production must satisfy three strict operational invariants:
1. **Zero Hallucinated Commitments**: Never promise refunds, order cancellations, or replacement deliveries when the model lacks backend tool execution to carry them out.
2. **Deterministic Grounding**: When providing procedural advice (how to drop off a return, manage Prime membership, or troubleshoot a device), the instruction must be grounded in historical brand evidence.
3. **Rigorous Escalation**: Accurately detect sensitive, ambiguous, or high-risk claims (stolen goods, account compromises) and route them to human agents with a documented, transparent reason.

### What We Deliberately Chose Not to Build (and Why)
- **Multilingual Support**: `@AmazonHelp` receives tweets in Spanish, German, French, and Portuguese. We deliberately restricted scope to English (`langdetect` filter). Building a multi-lingual system without language-specific grounded retrieval pools leads to vector collisions and lower response accuracy. Documented scope decision in `decision_log.md`.
- **Physical-Incident Full Automation**: We chose not to automate resolutions for missing packages, porch piracy, and damaged merchandise. In the real TWCS dataset, over 95% of these cases are deflected to private DMs because resolving them requires customer PII, carrier GPS logs, and financial authority.

---

## 2. System Architecture & Baseline Comparisons

Our architecture decouples raw semantic retrieval from operational decision-making through four specialized stages:
1. **Intent Classification (`src/intents.py`)**: Class-balanced sublinear TF-IDF + Logistic Regression predicting across 9 TWCS-grounded intents. Trained on `data/processed/threads_partially_usable.json`.
2. **Grounded Case Retriever (`src/retrieval.py`)**: Sublinear TF-IDF n-gram index restricted strictly to the 231 verified `usable` resolved conversations. Evaluates retrieval density to output an explicit `coverage_tier` (`HIGH`, `MODERATE`, `THIN_OR_ABSENT`).
3. **Constrained Reply Generator (`src/reply_generator.py`)**: Generates procedural guidance referencing historical resolutions as sole evidence, enforcing 6 strict behavioral constraints (zero unexecuted actions, zero public PII requests, explicit evidence-deficit disclosures).
4. **Deterministic Escalation Engine (`src/escalation.py`)**: Rule-and-evidence-backed gatekeeper that escalates on: (a) high-risk domain (`missing_stolen_delivery`, `damaged_defective_wrong_item`, `other_miscellaneous_feedback`), (b) coverage `THIN_OR_ABSENT`, (c) intent confidence < 0.60, or (d) unsupported action claims in draft.

### Clarification: Intent Classifier Is Shared Between Baseline 2 and Our Pipeline

**This is an honest and reportable finding, not a deficiency.** Baseline 2 and the Final Agent Pipeline both use the same Phase 6 TF-IDF intent classifier — confirmed empirically (0 mismatches across all 198 rows). They produce identical `intent` and `intent_confidence` outputs for every query.

The entire performance gap between them is attributable to **escalation architecture only**:
- Baseline 2 escalates purely on raw cosine similarity (< 0.25 threshold).
- Our pipeline applies layered deterministic rules: domain-risk gating, coverage-tier checks, confidence thresholds, and draft safety auditing.

This is the correct framing: **grounding and escalation safety improved — not classification accuracy.** Intent classification is a shared component. The thesis of the system is that safer escalation decisions protect the brand from hallucination failures, not that a better classifier alone suffices.

### Baseline Definitions
- **Baseline 1 (Zero-Shot / Ungrounded Bot)**: Keyword intent classification + ungrounded LLM-style replies with no retrieval index and no evidence checks. No safety filtering.
- **Baseline 2 (Classical IR / Verbatim)**: Same TF-IDF intent classifier but verbatim-copies the top retrieved historical agent tweet and escalates purely on a raw similarity threshold (< 0.25) with no domain-risk gating.

### Empirical Results (198 human-labeled golden set rows)

| System | Intent Acc | Macro F1 | FAHR ⚠️ | Safety Violations |
|---|---|---|---|---|
| Baseline 1 (Zero-Shot / Ungrounded) | 55.6% | 0.553 | **88.6%** | 24 (12.1%) |
| Baseline 2 (Classical IR / Verbatim) | 75.8% | 0.755 | 25.7% | 0 |
| **Final Agent Pipeline (Ours)** | **75.8%** | **0.755** | **20.0%** ✅ | **0** ✅ |

**Metric Definitions** (for direct verbal explanation):
- **FAHR (False Auto-Handle Rate)**: Fraction of queries the human annotator marked `escalate` that the system wrongly auto-handled. Formula: false_auto_handles / total_gold_escalate. A FAHR of 20% means 7 of 35 escalation-required queries were incorrectly auto-handled.
- **Safety Violations**: Count of draft replies containing a hallucinated *untaken* action commitment ("I have processed your refund") OR a public request for private account credentials. Measured by deterministic regex audit in `src/evaluation.py`. These are distinct from FAHR: a reply can be safe (no hallucination) but still be sent to the wrong tier (auto_handle vs escalate), and vice versa.

---

## 3. Top 5 Failure Modes with Real Examples

### Failure Mode 1: Persistent Account Security Queries Routed to Auto-Handle (3/7 false auto-handles)

The most common FAHR failure: account compromise and access recovery queries that the escalation engine auto-handled because the historical resolution link looks sufficient, but the human correctly recognized that any account with a changed email or deactivated password requires private authenticated verification — not a generic help URL.

**Real examples:**

- **Thread 522480** — *"Amazon keeps deactivating my password...I changed my password and it is not letting me into my account"* → Pipeline gave a generic help link. Human escalation reason: *requires account authentication & private PII lookup.* Pipeline confidence was 0.97 but coverage was MODERATE — the high confidence fooled the confidence gate.

- **Thread 9129** — *"It looks as though someone has changed the email address linked to my acct & I can no longer log in. Help!"* → Pipeline gave a self-service link. Human escalation reason: *requires account authentication & private PII lookup.* The link is technically correct, but the query implies active account compromise, which should always trigger human review.

- **Thread 320361** — *"I believe somebody has hacked into my account. They've changed my email and password & I can't access it."* → Pipeline auto-handled with a direct link. Human escalation reason: *Account security/possible compromise.* This is the clearest failure: a confirmed account takeover should never be auto-handled.

**Root cause**: The escalation engine's domain-risk gating correctly flags `missing_stolen_delivery` and `damaged_defective_wrong_item`, but does not apply equivalent mandatory escalation for `account_access_security` queries that contain compromise signals (changed email, hacked, locked out). The confidence and coverage checks are insufficient here — the *content* of the query (not just its confidence) should gate escalation.

---

### Failure Mode 2: Prolonged Dispute / Multi-Contact Escalation Missed

Two false auto-handles involved queries where the customer's primary signal was *repeated unresolved contact* — not just the surface problem type.

- **Thread 296765** — *"I ordered something 2 weeks ago paid 56 bucks for prime shipping and I still haven't gotten it???? I want my money back"* → Classified `return_refund_request`. Pipeline auto-handled with a phone number. Human escalation reason: *Prolonged unresolved refund — escalate.* The urgency markers ("2 weeks", "????") and financial dispute ("I want my money back") should trigger escalation.

- **Thread 253981** — *"Hello, I've been trying to return something for over a month. Contacted 3x, informed each time return is approved. Next step?"* → Classified `return_refund_request`. Pipeline auto-handled with a chat link. Human escalation reason: *financial refund or billing transaction dispute.* Three previous contact attempts is a clear escalation signal.

**Root cause**: The classifier correctly labels these as return/refund queries, but the escalation policy has no concept of *escalation history* or *urgency markers*. Adding pattern detection for phrases like "still haven't", "over a month", "contacted X times", or financial amounts above a threshold would correctly route these.

---

### Failure Mode 3: Cross-Domain Complaint Misclassified (`missing_stolen_delivery` — 9/25 misclassified)

The worst-performing intent by misclassification count. Customers frequently combine complaint narratives with requests that superficially match other categories.

- **Thread 550474** — *"it's been two months i did not received any refund as my product did not deliver"* → Classifier predicted `return_refund_request` because "refund" dominates the n-gram signal, but the root cause is a missing delivery. The correct intent is `missing_stolen_delivery`.

- **Thread 442830** — *"Your Leadership advisor named 'Arun' has lied to me. He failed to add the gift card amt of Rs.150 which was promised"* → Classifier predicted `promotions_gift_cards_policy`. Human labeled `missing_stolen_delivery` (high-value item + CS agent dispute). This is a compound complaint that defies single-label classification.

- **Thread 153603** — *"How can I request a delivery fee refund? I was due an item today which has now been delayed"* → Classifier predicted `delivery_delay_tracking`. Human labeled `missing_stolen_delivery`. The refund request for a missing delivery should distinguish from standard delay tracking.

**Root cause**: Single-label classification cannot handle compound complaints. A tweet simultaneously describing a missing item and requesting a refund will be drawn to whichever category has stronger n-gram signal. Partial solution: multi-label classification or a secondary escalation override when any `missing_*` signal phrase appears regardless of top predicted intent.

---

### Failure Mode 4: Catch-All Intent Bleed (`other_miscellaneous_feedback` — 6/18 misclassified)

The catch-all is definitionally diffuse, containing staff compliments, general complaints, and off-topic queries. When any strong domain keyword appears, the classifier anchors to a specific intent.

- **Thread 542087** — *"How can I pass on a compliment about a member of your staff?"* → Classifier predicted `prime_membership_billing` (likely because "member" co-occurs heavily with Prime-related training examples). This is a pure compliment routing issue.

- **Thread 489456** — *"can you please explain why some presents are being delivered with no paper or box"* → Classified `damaged_defective_wrong_item`. The customer is complaining about packaging presentation (a feedback complaint), not a defective product.

- **Thread 149293** — *"Would be super nice if one of your delivery drivers didn't ring my doorbell six times in a row"* → Classified `digital_devices_technical_support`. The word "ring" confused the classifier. This is a driver behaviour complaint, firmly `other_miscellaneous_feedback`.

**Root cause**: The catch-all class has insufficient training density and a diffuse boundary. Domain-specific keywords within a feedback sentence hijack the classifier. Improvement: train a separate binary classifier as a pre-filter ("Is this general feedback?") before the 9-way classifier.

---

### Failure Mode 5: Over-Conservative Coverage-Tier Escalation (128 false escalations, FER = 78.5%)

The largest quantitative failure by count. The pipeline over-escalates because:
1. Low confidence threshold (< 0.60) is triggered on ambiguous but answerable queries.
2. `THIN_OR_ABSENT` coverage fires on queries where the retrieval pool of 231 cases is insufficient.

**Real examples (gold=auto_handle, pipeline escalated):**

- **Thread 26929** — *"Hi, just curious as to why tracking details for my item haven't been updated. Can you help?"* → Escalated due to low confidence (0.39 < 0.60). This is a textbook tracking query that could have been auto-handled with a "check your tracking page" response.

- **Thread 487646** — *"I just started my Amazon/Twitch Prime trial, but the confirmation email was in German and now I can't see my Prime Trial anywhere"* → Escalated due to THIN_OR_ABSENT coverage (similarity 0.11). The query is unusual (cross-language confirmation) but procedurally answerable.

- **Thread 325863** — *"This still hasn't come 2 days after it was supposed to arrive"* → Escalated due to low confidence (0.50 < 0.60). A clear delivery delay query that human labeled `auto_handle`.

**Root cause**: The 0.60 confidence threshold is too conservative for standard procedural queries. The coverage threshold (similarity < 0.20 for THIN_OR_ABSENT) is too strict for short tweets which naturally produce lower cosine similarity. Calibrating these thresholds per-intent — lower thresholds for well-grounded intents like `delivery_delay_tracking` and `return_refund_request` — would reduce FER substantially while maintaining FAHR protection on genuinely risky intents.

---

## 4. "What Is Misleading About My Headline Number?"

### The Grounding Evidence Asymmetry
A naive evaluation of our RAG pipeline might show an impressively high groundedness score (3.00/4.0 across 50 judged cases). However, **this number is partially misleading**:
- In the TWCS dataset, verified public resolutions (`usable` threads) are heavily concentrated in informational and procedural domains (delivery tracking steps, return windows, Prime membership navigation).
- High-risk physical incidents (stolen packages, damaged goods) virtually never conclude with a public solution on Twitter; they are diverted to private DMs.
- Consequently, our high groundedness score is achieved because **the system auto-handles the well-documented procedural queries and appropriately escalates thin-evidence cases to human agents**. If the system were forced to auto-handle physical incidents, groundedness would collapse.
- The Escalation Appropriateness score (2.52/4.0) is the more diagnostic quality signal: it reveals that the system is systematically over-conservative, penalizing itself by escalating answerable queries it should have handled.

### The FER/FAHR Trade-Off Is Asymmetric by Design
The 78.5% False Escalation Rate looks alarming in isolation. But in production:
- A false escalation = a human agent handles a routine tracking question (costs 3–5 minutes of agent time).
- A false auto-handle = a hacked account gets a generic self-service link, or a 2-month billing dispute is met with a phone number (costs customer trust, legal exposure, brand damage).

The asymmetric harm justifies the asymmetric threshold design.

---

## 5. LLM-as-a-Judge Qualitative Scores (Final Pipeline, n=50)

| Dimension | Score / 4.0 | Interpretation |
|---|---|---|
| Groundedness | **3.00** | 100% scored 3 — substantively grounded in CS patterns, no hallucinations |
| Relevance | **3.28** | 64% directly actionable; 36% too generic for the specific complaint |
| Safety & Compliance | **4.00** ✅ | 100% perfect — zero untaken-action claims, zero public PII |
| Escalation Appropriateness | **2.52** | 28% optimal; 68% over-escalated (efficiency loss); 4% critically failed |
| **Overall** | **3.20** | Strong on safety and grounding; calibration of escalation is the primary gap |

---

## 6. What We'd Do with One More Week

1. **Account Compromise Signal Detection**: Add a pre-filter that detects account-compromise signals (changed email, hacked, locked out by third party) in `account_access_security` queries and routes them to mandatory escalation, regardless of confidence or coverage score. This would eliminate 3 of the 7 false auto-handles.

2. **Urgency & Persistence Markers in Escalation Policy**: Add pattern detection for escalation signals embedded in query text: financial amounts ("$56", "Rs.150"), repeat-contact markers ("contacted 3x", "still haven't"), and temporal urgency ("2 weeks", "over a month"). These are reliable escalation signals that the current architecture ignores.

3. **Per-Intent Confidence Threshold Calibration**: Lower the confidence gate threshold (from 0.60) for high-precision intents like `delivery_delay_tracking` and `return_refund_request`, and raise it for inherently ambiguous intents like `other_miscellaneous_feedback`. This would reduce the 128 false escalations substantially without increasing FAHR on risky domains.

4. **Multi-Signal Escalation for Physical Claims**: Replace single-label prediction with a secondary binary classifier that detects physical incident signals regardless of top predicted intent, triggering mandatory escalation for compound complaints that reference both a missing item and a refund.

5. **Expanded Usable Retrieval Pool**: The 231-case retrieval pool is the main source of `THIN_OR_ABSENT` over-escalation. A targeted data collection effort to add 50–100 verified public resolutions for `prime_membership_billing` and `digital_devices_technical_support` would shift coverage tiers from THIN to MODERATE for a broad category of answerable queries.
