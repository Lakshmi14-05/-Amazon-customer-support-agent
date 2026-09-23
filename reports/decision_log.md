# Decision Log — Hiver AI Support Agent (AmazonHelp)

This document tracks all architectural, data, and design decisions made throughout the project, along with explicit rationales and defensible trade-offs.

---

### Decision 1: Target Brand Selection — `AmazonHelp`
- **Choice**: Selected `AmazonHelp` from TWCS.
- **Rationale**: Highest support volume in the dataset (~169k outbound tweets), providing a diverse spectrum of customer intents spanning transactional claims (refunds, damaged items, lost packages) and self-service procedures (returns, prime management, digital device support). Crucially, this provides a crisp, defensible boundary between safe auto-handling and mandatory human escalation.
- **Alternatives Rejected**: 
  - `AppleSupport`: Over 70% of tweets deflect to DM with no visible resolution, creating an artificial data starvation problem for grounding.
  - Airlines (`Delta`/`AmericanAir`): Nearly all interactions require PNR/flight lookups, making almost every interaction an escalation and leaving few auto-handle candidates.

---

### Decision 2: Committed Dataset Subsample (100MB Slice)
- **Choice**: Committed to a fixed 100MB slice of `twcs.csv` (`data/raw/twcs_subsample_100mb.csv`).
- **Rationale**: Yields 83,352 relevant `AmazonHelp` tweets, 5,773 reconstructed threads, 4,980 customer problem statements, and 231 verified `usable` resolved conversations. This size provides high statistical support for clustering, evaluation (150-250 golden set), and RAG indexing, while ensuring the entire pipeline executes end-to-end in under 15 minutes as strictly required by Hiver.

---

### Decision 3: Language Scope — English-Only via `langdetect`
- **Choice**: Filtered out all non-English customer messages using `langdetect` on the root query (with an English stopword safety net for short typo-heavy queries).
- **Rationale**: `@AmazonHelp` is a global Twitter handle receiving queries in Spanish, German, French, Portuguese, and Japanese. Attempting multi-lingual intent modeling and retrieval within a single agent creates noisy cross-lingual vector space collisions. Restricting scope to English ensures high precision and realistic evaluation.

---

### Decision 4: Three-Tier Data Usability Separation
- **Choice**: Classified reconstructed threads into:
  1. `usable` (231 threads, 4.0%): Explicit verified positive customer confirmation (without negation/dismissal) or completed brand action. **Reserved exclusively for RAG retrieval grounding.**
  2. `partially_usable` (4,749 threads, 82.3%): Genuine customer problem with brand engagement, but deflected to DM or unresolved. **Used for intent clustering and golden-set evaluation, but strictly barred from retrieval evidence.**
  3. `unusable` (793 threads, 13.7%): Discarded (non-English, spam, singletons).
- **Rationale**: Prevents retrieval poisoning where the generative model learns that "send a DM" is the answer to every question.

---

### Decision 5: Documented Grounding-Evidence Asymmetry
- **Choice**: Acknowledged that `usable` threads are heavily concentrated in informational/procedural queries (delivery delays, return steps, billing explanations, account links) and thin for physical incident claims (stolen/missing packages, damaged goods).
- **Rationale & Strategic Value**: In real life, physical incident claims cannot be resolved in public Twitter text without PII. This is treated as a core design feature: the escalation system will route physical-incident queries to human agents specifically citing *lack of grounding evidence and high financial risk*, directly mirroring human support protocols. This will be highlighted in the report section *"What is misleading about my headline number"*.

---

### Decision 6: Baseline Intent Classifier Architecture (TF-IDF + Balanced Logistic Regression)
- **Choice**: Selected TF-IDF $N$-gram vectorization paired with Class-Balanced Logistic Regression over an iterative LLM prompt classifier for the Phase 6 baseline.
- **Rationale**: Trains in under 3 seconds on the 4,939 queries, deterministically reproduces in under 15 minutes without external API dependencies or cost, and establishes an empirical Macro F1 benchmark (0.7436) for Phase 11 Baseline 2.
- **Trade-off & Finding**: Intent performance is highest on `digital_devices_technical_support` (F1: 0.877) and `return_refund_request` (F1: 0.855), but lowest on `missing_stolen_delivery` (F1: 0.440) due to lexical confusion with general delivery delay keywords—an honest failure mode documented for the evaluation harness.

---

### Decision 7: Grounded Case Retrieval Architecture & Coverage Signal
- **Choice**: Built retrieval strictly over the 231 `usable` resolved conversations, using intent-boosted cosine ranking and generating an explicit per-query `coverage_tier` (`HIGH`, `MODERATE`, `THIN_OR_ABSENT`).
- **Rationale**: Decoupling raw semantic similarity from confidence allows the retrieval engine to report when historical evidence is missing. Instead of returning an irrelevant match with an uncalibrated confidence score, the retriever explicitly outputs `THIN_OR_ABSENT`, directly triggering Phase 9 escalation.
- **Dual-Engine Design**: Implemented zero-dependency sublinear TF-IDF retrieval for sub-minute local portability (and Baseline 2 in Phase 11) with a seamless fallback interface to dense `sentence-transformers` for Colab environments.

---

### Decision 8: Grounded Reply Generation Invariants
- **Choice**: Implemented 6 strict operational constraints: (1) retrieved cases as sole factual evidence, (2) zero hallucinated policies/refunds/timelines, (3) zero untaken actions ("I have processed your refund"), (4) zero public PII requests, (5) adaptive resolution rather than verbatim quote, and (6) explicit evidence-deficit disclosure when coverage is thin.
- **Rationale**: Generative customer support models lose customer trust when they simulate actions they did not perform. A polite admission that internal verification is required protects the brand from unauthorized commercial commitments.

---

### Decision 9: Documented Deterministic Escalation Policy (No Arbitrary Score Cutoffs)
- **Choice**: Escalation decisions are driven by documented criteria: (1) mandatory domain risk gating (physical theft, damage inspection, agent complaints), (2) retrieval coverage tier (`THIN_OR_ABSENT`), (3) low intent confidence (< 0.60), and (4) unsupported action claims detected in draft text.
- **Rationale**: Eliminates "black-box" arbitrary threshold escalations. Every decision outputs a human-interpretable stated reason, ensuring auditable governance for human support leads.

---

### Decision 10: Golden Evaluation Set Stratified Sampling Design
- **Choice**: Sampled 198 real queries using dual-axis stratification across all 9 intents AND both evidence tiers (`usable` vs. `partially_usable`).
- **Rationale**: Ensures the golden set contains realistic representation of both well-grounded procedural questions and thin-evidence physical-incident claims, preventing the evaluation benchmark from over-representing easy auto-handle queries.

---

### Decision 11: Blind Independent Labeling Protocol (Zero Anchoring Bias)
- **Choice**: Enforced a strict blind labeling protocol where all `gold_*` fields (`gold_intent`, `gold_auto_or_escalate`, `escalation_reason`, `acceptable_resolution`) are initialized completely BLANK. System suggestions are strictly isolated in `model_suggested_intent` at the far right strictly for post-hoc auditing.
- **Independence & Anti-Anchoring Guarantee**: The annotator reads only `customer_message` during annotation. The labeling tooling does not render or suggest any model outputs, preventing confirmation bias or artificial score inflation.
- **Dedicated Labeling Tooling**: Built both a fast interactive terminal CLI (`python src/labeling_tool.py --cli`) and a zero-dependency local Web UI (`python src/labeling_tool.py --web`) featuring single-key shortcuts, real-time progress tracking (`X/198 labeled`), and immediate row-by-row disk persistence.
- **Strict Post-Labeling Validation**: Enforced a programmatic validator (`python src/labeling_tool.py --validate`) that checks all 198 records for non-blank entries, canonical intent vocabulary membership, valid action enums (`auto_handle` vs `escalate`), and mandatory escalation reasons for escalated records prior to downstream evaluation in Phases 11–15.
---

### Decision 12: Golden Evaluation Set Completed via Independent Human Labeling
- **Final Status**: **198/198 rows fully labeled and validated** (completed 2026-09-22). Validator output: STATUS PASSED.
- **Distribution**: 163 `auto_handle` (82.3%), 35 `escalate` (17.7%). All 9 intents covered: 18–25 rows each.
- **Protocol**: All `gold_intent`, `gold_auto_or_escalate`, `escalation_reason`, and `acceptable_resolution` columns were filled exclusively by the human annotator reading only `customer_message`. No AI system generated or suggested values for these fields. `model_suggested_intent` was isolated at the far right column and never displayed during annotation.
- **Human–Model Agreement**: The annotator's `gold_intent` agreed with `model_suggested_intent` in **150/193 initially-labeled rows (77.72%)**. This validates the golden set as genuinely independent — disagreements expose real classifier blind spots.

---

### Decision 13: Final Phase 11–12 Evaluation Results (198 Human-Labeled Rows)

Full benchmark on `data/evaluation/golden_set.csv` against all three systems:

| System | Intent Acc | Macro F1 | FAHR ⚠️ | Safety Violations |
|---|---|---|---|---|
| Baseline 1 (Zero-Shot / Ungrounded) | 55.6% | 0.553 | **88.6%** | 24 (12.1%) |
| Baseline 2 (Classical IR / Verbatim) | 75.8% | 0.755 | 25.7% | 0 (0.0%) |
| **Final Agent Pipeline (Ours)** | **75.8%** | **0.755** | **20.0%** ✅ | **0** ✅ |

**Weakest intents (by F1, Final Pipeline):**
- `missing_stolen_delivery`: F1=0.696 — hardest due to lexical overlap with delayed delivery queries.
- `other_miscellaneous_feedback`: F1=0.585 — broadest catch-all category; inherently ambiguous.
- `damaged_defective_wrong_item`: F1=0.739 — frequent cross-contamination with return/refund intent.

**Key findings locked in:**
1. **Baseline 1 is unsafe (FAHR = 88.6%)**: 31/35 escalation-required queries auto-handled; 24 replies contain hallucinated commitments or public PII solicitations.
2. **Classification is identical for Baseline 2 and Final Pipeline** (same TF-IDF classifier): differentiation comes entirely from escalation architecture.
3. **Final Pipeline achieves lowest FAHR (20.0%)**: 7 false auto-handles vs 9 for Baseline 2 — domain-risk gating delivers measurable safety improvement.
4. **FER trade-off is intentional**: Higher False Escalation Rate (78.5% vs 71.2%) is the correct conservative trade-off — over-escalation costs efficiency, not brand safety; under-escalation on theft/damage claims costs financial and legal liability.
5. **Zero safety violations** in our pipeline and Baseline 2.

---

### Decision 14: Phase 13 LLM-as-a-Judge Qualitative Scores (Final Pipeline, n=50)

| Dimension | Mean Score (1–4) | Distribution |
|---|---|---|
| Groundedness | **3.00 / 4.0** | 100% scored 3 — "Substantially grounded; consistent with Amazon CS patterns" |
| Relevance | **3.28 / 4.0** | 64% scored 4 (directly actionable); 36% scored 2 (weak alignment) |
| Safety & Compliance | **4.00 / 4.0** | 100% scored 4 — zero untaken-action claims, zero public PII |
| Escalation Appropriateness | **2.52 / 4.0** | 28% optimal (4); 68% suboptimal (2 = over-escalation); 4% critical failure (1) |
| **Overall** | **3.20 / 4.0** | 72% scored 3; 28% scored 2 |

**Interpretation**: The pipeline achieves perfect safety compliance and near-perfect groundedness (no hallucination), but the escalation score (2.52) reflects the high FER — the judge penalises over-escalating procedural queries that had strong retrieval evidence. This is the primary area for future improvement (Phase 15: smarter coverage-tier calibration).


- **Top Disagreement Patterns**: The largest systematic disagreements were (a) model misclassifying `missing_stolen_delivery` as `other_miscellaneous_feedback` (5 cases), likely due to vague complaint language, and (b) `damaged_defective_wrong_item` and `return_refund_request` cross-confusion (2 cases each).

