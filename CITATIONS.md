# Citations & Attributions

## Dataset
- **Customer Support on Twitter (`twcs.csv`)** — Thought Vector, published on Kaggle.
  URL: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
  Approximately 2.81 million customer support tweets across multiple public brand accounts.
  Used under Kaggle's standard dataset terms. Only `@AmazonHelp` tweets retained.

---

## Libraries & Tools

| Library | Version (tested) | Used In | Purpose |
|---|---|---|---|
| `pandas` | 1.5+ | All phases | DataFrame operations, CSV I/O |
| `numpy` | 1.23+ | All phases | Numerical ops, array handling |
| `scikit-learn` | 1.2+ | Phases 4–7, 11–12 | TF-IDF vectorizer, LogisticRegression, KMeans, classification_report, metrics |
| `langdetect` | 1.0.9 | Phase 2–3 | Language detection for English-only filtering (CLD2 model internally) |
| `joblib` | 1.2+ | Phases 6–7 | Model serialization for intent classifier and TF-IDF retrieval index |
| `tqdm` | 4.64+ | Phases 2–7 | Progress bars for large-batch processing |
| `pyyaml` | 6.0+ | configs/ | `config.yaml` loading |
| `json`, `csv`, `re`, `http.server` | stdlib | All phases | Standard library — no attribution required |
| `sentence-transformers` | Scoped to Colab only | Phase 7 (Colab variant) | Dense embedding retrieval (`all-MiniLM-L6-v2`). **Not used in submitted local pipeline** due to missing Visual C++ 2015-2022 redistributable on evaluation machine. All local retrieval uses sparse TF-IDF only. |

> **No PyTorch, TensorFlow, or GPU required** for the submitted local pipeline. All classification and retrieval runs on pure scikit-learn/numpy.

---

## Methodological References

1. **LLM-as-a-Judge evaluation framework**
   Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*
   arXiv:2306.05685. https://arxiv.org/abs/2306.05685
   Used as the design basis for the 4-dimension rubric in `src/judge.py` (Groundedness, Relevance, Safety, Escalation Appropriateness, each scored 1–4).

2. **Retrieval-Augmented Generation (RAG)**
   Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
   NeurIPS 2020. https://arxiv.org/abs/2005.11401
   Our RAG design restricts the retrieval corpus to verified `usable` threads only — a deliberate grounding constraint absent from standard RAG formulations.

3. **Case-Based Reasoning (CBR) for Customer Support**
   Aamodt, A., & Plaza, E. (1994). *Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches.*
   AI Communications, 7(1), 39–59.
   The retriever's "find nearest precedent, adapt resolution" logic is a lightweight CBR instantiation.

4. **Sublinear TF-IDF weighting**
   Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval.*
   Information Processing & Management, 24(5), 513–523.
   `TfidfVectorizer(sublinear_tf=True)` used for both intent classifier and retrieval index.

5. **FAHR (False Auto-Handle Rate) metric**
   Custom metric defined in this project (`src/evaluation.py`). Inspired by standard false-negative rate formulation but adapted to the auto-handle/escalate binary routing decision. Not sourced from a prior publication.

6. **K-Means intent clustering**
   MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations.*
   Proceedings of the 5th Berkeley Symposium. Used in Phase 4–5 for initial problem cluster discovery (`n_clusters=9`).

7. **Langdetect**
   Shuyo Nakatani (2010). *Language Detection Library for Java.*
   Ported to Python. https://github.com/Mimino666/langdetect
   Used in Phase 2–3 to filter non-English threads before quality tiering.

---

## Prompt Engineering Attribution

The grounded reply generation constraints in `src/reply_generator.py` (6 strict behavioral rules: zero unexecuted actions, zero public PII requests, explicit evidence-deficit disclosures) were designed independently based on analysis of AmazonHelp TWCS agent response patterns. No external prompt template library was used.

---

## What Was Not Used

The following tools/APIs were **not** used in the final submitted pipeline, though they may appear in comments or exploratory code:
- `openai` / `google-generativeai` — No LLM API calls in the production pipeline. `src/judge.py` contains an optional LLM judge hook but the Phase 13 evaluation used the deterministic rule-based scorer.
- `sentence-transformers` — Evaluated but excluded from local pipeline (dependency issue). Scoped to Colab only.
- `faiss` / `chromadb` / `weaviate` — No vector database used. All retrieval is sparse TF-IDF.
- `spacy` / `nltk` — No NLP preprocessing libraries. Raw tokenization via TF-IDF analyzer only.
