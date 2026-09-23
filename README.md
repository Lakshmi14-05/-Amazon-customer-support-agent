# Hiver AI Support Agent — AmazonHelp

An AI-powered customer-support automation system built on the Kaggle Customer Support on Twitter (`twcs.csv`) dataset, targeting the `@AmazonHelp` brand.

## Core Capabilities
1. **Intent Classification**: 9-category taxonomy derived directly from real `@AmazonHelp` TWCS interactions, trained via class-balanced sublinear TF-IDF + Logistic Regression.
2. **Grounded Reply Generation**: Retrieval-Augmented Generation (RAG) referencing only verified, historically resolved conversations. Zero hallucinated refunds, timelines, or policy promises.
3. **Auto-Handle vs. Escalate Decision**: Deterministic rule-and-evidence-backed escalation with explicit, auditable rationale per query.

## Committed Dataset
- **File**: `data/raw/twcs_subsample_100mb.csv` (100MB, ~83,352 AmazonHelp tweets, 5,773 threads)
- **Usable resolved threads** (RAG index): 231 (`data/processed/threads_usable.json`)
- **Partially usable threads** (classifier training + eval sampling): 4,749

## Quickstart & Reproducibility (Sub-15 Minutes)

### 1. Requirements
```bash
pip install pandas numpy scikit-learn langdetect pyyaml tqdm joblib
```
> No PyTorch required. All retrieval and classification runs on pure scikit-learn/numpy.
> API keys (if using LLM judge): set `GEMINI_API_KEY` or `OPENAI_API_KEY` as environment variables — never hardcode.

### 2. Dataset
Download `twcs.csv` (~530MB) from Kaggle:
```bash
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw/ --unzip
```
Then slice the 100MB subsample:
```bash
python src/run_real_reconstruction_and_sample.py
```

### 3. Full Pipeline — Run in Order

```bash
# Phase 1-3: Data inspection, thread reconstruction, quality filtering
python src/data_inspection.py
# (Rebuilds threads_usable.json and threads_partially_usable.json from the 100MB slice)
python src/analyze_usable_distribution.py

# Phase 4-6: Problem clustering, 9-intent taxonomy, classifier training
python src/intents.py

# Phase 7: Grounded case retrieval engine (builds TF-IDF index over 231 usable cases)
python src/retrieval.py

# Phase 8-9: Grounded reply generator + escalation engine
# (imported by agent_pipeline.py — no standalone run needed)

# Phase 10: Golden evaluation set (198 rows, human-labeled)
# The golden set was manually labeled using a zero-dependency local web tool. 
# Reviewers can verify the labeling environment by spinning it up locally:
python src/labeling_tool.py --web        # Opens UI at http://127.0.0.1:8080 (local machine only)
python src/labeling_tool.py --stats      # check labeling progress
python src/labeling_tool.py --validate   # validate schema before eval

# Phase 11-12: Baseline comparison + evaluation harness
python src/baselines.py                  # verify baselines load correctly
python src/evaluation.py                 # runs all 3 systems on golden set -> reports/

# Phase 13: LLM-as-a-Judge qualitative evaluation
python src/judge.py

# Phase 14: Failure analysis
python src/failure_analysis.py

# Demo: run end-to-end pipeline on 5 test queries
python src/agent_pipeline.py
```

### 4. Key Output Files

| File | Contents |
|---|---|
| `data/evaluation/golden_set.csv` | 198 human-labeled gold rows |
| `reports/evaluation_benchmark_results.json` | Full per-intent + escalation metrics for all 3 systems |
| `reports/failure_analysis_examples.json` | Real false auto-handle and false escalation examples |
| `reports/final_report.md` | Complete 6-section analysis report |
| `reports/decision_log.md` | 14 locked design decisions with rationale |
| `reports/pipeline_inference_demo.json` | Live inference examples on 5 test queries |

### 5. Headline Results (198-row golden set, independent human labels)

| System | Intent Acc | Macro F1 | FAHR (Safety) | Safety Violations |
|---|---|---|---|---|
| Baseline 1 (Zero-Shot) | 55.6% | 0.553 | **88.6%** | 24 (12.1%) |
| Baseline 2 (Classical IR) | 75.8% | 0.755 | 25.7% | 0 |
| **Our Pipeline** | **75.8%** | **0.755** | **20.0%** | **0** |

> **FAHR** = False Auto-Handle Rate: fraction of gold-escalate rows incorrectly auto-handled. The primary safety metric.
