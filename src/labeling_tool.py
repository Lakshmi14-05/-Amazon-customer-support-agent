"""
Phase 10: Golden Evaluation Set Sampler & Independent Human Labeling Tool.

Provides:
  1. Dataset generator (dual-axis stratification across 9 intents x 2 evidence tiers)
  2. Fast Interactive CLI labeling tool with incremental saving
  3. Zero-dependency Local Web UI for rapid point-and-click / keyboard shortcut labeling
  4. Real-time progress tracking ('X/198 labeled')
  5. Strict post-labeling schema validation (--validate)

Strict Invariants:
  - All gold fields (gold_intent, gold_auto_or_escalate, escalation_reason, acceptable_resolution)
    are initialized completely BLANK.
  - The tool displays ONLY `customer_message` during annotation to eliminate anchoring bias.
  - Model suggestions are stored strictly in `model_suggested_intent` for post-hoc auditing.
"""

import os
import csv
import json
import random
import sys
import argparse
from collections import defaultdict, Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

GOLDEN_CSV_PATH = os.path.abspath("data/evaluation/golden_set.csv")
SAMPLING_LOG_PATH = os.path.abspath("data/evaluation/sampling_log.json")

CANONICAL_INTENTS = [
    "delivery_delay_tracking",
    "missing_stolen_delivery",
    "damaged_defective_wrong_item",
    "return_refund_request",
    "prime_membership_billing",
    "account_access_security",
    "digital_devices_technical_support",
    "promotions_gift_cards_policy",
    "other_miscellaneous_feedback"
]

FIELDNAMES = [
    "thread_id", "tweet_id", "customer_message",
    "gold_intent", "gold_auto_or_escalate", "escalation_reason",
    "acceptable_resolution",
    "retrieval_relevance_flag", "groundedness_flag", "unsupported_claims_flag",
    "evidence_tier", "model_suggested_intent"
]

COMMON_ESCALATION_REASONS = [
    "physical incident / carrier claim required",
    "requires account authentication & private PII lookup",
    "financial refund or billing transaction dispute",
    "customer dissatisfaction / complaint regarding previous agent",
    "complex technical issue / off-policy / unresolvable without human"
]


# ==============================================================================
# 1. SAMPLING & GENERATION
# ==============================================================================
def generate_blind_golden_set(
    usable_path="data/processed/threads_usable.json",
    partially_path="data/processed/threads_partially_usable.json",
    target_total=198,
    random_seed=42
):
    sys.path.append(os.path.abspath("src"))
    from explore_problems import load_all_customer_queries
    from intents import INTENT_TAXONOMY, label_queries_by_taxonomy

    random.seed(random_seed)
    queries = load_all_customer_queries(usable_path, partially_path)
    labeled = label_queries_by_taxonomy(queries)

    strata = defaultdict(list)
    for q in labeled:
        key = (q["intent"], q["tier"])
        strata[key].append(q)

    intents = CANONICAL_INTENTS
    sampled_records = []
    sampling_log = {
        "random_seed": random_seed,
        "target_total": target_total,
        "sampling_method": "Stratified random sampling across 9 candidate intent clusters and 2 evidence tiers (usable vs partially_usable).",
        "labeling_protocol": "Blind independent human labeling. All gold_* fields are initialized BLANK. The annotator reads only customer_message to prevent anchoring bias.",
        "strata_counts": {},
        "allocated_counts": {}
    }

    per_intent_target = max(15, target_total // len(intents))

    for it in intents:
        usable_pool = strata.get((it, "usable"), [])
        partially_pool = strata.get((it, "partially_usable"), [])

        sampling_log["strata_counts"][it] = {
            "usable_available": len(usable_pool),
            "partially_available": len(partially_pool)
        }

        usable_target = min(len(usable_pool), max(3, int(per_intent_target * 0.35)))
        partially_target = min(len(partially_pool), per_intent_target - usable_target)

        chosen_usable = random.sample(usable_pool, usable_target) if usable_pool else []
        chosen_partially = random.sample(partially_pool, partially_target) if partially_pool else []

        for q in chosen_usable + chosen_partially:
            sampled_records.append({
                "thread_id": q["thread_id"],
                "tweet_id": q["tweet_id"],
                "customer_message": q["raw_text"],
                "gold_intent": "",
                "gold_auto_or_escalate": "",
                "escalation_reason": "",
                "acceptable_resolution": "",
                "retrieval_relevance_flag": "",
                "groundedness_flag": "",
                "unsupported_claims_flag": "",
                "evidence_tier": q["tier"],
                "model_suggested_intent": q["intent"]
            })

        sampling_log["allocated_counts"][it] = len(chosen_usable) + len(chosen_partially)

    random.shuffle(sampled_records)
    sampled_records = sampled_records[:target_total]
    sampling_log["actual_sampled_total"] = len(sampled_records)

    os.makedirs(os.path.dirname(SAMPLING_LOG_PATH), exist_ok=True)
    with open(SAMPLING_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(sampling_log, f, indent=2)

    save_golden_set(sampled_records)
    print(f"Generated clean blind golden set ({len(sampled_records)} rows) -> {GOLDEN_CSV_PATH}")
    return sampled_records


# ==============================================================================
# 2. IO & PERSISTENCE
# ==============================================================================
def load_golden_set():
    if not os.path.exists(GOLDEN_CSV_PATH):
        raise FileNotFoundError(f"Golden set not found at {GOLDEN_CSV_PATH}. Run --generate first.")
    with open(GOLDEN_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_golden_set(rows):
    os.makedirs(os.path.dirname(GOLDEN_CSV_PATH), exist_ok=True)
    with open(GOLDEN_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def is_row_labeled(row):
    return bool(row.get("gold_intent") and row.get("gold_auto_or_escalate"))


def get_progress_stats(rows=None):
    if rows is None:
        rows = load_golden_set()
    total = len(rows)
    labeled = sum(1 for r in rows if is_row_labeled(r))
    intents_count = Counter(r["gold_intent"] for r in rows if r.get("gold_intent"))
    actions_count = Counter(r["gold_auto_or_escalate"] for r in rows if r.get("gold_auto_or_escalate"))
    pct = (labeled / total * 100.0) if total > 0 else 0.0
    return {
        "total": total,
        "labeled": labeled,
        "remaining": total - labeled,
        "percentage": round(pct, 1),
        "intents": dict(intents_count),
        "actions": dict(actions_count)
    }


# ==============================================================================
# 3. CLI INTERACTIVE LABELING TOOL
# ==============================================================================
def run_cli_labeling():
    rows = load_golden_set()
    total = len(rows)

    print("=" * 80)
    print(" HIVER SUPPORT AGENT - BLIND GOLDEN SET LABELING TOOL")
    print("=" * 80)
    print("Protocol:")
    print("  - You will see ONLY the customer message.")
    print("  - Assign gold_intent, gold_auto_or_escalate, escalation_reason, acceptable_resolution.")
    print("  - All progress is saved immediately and incrementally to disk.")
    print("Commands:")
    print("  - 's' to skip/next, 'b' to go back, 'j <num>' to jump to row, 'q' to quit.")
    print("=" * 80)

    idx = 0
    # Find first unlabeled row
    for i, r in enumerate(rows):
        if not is_row_labeled(r):
            idx = i
            break

    while 0 <= idx < total:
        row = rows[idx]
        stats = get_progress_stats(rows)

        print("\n" + "-" * 80)
        status_tag = "[LABELED]" if is_row_labeled(row) else "[UNLABELED]"
        print(f"Row {idx + 1} of {total}  {status_tag} | Overall Progress: {stats['labeled']}/{total} ({stats['percentage']}%)")
        print("-" * 80)
        print(f"CUSTOMER MESSAGE:\n  \"{row['customer_message']}\"\n")
        if is_row_labeled(row):
            print(f"Current Labels:")
            print(f"  Intent:        {row['gold_intent']}")
            print(f"  Action:        {row['gold_auto_or_escalate']}")
            print(f"  Reason:        {row['escalation_reason']}")
            print(f"  Resolution:    {row['acceptable_resolution']}")
            print("-" * 80)

        print("Select Intent:")
        for i, it in enumerate(CANONICAL_INTENTS, 1):
            print(f"  [{i}] {it}")

        choice = input(f"\nEnter Intent [1-9], 's' (skip), 'b' (back), 'j <N>' (jump), 'q' (quit) [current: {row.get('gold_intent') or 'None'}]: ").strip()

        if choice.lower() == 'q':
            print("\nProgress saved. Exiting labeling tool.")
            break
        elif choice.lower() == 's':
            idx += 1
            continue
        elif choice.lower() == 'b':
            idx = max(0, idx - 1)
            continue
        elif choice.lower().startswith('j'):
            parts = choice.split()
            if len(parts) > 1 and parts[1].isdigit():
                target_idx = int(parts[1]) - 1
                if 0 <= target_idx < total:
                    idx = target_idx
                    continue
            print("Invalid row number.")
            continue
        elif not choice and row.get("gold_intent"):
            selected_intent = row["gold_intent"]
        elif choice.isdigit() and 1 <= int(choice) <= 9:
            selected_intent = CANONICAL_INTENTS[int(choice) - 1]
        else:
            print("Invalid input. Please enter a number 1-9.")
            continue

        # Select Action
        print("\nSelect Action:")
        print("  [1] auto_handle")
        print("  [2] escalate")
        curr_act_num = "1" if row.get("gold_auto_or_escalate") == "auto_handle" else ("2" if row.get("gold_auto_or_escalate") == "escalate" else "")
        act_prompt = f"Action [1/2] (current: {row.get('gold_auto_or_escalate') or 'None'}): "
        act_choice = input(act_prompt).strip()

        if not act_choice and row.get("gold_auto_or_escalate"):
            selected_action = row["gold_auto_or_escalate"]
        elif act_choice == "1":
            selected_action = "auto_handle"
        elif act_choice == "2":
            selected_action = "escalate"
        else:
            print("Invalid action. Defaulting to escalate.")
            selected_action = "escalate"

        # Escalation reason
        escalation_reason = ""
        if selected_action == "escalate":
            print("\nSelect / Type Escalation Reason:")
            for r_idx, r_text in enumerate(COMMON_ESCALATION_REASONS, 1):
                print(f"  [{r_idx}] {r_text}")
            r_input = input(f"Reason [1-5 or custom text] [current: {row.get('escalation_reason') or 'None'}]: ").strip()
            if not r_input and row.get("escalation_reason"):
                escalation_reason = row["escalation_reason"]
            elif r_input.isdigit() and 1 <= int(r_input) <= len(COMMON_ESCALATION_REASONS):
                escalation_reason = COMMON_ESCALATION_REASONS[int(r_input) - 1]
            elif r_input:
                escalation_reason = r_input
            else:
                escalation_reason = "human review required"

        # Acceptable resolution
        res_prompt = f"Acceptable Resolution note (optional) [current: {row.get('acceptable_resolution') or ''}]: "
        res_input = input(res_prompt).strip()
        acceptable_res = res_input if res_input else row.get("acceptable_resolution", "")

        # Commit update
        row["gold_intent"] = selected_intent
        row["gold_auto_or_escalate"] = selected_action
        row["escalation_reason"] = escalation_reason
        row["acceptable_resolution"] = acceptable_res

        save_golden_set(rows)
        print(f"-> Saved Row {idx + 1}!")
        idx += 1

    final_stats = get_progress_stats(rows)
    print("\n" + "=" * 80)
    print(f"Session Finished! Total Labeled: {final_stats['labeled']}/{final_stats['total']} ({final_stats['percentage']}%)")
    print("=" * 80)


# ==============================================================================
# 4. LIGHTWEIGHT ZERO-DEPENDENCY WEB UI
# ==============================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hiver Support Agent - Golden Set Labeling</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --border: #334155;
    --btn-bg: #334155;
    --btn-hover: #475569;
    --escalate: #f43f5e;
    --autohandle: #10b981;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg); color: var(--text); padding: 24px; display: flex; justify-content: center; }
  .container { max-width: 900px; width: 100%; display: flex; flex-direction: column; gap: 20px; }
  
  /* Header */
  .header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
  .header h1 { font-size: 1.3rem; font-weight: 600; color: var(--accent); }
  .progress-wrap { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }
  .progress-bar-bg { width: 220px; height: 10px; background: var(--card); border-radius: 5px; overflow: hidden; border: 1px solid var(--border); }
  .progress-bar-fill { height: 100%; width: 0%; background: var(--accent); transition: width 0.3s; }
  .progress-text { font-size: 0.85rem; color: var(--muted); }

  /* Message Card */
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .card-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 8px; }
  .customer-msg { font-size: 1.15rem; line-height: 1.6; color: #ffffff; background: #0b1120; padding: 18px; border-radius: 8px; border: 1px solid #1e293b; min-height: 80px; word-break: break-word; }

  /* Form */
  .section-title { font-size: 0.9rem; font-weight: 600; color: var(--accent); margin-bottom: 10px; margin-top: 15px; }
  .intent-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .intent-btn { background: var(--btn-bg); border: 1px solid var(--border); color: var(--text); padding: 10px 12px; border-radius: 6px; font-size: 0.82rem; cursor: pointer; text-align: left; transition: all 0.15s; }
  .intent-btn:hover { background: var(--btn-hover); }
  .intent-btn.selected { background: #0284c7; border-color: #38bdf8; font-weight: bold; }
  .shortcut { opacity: 0.6; font-size: 0.75rem; margin-right: 4px; }

  .action-wrap { display: flex; gap: 12px; }
  .action-btn { flex: 1; padding: 12px; border-radius: 6px; font-size: 0.95rem; font-weight: 600; border: 1px solid var(--border); cursor: pointer; background: var(--btn-bg); color: var(--text); text-align: center; }
  .action-btn.autohandle.selected { background: var(--autohandle); border-color: #34d399; color: #fff; }
  .action-btn.escalate.selected { background: var(--escalate); border-color: #fb7185; color: #fff; }

  .reasons-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .reason-pill { background: #1e293b; border: 1px solid var(--border); font-size: 0.75rem; padding: 4px 8px; border-radius: 4px; cursor: pointer; color: var(--muted); }
  .reason-pill:hover { border-color: var(--accent); color: var(--text); }
  input[type="text"] { width: 100%; padding: 10px; background: #0b1120; border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 0.9rem; margin-top: 6px; }

  /* Navigation */
  .footer-nav { display: flex; justify-content: space-between; align-items: center; margin-top: 20px; }
  .nav-btn { background: var(--btn-bg); border: 1px solid var(--border); color: var(--text); padding: 10px 18px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; }
  .nav-btn:hover { background: var(--btn-hover); }
  .save-btn { background: #0284c7; border-color: #38bdf8; color: white; font-weight: bold; padding: 12px 28px; }
  .save-btn:hover { background: #0369a1; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>Hiver Support Agent — Golden Set Labeler</h1>
      <span style="font-size:0.8rem; color:var(--muted);">Blind independent labeling protocol (reading customer message only)</span>
    </div>
    <div class="progress-wrap">
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="p-bar"></div></div>
      <div class="progress-text" id="p-text">Loading...</div>
    </div>
  </div>

  <div class="card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div class="card-label" id="row-indicator">Row #...</div>
      <button class="nav-btn" style="font-size:0.75rem; padding:4px 8px;" onclick="jumpNextUnlabeled()">Jump to Next Unlabeled</button>
    </div>
    <div class="customer-msg" id="msg-text">Loading...</div>

    <div class="section-title">1. Intent Classification (Keyboard: 1-9)</div>
    <div class="intent-grid" id="intent-container"></div>

    <div class="section-title">2. Action Decision (Keyboard: A for Auto-Handle, E for Escalate)</div>
    <div class="action-wrap">
      <button class="action-btn autohandle" id="btn-auto" onclick="setAction('auto_handle')">Auto-Handle [A]</button>
      <button class="action-btn escalate" id="btn-esc" onclick="setAction('escalate')">Escalate to Human [E]</button>
    </div>

    <div id="esc-group" style="margin-top:12px; display:none;">
      <div class="section-title" style="margin-top:0;">3. Escalation Rationale</div>
      <div class="reasons-wrap" id="preset-reasons"></div>
      <input type="text" id="input-reason" placeholder="Type or click a common reason above...">
    </div>

    <div style="margin-top:12px;">
      <div class="section-title" style="margin-top:0;">4. Acceptable Resolution Note (Optional)</div>
      <input type="text" id="input-resolution" placeholder="e.g., Guide customer to returns portal / Inform tracking ETA / Advise contact carrier">
    </div>
  </div>

  <div class="footer-nav">
    <button class="nav-btn" onclick="prevRow()">&larr; Previous [Left]</button>
    <button class="nav-btn save-btn" onclick="saveAndNext()">Save & Next [Enter] &rarr;</button>
    <button class="nav-btn" onclick="nextRow()">Skip / Next [Right] &rarr;</button>
  </div>
</div>

<script>
const INTENTS = [
  "delivery_delay_tracking",
  "missing_stolen_delivery",
  "damaged_defective_wrong_item",
  "return_refund_request",
  "prime_membership_billing",
  "account_access_security",
  "digital_devices_technical_support",
  "promotions_gift_cards_policy",
  "other_miscellaneous_feedback"
];

const REASONS = [
  "physical incident / carrier claim required",
  "requires account authentication & private PII lookup",
  "financial refund or billing transaction dispute",
  "customer dissatisfaction / complaint regarding previous agent",
  "complex technical issue / off-policy / unresolvable without human"
];

let currentIndex = 0;
let totalRows = 0;
let currentRow = {};

function init() {
  const container = document.getElementById("intent-container");
  container.innerHTML = "";
  INTENTS.forEach((it, idx) => {
    const btn = document.createElement("button");
    btn.className = "intent-btn";
    btn.id = "intent-btn-" + (idx + 1);
    btn.innerHTML = `<span class="shortcut">[${idx + 1}]</span> ${it}`;
    btn.onclick = () => selectIntent(it);
    container.appendChild(btn);
  });

  const rWrap = document.getElementById("preset-reasons");
  rWrap.innerHTML = "";
  REASONS.forEach(r => {
    const pill = document.createElement("div");
    pill.className = "reason-pill";
    pill.innerText = r;
    pill.onclick = () => { document.getElementById("input-reason").value = r; };
    rWrap.appendChild(pill);
  });

  loadRow(0);
}

async function loadRow(idx) {
  const res = await fetch(`/api/get?idx=${idx}`);
  const data = await res.json();
  currentIndex = data.index;
  totalRows = data.total;
  currentRow = data.row;

  document.getElementById("row-indicator").innerText = `Row ${currentIndex + 1} of ${totalRows}  (ID: ${currentRow.thread_id}) ${currentRow.gold_intent ? "✓ Labeled" : "⏳ Unlabeled"}`;
  document.getElementById("msg-text").innerText = currentRow.customer_message;

  // render intent
  selectIntent(currentRow.gold_intent, false);
  // render action
  setAction(currentRow.gold_auto_or_escalate, false);
  // inputs
  document.getElementById("input-reason").value = currentRow.escalation_reason || "";
  document.getElementById("input-resolution").value = currentRow.acceptable_resolution || "";

  updateStats(data.stats);
}

function selectIntent(it, dirty=true) {
  currentRow.gold_intent = it || "";
  INTENTS.forEach((intentName, idx) => {
    const el = document.getElementById("intent-btn-" + (idx + 1));
    if (intentName === it) {
      el.classList.add("selected");
    } else {
      el.classList.remove("selected");
    }
  });
}

function setAction(act, dirty=true) {
  currentRow.gold_auto_or_escalate = act || "";
  const btnAuto = document.getElementById("btn-auto");
  const btnEsc = document.getElementById("btn-esc");
  const escGroup = document.getElementById("esc-group");

  btnAuto.classList.toggle("selected", act === "auto_handle");
  btnEsc.classList.toggle("selected", act === "escalate");

  if (act === "escalate") {
    escGroup.style.display = "block";
  } else {
    escGroup.style.display = "none";
  }
}

function updateStats(stats) {
  if (!stats) return;
  document.getElementById("p-bar").style.width = stats.percentage + "%";
  document.getElementById("p-text").innerText = `${stats.labeled} / ${stats.total} labeled (${stats.percentage}%)`;
}

async function saveAndNext() {
  const body = {
    index: currentIndex,
    gold_intent: currentRow.gold_intent,
    gold_auto_or_escalate: currentRow.gold_auto_or_escalate,
    escalation_reason: document.getElementById("input-reason").value.trim(),
    acceptable_resolution: document.getElementById("input-resolution").value.trim()
  };

  const res = await fetch("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  const data = await res.json();
  updateStats(data.stats);

  if (currentIndex < totalRows - 1) {
    loadRow(currentIndex + 1);
  } else {
    alert("All items completed! Run `python src/labeling_tool.py --validate` to verify your dataset.");
  }
}

function prevRow() {
  if (currentIndex > 0) loadRow(currentIndex - 1);
}

function nextRow() {
  if (currentIndex < totalRows - 1) loadRow(currentIndex + 1);
}

async function jumpNextUnlabeled() {
  const res = await fetch(`/api/next_unlabeled?from=${currentIndex}`);
  const data = await res.json();
  if (data.index !== -1) {
    loadRow(data.index);
  } else {
    alert("No more unlabeled items found!");
  }
}

document.addEventListener("keydown", (e) => {
  if (["input", "textarea"].includes(e.target.tagName.toLowerCase())) {
    if (e.key === "Enter") {
      saveAndNext();
    }
    return;
  }
  if (e.key >= "1" && e.key <= "9") {
    selectIntent(INTENTS[parseInt(e.key) - 1]);
  } else if (e.key.toLowerCase() === "a") {
    setAction("auto_handle");
  } else if (e.key.toLowerCase() === "e") {
    setAction("escalate");
  } else if (e.key === "ArrowLeft") {
    prevRow();
  } else if (e.key === "ArrowRight") {
    nextRow();
  } else if (e.key === "Enter") {
    saveAndNext();
  }
});

window.onload = init;
</script>
</body>
</html>
"""


class LabelingServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        elif parsed.path == "/api/get":
            rows = load_golden_set()
            idx = int(qs.get("idx", [0])[0])
            idx = max(0, min(len(rows) - 1, idx))
            stats = get_progress_stats(rows)
            resp = {
                "index": idx,
                "total": len(rows),
                "row": rows[idx],
                "stats": stats
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed.path == "/api/next_unlabeled":
            rows = load_golden_set()
            start_from = int(qs.get("from", [0])[0]) + 1
            target_idx = -1
            # search from start_from to end
            for i in range(start_from, len(rows)):
                if not is_row_labeled(rows[i]):
                    target_idx = i
                    break
            # wrap around
            if target_idx == -1:
                for i in range(0, start_from):
                    if not is_row_labeled(rows[i]):
                        target_idx = i
                        break
            resp = {"index": target_idx}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/save":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len)
            data = json.loads(post_body.decode("utf-8"))

            rows = load_golden_set()
            idx = int(data["index"])
            if 0 <= idx < len(rows):
                rows[idx]["gold_intent"] = data.get("gold_intent", "")
                rows[idx]["gold_auto_or_escalate"] = data.get("gold_auto_or_escalate", "")
                rows[idx]["escalation_reason"] = data.get("escalation_reason", "")
                rows[idx]["acceptable_resolution"] = data.get("acceptable_resolution", "")
                save_golden_set(rows)

            stats = get_progress_stats(rows)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "stats": stats}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence standard HTTP logs for clean console
        pass


def run_web_server(port=8080):
    server = HTTPServer(("127.0.0.1", port), LabelingServerHandler)
    stats = get_progress_stats()
    print("=" * 80)
    print(f" HIVER SUPPORT AGENT - WEB LABELING INTERFACE STARTED")
    print("=" * 80)
    print(f" URL: http://127.0.0.1:{port}")
    print(f" Current Progress: {stats['labeled']}/{stats['total']} labeled ({stats['percentage']}%)")
    print(" Features:")
    print("   - Point-and-click or keyboard shortcuts (1-9 for Intent, A/E for Action)")
    print("   - Real-time incremental auto-saving to data/evaluation/golden_set.csv")
    print("   - Press Ctrl+C in this terminal when finished.")
    print("=" * 80)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWeb server stopped.")


# ==============================================================================
# 5. POST-LABELING VALIDATION SUITE
# ==============================================================================
def validate_golden_set(expected_count=198):
    if not os.path.exists(GOLDEN_CSV_PATH):
        print(f"ERROR: {GOLDEN_CSV_PATH} does not exist.")
        return False

    rows = load_golden_set()
    errors = []

    if len(rows) != expected_count:
        errors.append(f"Row count mismatch: expected {expected_count}, got {len(rows)}.")

    intent_counts = Counter()
    action_counts = Counter()

    for idx, r in enumerate(rows, 1):
        g_intent = (r.get("gold_intent") or "").strip()
        g_action = (r.get("gold_auto_or_escalate") or "").strip()
        g_reason = (r.get("escalation_reason") or "").strip()

        # Check intent
        if not g_intent:
            errors.append(f"Row {idx} (Tweet {r.get('tweet_id')}): blank 'gold_intent'")
        elif g_intent not in CANONICAL_INTENTS:
            errors.append(f"Row {idx}: invalid gold_intent '{g_intent}' (must be one of {CANONICAL_INTENTS})")
        else:
            intent_counts[g_intent] += 1

        # Check action
        if not g_action:
            errors.append(f"Row {idx}: blank 'gold_auto_or_escalate'")
        elif g_action not in ["auto_handle", "escalate"]:
            errors.append(f"Row {idx}: invalid action '{g_action}' (must be 'auto_handle' or 'escalate')")
        else:
            action_counts[g_action] += 1

        # Check escalation reason
        if g_action == "escalate" and not g_reason:
            errors.append(f"Row {idx}: action is 'escalate' but 'escalation_reason' is blank")

    print("\n" + "=" * 80)
    print(" GOLDEN SET VALIDATION REPORT")
    print("=" * 80)
    print(f"Total Rows Checked: {len(rows)}")

    if errors:
        print(f"\nSTATUS: FAILED ({len(errors)} validation errors detected)")
        print("-" * 80)
        for err in errors[:25]:
            print(f"  [!] {err}")
        if len(errors) > 25:
            print(f"  ... and {len(errors) - 25} more errors.")
        print("-" * 80)
        print("Please complete the remaining rows via CLI (`python src/labeling_tool.py --cli`) or Web UI (`python src/labeling_tool.py --web`).")
        return False
    else:
        print("\nSTATUS: PASSED (100% Validated)")
        print("-" * 80)
        print("Action Distribution:")
        for act, count in action_counts.items():
            print(f"  - {act:15s}: {count:3d} ({count/len(rows)*100:.1f}%)")
        print("\nIntent Distribution:")
        for it, count in intent_counts.items():
            print(f"  - {it:35s}: {count:3d} ({count/len(rows)*100:.1f}%)")
        print("=" * 80)
        return True


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 10: Golden Set Sampler, Labeling Interface & Validator")
    parser.add_argument("--generate", action="store_true", help="Generate fresh blind golden_set.csv")
    parser.add_argument("--cli", action="store_true", help="Launch interactive CLI labeling interface")
    parser.add_argument("--web", action="store_true", help="Launch local web-based labeling interface")
    parser.add_argument("--port", type=int, default=8080, help="Port for web server (default: 8080)")
    parser.add_argument("--stats", action="store_true", help="Print current labeling progress statistics")
    parser.add_argument("--validate", action="store_true", help="Run strict schema validation on golden_set.csv")

    args = parser.parse_args()

    if args.generate:
        generate_blind_golden_set()
    elif args.cli:
        run_cli_labeling()
    elif args.web:
        run_web_server(port=args.port)
    elif args.validate:
        validate_golden_set()
    elif args.stats:
        stats = get_progress_stats()
        print(f"Current Golden Set Progress: {stats['labeled']}/{stats['total']} labeled ({stats['percentage']}%)")
        print("Intent counts:", stats["intents"])
        print("Action counts:", stats["actions"])
    else:
        # Default behavior: show stats & usage
        stats = get_progress_stats()
        print(f"Golden Set Status: {stats['labeled']}/{stats['total']} labeled ({stats['percentage']}%)")
        print("\nUsage:")
        print("  py311/python src/labeling_tool.py --cli       (Interactive Terminal Labeling)")
        print("  py311/python src/labeling_tool.py --web       (Interactive Web Browser UI on http://127.0.0.1:8080)")
        print("  py311/python src/labeling_tool.py --validate  (Validate dataset completeness & schema)")
        print("  py311/python src/labeling_tool.py --stats     (View current progress breakdown)")
