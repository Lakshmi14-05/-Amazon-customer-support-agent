"""Failure analysis script for Phase 14"""
import csv, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'src')

from agent_pipeline import AmazonSupportAgent
from baselines import Baseline2ClassicalIRBot

with open('data/evaluation/golden_set.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

agent = AmazonSupportAgent()
b2 = Baseline2ClassicalIRBot()

# 1. Verify classifier identity
print('=== CLASSIFIER IDENTITY CHECK ===')
mismatches = 0
for r in rows[:10]:
    a_out = agent.process_message(r['customer_message'])
    b2_out = b2.process_message(r['customer_message'])
    if a_out['intent'] != b2_out['intent']:
        mismatches += 1
        print('MISMATCH on:', r['customer_message'][:60])
print('Mismatches in first 10 rows:', mismatches, '(0 = identical classifier)')
print()

# 2. Find false auto-handles (gold=escalate, pipeline=auto_handle)
print('=== FALSE AUTO-HANDLES (gold=escalate, pipeline=auto_handle) ===')
fah_rows = []
for r in rows:
    out = agent.process_message(r['customer_message'])
    if r['gold_auto_or_escalate'] == 'escalate' and out['decision'] == 'auto_handle':
        fah_rows.append({
            'thread_id': r['thread_id'],
            'gold_intent': r['gold_intent'],
            'pred_intent': out['intent'],
            'confidence': out['intent_confidence'],
            'draft_reply': out['draft_reply'],
            'escalation_reason': r['escalation_reason'],
            'msg': r['customer_message']
        })

print('Total false auto-handles:', len(fah_rows))
for i, f in enumerate(fah_rows, 1):
    print()
    print(str(i) + '. Thread ' + f['thread_id'] + '  gold=' + f['gold_intent'] + '  pred=' + f['pred_intent'] + '  conf=' + str(f['confidence']))
    print('   Your escalation reason: ' + f['escalation_reason'])
    print('   Message: ' + f['msg'][:140])
    print('   Pipeline draft: ' + f['draft_reply'][:140])

print()

# 3. Worst-intent failures
print('=== INTENT MISCLASSIFICATIONS BY BAD INTENT ===')
target_intents = ['other_miscellaneous_feedback', 'damaged_defective_wrong_item', 'missing_stolen_delivery']
for intent in target_intents:
    intent_rows = [r for r in rows if r['gold_intent'] == intent]
    misclassified = []
    for r in intent_rows:
        out = agent.process_message(r['customer_message'])
        if out['intent'] != intent:
            misclassified.append({'thread_id': r['thread_id'], 'predicted': out['intent'], 'msg': r['customer_message'], 'action': r['gold_auto_or_escalate']})
    print()
    print('Intent: ' + intent + ' (' + str(len(misclassified)) + '/' + str(len(intent_rows)) + ' misclassified)')
    for m in misclassified[:4]:
        print('  Thread ' + m['thread_id'] + ' -> predicted: ' + m['predicted'])
        print('  Msg: ' + m['msg'][:120])

print()

# 4. Low escalation score cases (gold=auto_handle but pipeline=escalate, wasting human time)
print('=== FALSE ESCALATIONS (gold=auto_handle, pipeline=escalate) ===')
fe_rows = []
for r in rows:
    out = agent.process_message(r['customer_message'])
    if r['gold_auto_or_escalate'] == 'auto_handle' and out['decision'] == 'escalate':
        fe_rows.append({
            'thread_id': r['thread_id'],
            'gold_intent': r['gold_intent'],
            'pred_intent': out['intent'],
            'decision_reason': out['decision_reason'],
            'msg': r['customer_message']
        })

print('Total false escalations:', len(fe_rows))
for f in fe_rows[:5]:
    print()
    print('Thread ' + f['thread_id'] + '  gold_intent=' + f['gold_intent'] + '  pred=' + f['pred_intent'])
    print('Pipeline decision reason: ' + f['decision_reason'])
    print('Message: ' + f['msg'][:140])

# Save all to JSON for report
output = {
    'false_auto_handles': fah_rows,
    'false_escalations': fe_rows[:20],
}
with open('reports/failure_analysis_examples.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)
print()
print('Saved to reports/failure_analysis_examples.json')
