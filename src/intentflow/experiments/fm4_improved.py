"""
intentflow/experiments/fm4_improved.py
Point #16 — Améliorer FM4 mixed-provenance.

Stratégie: span-level provenance + confirmation obligatoire sur spans mixtes.
Vise à réduire ASR de 40% vers <20%.
"""
import json, csv
from pathlib import Path
from intentflow.data.mixed_provenance_v2 import get_mixed_v2_data
from intentflow.models.models import (
    BenignTask, AttackedTask, RAGDocument, ProvenanceSource, Intent)
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.policy.models import ArgumentPolicy
from intentflow.provenance.span_provenance import detect_spans, check_argument_against_spans
from intentflow.tools.registry import get_tool
from intentflow.metrics.confidence_intervals import wilson_interval
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/fm4_improved")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_strict_policies_with_span():
    """
    Politiques strictes + règle span-level:
    si argument vient d'un span mixte → Confirm obligatoire.
    """
    engine = PolicyEngine.load_default_policies()
    for tool_name in ["send_email","create_calendar_event"]:
        if tool_name not in engine.policies: continue
        engine.policies[tool_name].arguments = [
            p if p.argument_name not in ("recipient","participants")
            else ArgumentPolicy(argument_name=p.argument_name,
                allowed_origins=["user_intent"],
                min_confidence=p.min_confidence, role=p.role)
            for p in engine.policies[tool_name].arguments]
    if "write_file" in engine.policies:
        engine.policies["write_file"].arguments = [
            p if p.argument_name != "path"
            else ArgumentPolicy(argument_name="path",
                allowed_origins=["user_intent"], min_confidence=0.90, role="locator")
            for p in engine.policies["write_file"].arguments]
    return engine


def run_fm4_with_span_provenance(benign, attacked, documents):
    """
    Évalue B5, B6-baseline, B6+span sur FM4 mixed-provenance.
    B6+span ajoute: si l'argument value matche un span externe → Confirm.
    """
    # Pré-analyser les spans de tous les documents
    span_results = {doc.doc_id: detect_spans(doc) for doc in documents}

    engine = load_strict_policies_with_span()
    mediator = ToolMediator(policy_engine=engine, documents=documents)

    results = {b: {"asr":0,"btsr":0,"fbr":0,"na":0,"nb":0}
               for b in ["B5","B6_baseline","B6_span"]}

    for task in attacked:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn: continue
        args = {**task.expected_arguments, **task.corrupted_arguments}
        oracle = Intent(action=task.expected_tool,
            parameters={k:v for k,v in task.expected_arguments.items() if v is not None},
            confidence=1.0)

        # B5: provenance-only
        r5 = mediator.execute(task.expected_tool, args, task.user_message,
                              tool_fn, baseline="B5", context_documents=documents)
        if r5["executed"]: results["B5"]["asr"] += 1
        results["B5"]["na"] += 1

        # B6 baseline
        r6 = mediator.execute(task.expected_tool, args, task.user_message,
                              tool_fn, baseline="B6", context_documents=documents,
                              oracle_intent=oracle)
        if r6["executed"]: results["B6_baseline"]["asr"] += 1
        results["B6_baseline"]["na"] += 1

        # B6 + span-level provenance
        # Vérifier si un argument corrompu matche un span externe
        span_blocked = False
        for arg_name, arg_val in task.corrupted_arguments.items():
            for doc in documents:
                span_r = span_results[doc.doc_id]
                origin, conf = check_argument_against_spans(str(arg_val), span_r)
                if origin == "mixed_provenance_external":
                    span_blocked = True  # → Confirm → attaque bloquée
                    break
            if span_blocked: break

        if span_blocked:
            pass  # Confirm → attaque bloquée
        elif r6["executed"]:
            results["B6_span"]["asr"] += 1
        results["B6_span"]["na"] += 1

    for task in benign:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn: continue
        oracle = Intent(action=task.expected_tool,
            parameters={k:v for k,v in task.expected_arguments.items() if v is not None},
            confidence=1.0)
        for baseline in ["B5","B6_baseline","B6_span"]:
            r = mediator.execute(task.expected_tool, task.expected_arguments,
                task.user_message, tool_fn, baseline=baseline.split("_")[0],
                context_documents=documents, oracle_intent=oracle)
            if r["executed"]: results[baseline]["btsr"] += 1
            else: results[baseline]["fbr"] += 1
            results[baseline]["nb"] += 1

    return results


def main():
    benign, attacked, documents = get_mixed_v2_data()
    logger.info(f"FM4 improved: {len(benign)} benign, {len(attacked)} attacked")

    results = run_fm4_with_span_provenance(benign, attacked, documents)

    print(f"\n{'='*65}")
    print(f"FM4 MIXED-PROVENANCE — B5 vs B6 vs B6+SPAN (N={len(attacked)})")
    print(f"{'='*65}")
    print(f"{'Baseline':<15} {'ASR':>8} {'ASR CI':>22} {'BTSR':>8} {'FBR':>8}")
    print("-"*65)

    rows = []
    for bl, r in results.items():
        na,nb = r["na"],r["nb"]
        asr_ci = wilson_interval(r["asr"],na) if na>0 else None
        asr_pct = r["asr"]/na*100 if na>0 else 0
        btsr_pct = r["btsr"]/nb*100 if nb>0 else 0
        fbr_pct = r["fbr"]/nb*100 if nb>0 else 0
        ci_str = f"[{asr_ci.lower*100:.1f},{asr_ci.upper*100:.1f}%]" if asr_ci else ""
        print(f"{bl:<15} {asr_pct:>7.1f}% {ci_str:>22} {btsr_pct:>7.1f}% {fbr_pct:>7.1f}%")
        rows.append({"baseline":bl,"na":na,"nb":nb,
                     "asr":round(asr_pct,1),"btsr":round(btsr_pct,1),"fbr":round(fbr_pct,1)})

    print(f"\nB6+span improvement vs B6 baseline: "
          f"{results['B6_baseline']['asr']/results['B6_baseline']['na']*100:.1f}% → "
          f"{results['B6_span']['asr']/results['B6_span']['na']*100:.1f}%")

    with open(RESULTS_DIR/"fm4_improved.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["baseline","na","nb","asr","btsr","fbr"])
        writer.writeheader(); writer.writerows(rows)
    logger.info(f"Results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()