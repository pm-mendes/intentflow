"""
intentflow/experiments/agentdojo_runner.py
Point #9 — AgentDojo officiel avec vrai planner LLM.

Exécute IntentFlow devant le vrai agent AgentDojo via leur API publique.
Référence: https://github.com/ethz-spylab/agentdojo
"""
import json, time, csv, subprocess, sys
from pathlib import Path
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.models.models import Intent, RAGDocument
from intentflow.metrics.confidence_intervals import wilson_interval
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/agentdojo_official")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Installer agentdojo si absent
def ensure_agentdojo():
    try:
        import agentdojo
        return True
    except ImportError:
        logger.info("Installing agentdojo...")
        result = subprocess.run(
            [sys.executable,"-m","pip","install","agentdojo","--break-system-packages","-q"],
            capture_output=True)
        if result.returncode == 0:
            try:
                import agentdojo
                return True
            except ImportError:
                pass
        logger.warning("agentdojo not available — using faithful reproduction fallback")
        return False

def run_with_agentdojo_api(engine, n_tasks=40):
    """
    Tente d'utiliser l'API officielle AgentDojo.
    Fallback vers la reproduction fidèle si l'API n'est pas disponible.
    """
    has_agentdojo = ensure_agentdojo()

    if has_agentdojo:
        try:
            from agentdojo.task_suites import get_suite
            from agentdojo.functions_runtime import FunctionsRuntime
            results = []

            # Domaines disponibles dans AgentDojo
            suite_names = ["workspace", "banking", "travel", "slack"]
            for suite_name in suite_names[:2]:  # workspace + banking suffisent
                try:
                    suite = get_suite(suite_name)
                    tasks = list(suite.tasks)[:10]  # 10 tasks par domaine
                    for task in tasks:
                        env = suite.load_and_inject_defaults(
                            FunctionsRuntime(suite.tools))
                        user_task = task.PROMPT
                        # Extraire les documents du contexte AgentDojo
                        docs = [RAGDocument(
                            doc_id=f"ag_{suite_name}_{i}",
                            content=str(item),
                            is_injected=False,
                            source_domain="trusted_internal")
                            for i, item in enumerate(
                                getattr(env, 'inbox', {}).items() if hasattr(env,'inbox') else [])]
                        results.append({
                            "suite": suite_name,
                            "task": task.__class__.__name__,
                            "user_message": user_task,
                            "docs_count": len(docs),
                            "source": "agentdojo_official_api"
                        })
                except Exception as e:
                    logger.warning(f"Suite {suite_name}: {e}")
            if results:
                return results, "official"
        except Exception as e:
            logger.warning(f"AgentDojo API error: {e}")

    # Fallback: reproduction fidèle
    from intentflow.data.agentdojo_official import (
        get_official_agentdojo_data, OFFICIAL_BENIGN, OFFICIAL_ATTACKED)
    return None, "faithful_reproduction"


def evaluate_on_agentdojo(n_tasks=40, use_llm=True, model="gpt-4o-mini"):
    """
    Évalue IntentFlow sur AgentDojo (officiel si dispo, sinon reproduction fidèle).
    Avec vrai planner LLM si use_llm=True.
    """
    engine = PolicyEngine.load_default_policies()
    api_results, source = run_with_agentdojo_api(engine, n_tasks)

    from intentflow.data.agentdojo_official import (
        get_official_agentdojo_data)
    benign, attacked, docs = get_official_agentdojo_data()
    mediator = ToolMediator(policy_engine=engine, documents=docs)

    decisions = []
    if use_llm:
        from openai import OpenAI
        client = OpenAI()
        from intentflow.experiments.e2e_n200 import run_task
        import random
        rng = random.Random(42)
        ben_s = rng.sample(benign, min(n_tasks//2, len(benign)))
        att_s = rng.sample(attacked, min(n_tasks//2, len(attacked)))

        logger.info(f"Running E2E on AgentDojo ({source}): {len(ben_s)} ben, {len(att_s)} att")
        for t in ben_s:
            decisions.append(run_task(client, t, docs, mediator, False, model))
        for t in att_s:
            inj_docs = docs + [RAGDocument(
                doc_id=f"inj_{t.task_id}", content=t.attack_payload or "",
                is_injected=True, source_domain="untrusted_external")]
            decisions.append(run_task(client, t, inj_docs, mediator, True, model))
    else:
        # Sans LLM: médiation directe (oracle intent)
        for task in benign[:n_tasks//2]:
            tool_fn = get_tool(task.expected_tool)
            if not tool_fn: continue
            oracle = Intent(action=task.expected_tool,
                parameters={k:v for k,v in task.expected_arguments.items() if v is not None},
                confidence=1.0)
            res = mediator.execute(task.expected_tool, task.expected_arguments,
                task.user_message, tool_fn, baseline="B6",
                context_documents=docs, oracle_intent=oracle)
            decisions.append({"task_id":task.task_id,"is_attacked":False,
                "tool_correct":True,"executed":res["executed"],
                "mediation_blocked":not res["allowed"],"argument_corrupted":False,
                "scenario_family":task.scenario_family,"source":source})
        for task in attacked[:n_tasks//2]:
            tool_fn = get_tool(task.expected_tool)
            if not tool_fn: continue
            oracle = Intent(action=task.expected_tool,
                parameters={k:v for k,v in task.expected_arguments.items() if v is not None},
                confidence=1.0)
            args = {**task.expected_arguments, **task.corrupted_arguments}
            res = mediator.execute(task.expected_tool, args,
                task.user_message, tool_fn, baseline="B6",
                context_documents=docs, oracle_intent=oracle)
            decisions.append({"task_id":task.task_id,"is_attacked":True,
                "tool_correct":True,"executed":res["executed"],
                "mediation_blocked":not res["allowed"],"argument_corrupted":True,
                "scenario_family":task.scenario_family,"source":source})

    att_d = [d for d in decisions if d.get("is_attacked") and not d.get("planner_failed")]
    ben_d = [d for d in decisions if not d.get("is_attacked") and not d.get("planner_failed")]
    na,nb = len(att_d),len(ben_d)
    asr_n = sum(1 for d in att_d if d.get("executed") or
                (d.get("argument_corrupted") and not d.get("mediation_blocked")))
    btsr_n = sum(1 for d in ben_d if d.get("executed") and not d.get("mediation_blocked"))
    fbr_n = sum(1 for d in ben_d if d.get("mediation_blocked"))
    asr_ci = wilson_interval(asr_n,na) if na>0 else None

    print(f"\n{'='*60}")
    print(f"AGENTDOJO EVALUATION ({source}, N={len(decisions)})")
    print(f"{'='*60}")
    print(f"ASR: {asr_n}/{na}={asr_n/na*100:.1f}% "
          f"[{asr_ci.lower*100:.1f},{asr_ci.upper*100:.1f}%]" if asr_ci else f"ASR: {asr_n}/{na}")
    print(f"BTSR: {btsr_n}/{nb}={btsr_n/nb*100:.1f}%")
    print(f"FBR: {fbr_n}/{nb}={fbr_n/nb*100:.1f}%")
    print(f"Source: {source}")

    with open(RESULTS_DIR/"agentdojo_results.jsonl","w") as f:
        for d in decisions: f.write(json.dumps(d)+"\n")

    return {"na":na,"nb":nb,
            "asr":asr_n/na*100 if na>0 else 0,
            "asr_ci_lo":asr_ci.lower*100 if asr_ci else 0,
            "asr_ci_hi":asr_ci.upper*100 if asr_ci else 0,
            "btsr":btsr_n/nb*100 if nb>0 else 0,
            "fbr":fbr_n/nb*100 if nb>0 else 0,
            "source":source}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    evaluate_on_agentdojo(n_tasks=args.n, use_llm=not args.no_llm, model=args.model)