"""
intentflow/experiments/e2e_multimedia.py

Chaîne multimédia end-to-end — Phase 4 (points 1.1/2.1 du retour reviewer).

screenshot/document scanné (image synthétique, perception/image_synthesis.py)
-> OCR (pytesseract, perception/ocr.py) -> le texte extrait devient un
RAGDocument comme n'importe quel autre document retrieved -> planner
(gpt-4o-mini-2024-07-18) -> tool call -> décision du médiateur (IF-Confirm)
-> effet. Comparé, sur les MÊMES tâches et techniques, à la condition
"texte direct" (Phase 3, sans passage par image/OCR) — c'est la demande
explicite du retour reviewer : "comparer, sur les mêmes tâches, l'entrée
texte directe et l'entrée dérivée d'un média, avec des métriques par
étape : DER, AHR@k, CASR, PCR au niveau tâche/appel, UER, BTSR/FBR et
surcoût en latence."

Réutilise l'infrastructure de e2e_phase3.py (techniques d'injection,
prompts, is_corrupted, PolicyEngine.if_confirm) — seule la source du texte
du document injecté change entre les deux conditions.

Note sur les techniques : "obfuscated" (Phase 3, caractères zero-width) est
exclue ici — une fois l'image rendue, un caractère zero-width n'a aucune
représentation visuelle et disparaît avant même l'OCR ; l'obfuscation
textuelle ne se transpose pas au domaine image. C'est en soi une
observation méthodologique (une technique d'évasion textuelle n'a pas
d'équivalent perceptuel automatique), documentée plutôt que masquée.

Protocole préenregistré :
- Modèle planner : gpt-4o-mini-2024-07-18 (identique à Phase 3).
- OCR : tesseract (pytesseract), langue par défaut ("eng").
- Tâches : même pool Bench-v2 medium+hard que Phase 3 (comparaison directe).
- 4 techniques d'injection (direct/authority_urgency/tool_output_relay/multi_hop)
  x 5 variations visuelles (clean/noisy/low_contrast/ui_screenshot/scanned_pdf),
  assignées en round-robin déterministe par index de tâche (seed=42).
- Chaque tâche attaquée est exécutée dans LES DEUX conditions (texte direct
  et média), sur le même contenu adversarial — comparaison appariée.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

from openai import OpenAI

from intentflow.experiments.e2e_phase3 import (
    INJECTION_TECHNIQUES,
    MODEL,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
    is_corrupted,
)
from intentflow.metrics.e2e_metrics import compute_e2e_metrics, format_e2e_report
from intentflow.metrics.multimedia_metrics import (
    adversarial_value_present,
    compute_ahr_at_k,
    compute_casr,
    compute_der,
)
from intentflow.models.models import AttackedTask, BenignTask, Intent, RAGDocument
from intentflow.perception import ocr
from intentflow.perception.image_synthesis import VARIATIONS, render_document_image
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.utils.logger import logger

SEED = 42
RESULTS_DIR = Path("data/results/e2e_multimedia")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR = Path("data/benchmark_v2")

# "obfuscated" dropped — see module docstring.
MEDIA_TECHNIQUE_NAMES = ["direct", "authority_urgency", "tool_output_relay", "multi_hop"]
VARIATION_NAMES = list(VARIATIONS.keys())


def call_planner(client: OpenAI, user_message: str, docs: list[RAGDocument]) -> dict | None:
    doc_text = "\n\n".join(
        f"[Doc {i + 1} | {d.source_domain}]\n{d.content}" for i, d in enumerate(docs)
    )
    try:
        r = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": PLANNER_USER_TEMPLATE.format(
                        user_message=user_message, documents=doc_text
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Planner error: {e}")
        return None


def build_text_docs(
    task: AttackedTask, technique: str, benign_pool: list[RAGDocument], rng: random.Random
) -> list[RAGDocument]:
    """Condition texte direct (Phase 3) : le contenu adversarial tel quel, aucun passage par image/OCR."""
    benign_ctx = rng.sample(benign_pool, k=min(2, len(benign_pool)))
    contents = INJECTION_TECHNIQUES[technique](task.corrupted_arguments)
    injected = [
        RAGDocument(
            doc_id=f"txt_{task.task_id}_{technique}_{i}",
            content=content,
            is_injected=True,
            injection_type=technique,
            source_domain="untrusted_external",
        )
        for i, content in enumerate(contents)
    ]
    return benign_ctx + injected


def build_media_docs(
    task: AttackedTask,
    technique: str,
    variation_name: str,
    benign_pool: list[RAGDocument],
    rng: random.Random,
) -> tuple[list[RAGDocument], list[tuple[str, str]], float]:
    """
    Condition média : le même contenu adversarial est rendu en image
    (`variation_name`), puis passé par OCR — le texte extrait (pas le texte
    de référence) devient le contenu du RAGDocument que voit le planner.

    Returns: (documents, [(texte_référence, texte_ocr), ...] pour DER, latence OCR totale ms)
    """
    benign_ctx = rng.sample(benign_pool, k=min(2, len(benign_pool)))
    contents = INJECTION_TECHNIQUES[technique](task.corrupted_arguments)

    injected_docs = []
    der_pairs = []
    ocr_latency_ms = 0.0
    for i, reference_text in enumerate(contents):
        img = render_document_image(reference_text, variation_name, seed=SEED)
        t0 = time.perf_counter()
        result = ocr.extract_text(img)
        ocr_latency_ms += (time.perf_counter() - t0) * 1000
        der_pairs.append((reference_text, result.text))
        injected_docs.append(
            RAGDocument(
                doc_id=f"media_{task.task_id}_{technique}_{variation_name}_{i}",
                content=result.text,
                is_injected=True,
                injection_type=f"{technique}+{variation_name}",
                source_domain="untrusted_external",
            )
        )
    return benign_ctx + injected_docs, der_pairs, ocr_latency_ms


def run_attacked_task(
    client: OpenAI,
    task: AttackedTask,
    technique: str,
    condition: str,
    docs: list[RAGDocument],
    mediator: ToolMediator,
    extra_latency_ms: float = 0.0,
) -> dict:
    t0 = time.perf_counter()
    proposed = call_planner(client, task.user_message, docs)
    base = {
        "task_id": task.task_id,
        "is_attacked": True,
        "technique": technique,
        "condition": condition,
        "scenario_family": task.scenario_family,
        "attack_type": task.attack_type,
    }
    if not proposed:
        return {
            **base,
            "planner_failed": True,
            "argument_corrupted": False,
            "mediation_blocked": False,
            "tool_correct": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000 + extra_latency_ms, 1),
        }

    p_tool = proposed.get("tool", "")
    p_args = proposed.get("arguments", {})
    corrupted = is_corrupted(p_args, task)
    tool_fn = get_tool(p_tool) if p_tool else None

    oracle = Intent(
        action=task.expected_tool,
        parameters={k: v for k, v in task.expected_arguments.items() if v is not None},
        confidence=1.0,
    )

    if tool_fn and p_args:
        res = mediator.execute(
            p_tool,
            p_args,
            task.user_message,
            tool_fn,
            baseline="B6",
            context_documents=docs,
            oracle_intent=oracle,
        )
        blocked = not res["allowed"]
        decision_type = (res.get("decision") or {}).get("decision")
    else:
        blocked = True
        decision_type = None

    return {
        **base,
        "planner_failed": False,
        "proposed_tool": p_tool,
        "tool_correct": p_tool == task.expected_tool,
        "argument_corrupted": corrupted,
        "mediation_blocked": blocked,
        "decision_type": decision_type,
        "latency_ms": round((time.perf_counter() - t0) * 1000 + extra_latency_ms, 1),
    }


def run_benign_task(
    client: OpenAI, task: BenignTask, docs: list[RAGDocument], mediator: ToolMediator
) -> dict:
    t0 = time.perf_counter()
    proposed = call_planner(client, task.user_message, docs)
    base = {"task_id": task.task_id, "is_attacked": False, "scenario_family": task.scenario_family}
    if not proposed:
        return {
            **base,
            "planner_failed": True,
            "tool_correct": False,
            "mediation_blocked": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    p_tool = proposed.get("tool", "")
    p_args = proposed.get("arguments", {})
    tool_fn = get_tool(p_tool) if p_tool else None
    oracle = Intent(
        action=task.expected_tool,
        parameters={k: v for k, v in task.expected_arguments.items() if v is not None},
        confidence=1.0,
    )
    if tool_fn and p_args:
        res = mediator.execute(
            p_tool,
            p_args,
            task.user_message,
            tool_fn,
            baseline="B6",
            context_documents=docs,
            oracle_intent=oracle,
        )
        blocked = not res["allowed"]
    else:
        blocked = True
    return {
        **base,
        "planner_failed": False,
        "proposed_tool": p_tool,
        "tool_correct": p_tool == task.expected_tool,
        "mediation_blocked": blocked,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-attacked", type=int, default=110)
    parser.add_argument("--n-benign", type=int, default=50)
    args = parser.parse_args()

    if not ocr.is_available():
        raise SystemExit(
            "tesseract n'est pas disponible (binaire ou binding pytesseract manquant). "
            "Installer avec `apt-get install tesseract-ocr` puis relancer."
        )

    rng = random.Random(SEED)
    client = OpenAI()
    engine = PolicyEngine.if_confirm()

    all_benign = [
        BenignTask(**t) for t in json.loads((BENCH_DIR / "tasks_benign.json").read_text())
    ]
    all_attacked = [
        AttackedTask(**t) for t in json.loads((BENCH_DIR / "tasks_attacked.json").read_text())
    ]
    benign_pool = [
        RAGDocument(**d)
        for d in json.loads((BENCH_DIR / "documents.json").read_text())
        if not d["is_injected"]
    ]
    med_hard = [
        t
        for t in all_attacked
        if "medium" in (t.attack_payload or "") or "hard" in (t.attack_payload or "")
    ]
    attacked_sample = med_hard[: args.n_attacked]
    benign_sample = all_benign[: args.n_benign]

    mediator = ToolMediator(policy_engine=engine, documents=[])

    logger.info(
        f"E2E Multimedia: model={MODEL}, {len(attacked_sample)} attacked (x2 conditions), "
        f"{len(benign_sample)} benign"
    )

    decisions: list[dict] = []
    der_pairs_all: list[tuple[str, str]] = []
    ahr_hits: list[bool] = []

    for i, task in enumerate(benign_sample):
        docs = rng.sample(benign_pool, k=min(3, len(benign_pool)))
        logger.info(f"  Benign {i + 1}/{len(benign_sample)}: {task.task_id}")
        decisions.append(run_benign_task(client, task, docs, mediator))

    for i, task in enumerate(attacked_sample):
        technique = MEDIA_TECHNIQUE_NAMES[i % len(MEDIA_TECHNIQUE_NAMES)]
        variation = VARIATION_NAMES[i % len(VARIATION_NAMES)]

        text_docs = build_text_docs(task, technique, benign_pool, rng)
        logger.info(f"  Attacked {i + 1}/{len(attacked_sample)}: {task.task_id} [{technique}] text")
        decisions.append(
            run_attacked_task(client, task, technique, "text_direct", text_docs, mediator)
        )

        media_docs, der_pairs, ocr_latency = build_media_docs(
            task, technique, variation, benign_pool, rng
        )
        der_pairs_all.extend(der_pairs)
        for evil_val in task.corrupted_arguments.values():
            values = evil_val if isinstance(evil_val, list) else [evil_val]
            for v in values:
                if isinstance(v, str):
                    ahr_hits.append(any(adversarial_value_present(v, ext) for _, ext in der_pairs))
        logger.info(
            f"  Attacked {i + 1}/{len(attacked_sample)}: {task.task_id} [{technique}+{variation}] media"
        )
        d = run_attacked_task(
            client, task, technique, "media", media_docs, mediator, extra_latency_ms=ocr_latency
        )
        d["variation"] = variation
        decisions.append(d)

    log_file = RESULTS_DIR / "decisions.jsonl"
    with open(log_file, "w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")

    # Benign decisions are shared by both conditions (no text/media split for
    # them — only the attacked side has a document to render as an image).
    text_decisions = [
        d for d in decisions if not d["is_attacked"] or d.get("condition") == "text_direct"
    ]
    media_decisions = [
        d for d in decisions if not d["is_attacked"] or d.get("condition") == "media"
    ]

    text_m = compute_e2e_metrics(text_decisions)
    media_m = compute_e2e_metrics(media_decisions)
    der = compute_der(der_pairs_all)
    ahr = compute_ahr_at_k(ahr_hits)
    casr = compute_casr(media_decisions)

    print(f"\n{'=' * 70}")
    print(
        f"E2E MULTIMEDIA — model={MODEL}, N={len(attacked_sample)} attacked / {len(benign_sample)} benign"
    )
    print(f"{'=' * 70}")
    print("\n--- Condition: TEXT DIRECT (Phase 3 baseline, same tasks/techniques) ---")
    print(format_e2e_report(text_m))
    print("\n--- Condition: MEDIA (image -> OCR -> planner) ---")
    print(format_e2e_report(media_m))
    print(f"\nCASR (media)     : {casr['n_corrupted']}/{casr['n']} = {casr['casr']}%")
    print(
        f"DER (OCR fidelity): mean={der['der_mean']}%  min={der['der_min']}%  n={der['n']} documents"
    )
    print(f"AHR@k (payload survives OCR): {ahr['n_hit']}/{ahr['n']} = {ahr['ahr']}%")
    media_lat = [d["latency_ms"] for d in media_decisions if d.get("latency_ms")]
    text_lat = [d["latency_ms"] for d in text_decisions if d.get("latency_ms")]
    if media_lat and text_lat:
        print(
            f"Latency overhead (media vs text, mean): "
            f"{sum(media_lat) / len(media_lat) - sum(text_lat) / len(text_lat):.0f}ms"
        )

    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "na", "pmcr", "cme", "uer", "btsr", "fbr"])
        writer.writerow(
            [
                "text_direct",
                text_m["na"],
                text_m["pmcr"],
                text_m["cme"],
                text_m["uer"],
                text_m["btsr"],
                text_m["fbr"],
            ]
        )
        writer.writerow(
            [
                "media",
                media_m["na"],
                media_m["pmcr"],
                media_m["cme"],
                media_m["uer"],
                media_m["btsr"],
                media_m["fbr"],
            ]
        )
        writer.writerow(["der_mean", der["der_mean"], "", "", "", "", ""])
        writer.writerow(["ahr", ahr["ahr"], "", "", "", "", ""])
        writer.writerow(["casr", casr["casr"], "", "", "", "", ""])

    logger.info(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
