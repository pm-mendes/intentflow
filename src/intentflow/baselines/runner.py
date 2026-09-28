"""
Runner pour les expériences de baselines.

Pourquoi un runner séparé du mediator ?
Le mediator est la logique de décision ; le runner est la logique d'expérience.
Cette séparation permet de changer la façon dont on itère sur les tâches
sans toucher à la logique de sécurité.

Support oracle (ajouté après coup) : `run()` n'a jamais construit ni transmis
d'`oracle_intent` à `mediator.execute()`. Pour B4/B6, cela veut dire un mode
"llm" forcé — un vrai appel API par tâche — quel que soit l'appelant. C'est ce
qui a produit le chiffre principal du papier (Bench-v2 B6, ASR=24%) alors que
le texte affirme explicitement que "all controlled benchmarks use oracle
intent contracts to isolate mediation quality from intent-extraction error".
`use_oracle=True` construit maintenant un `Intent` gold depuis
`task.expected_arguments` (jamais `corrupted_arguments`) et le transmet —
gratuit (aucun appel LLM) et conforme à ce que le papier prétend mesurer.
Reste opt-in (défaut `False`) pour ne changer le comportement d'aucun script
existant sans décision explicite.
"""

from pathlib import Path

from tqdm import tqdm

from intentflow.intent.extractor import INTENT_SCHEMA_FIELDS
from intentflow.models.models import (
    AttackedTask,
    BenignTask,
    Intent,
    MediationDecision,
    RAGDocument,
)
from intentflow.pipeline.mediator import ToolMediator
from intentflow.tools.registry import get_tool
from intentflow.utils.logger import logger


def oracle_intent_for(task: BenignTask) -> Intent:
    """
    Gold intent contract for a task: what the user actually authorized,
    derived from `task.expected_arguments` — restricted to
    `INTENT_SCHEMA_FIELDS` (never a payload field like `content`/`body`, and
    never `corrupted_arguments`, even for an `AttackedTask`).
    """
    return Intent(
        action=task.expected_tool,
        parameters={
            k: v
            for k, v in task.expected_arguments.items()
            if v is not None and k in INTENT_SCHEMA_FIELDS
        },
        confidence=1.0,
    )


class BaselineRunner:
    """
    Exécute une baseline sur l'ensemble des tâches bénignes et attaquées.

    Un objet BaselineRunner est créé pour chaque baseline (B0–B6).
    Il itère sur toutes les tâches et enregistre les décisions dans un fichier JSONL.
    """

    def __init__(self, mediator: ToolMediator, output_dir: str = "data/results") -> None:
        self.mediator = mediator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        baseline: str,
        benign_tasks: list[BenignTask],
        attacked_tasks: list[AttackedTask],
        documents: list[RAGDocument],
        use_oracle: bool = False,
    ) -> Path:
        """
        Exécute une baseline et enregistre les décisions.

        Args:
            use_oracle: si True, B4/B6 reçoivent l'intent gold construit
                depuis `task.expected_arguments` (voir `oracle_intent_for`)
                au lieu de faire un appel LLM réel par tâche. Sans effet sur
                B0/B1/B2/B3/B5, qui n'utilisent jamais `oracle_intent`.
                Défaut `False` pour ne changer le comportement d'aucun
                appelant existant sans qu'il le demande explicitement.

        Returns:
            Chemin du fichier JSONL de décisions.
        """
        log_file = self.output_dir / f"decisions_{baseline}.jsonl"
        logger.info(f"Running baseline {baseline} → {log_file}")

        with open(log_file, "w") as f:
            # === Tâches bénignes ===
            for task in tqdm(benign_tasks, desc=f"[{baseline}] Benign"):
                tool_fn = get_tool(task.expected_tool)
                if not tool_fn:
                    logger.warning(f"Tool '{task.expected_tool}' not found, skipping.")
                    continue

                result = self.mediator.execute(
                    tool_name=task.expected_tool,
                    arguments=task.expected_arguments,
                    user_message=task.user_message,
                    tool_fn=tool_fn,
                    baseline=baseline,
                    context_documents=documents,
                    oracle_intent=oracle_intent_for(task) if use_oracle else None,
                )

                decision = self._build_decision(
                    task.task_id, baseline, task, result, is_benign=True
                )
                f.write(decision.model_dump_json() + "\n")

            # === Tâches attaquées ===
            for task in tqdm(attacked_tasks, desc=f"[{baseline}] Attacked"):
                tool_fn = get_tool(task.expected_tool)
                if not tool_fn:
                    continue

                # Fusionner les arguments légitimes avec les arguments corrompus
                corrupted_args = {**task.expected_arguments, **task.corrupted_arguments}

                result = self.mediator.execute(
                    tool_name=task.expected_tool,
                    arguments=corrupted_args,
                    user_message=task.user_message,
                    tool_fn=tool_fn,
                    baseline=baseline,
                    context_documents=documents,
                    # Built from `task.expected_arguments` (the pre-attack,
                    # legitimate values), never `corrupted_args` — the oracle
                    # represents what the user authorized, not what the
                    # planner ended up proposing.
                    oracle_intent=oracle_intent_for(task) if use_oracle else None,
                )

                decision = self._build_decision(
                    task.task_id,
                    baseline,
                    task,
                    result,
                    is_benign=False,
                    attack_type=task.attack_type,
                )
                f.write(decision.model_dump_json() + "\n")

        logger.info(f"[{baseline}] Done. Logged to {log_file}")
        return log_file

    @staticmethod
    def _build_decision(
        task_id: str,
        baseline: str,
        task: BenignTask,
        result: dict,
        is_benign: bool,
        attack_type: str = None,
    ) -> MediationDecision:
        """Construit un objet MediationDecision depuis le résultat du mediator."""
        decision_data = result.get("decision") or {}
        return MediationDecision(
            task_id=task_id,
            baseline=baseline,
            tool=task.expected_tool,
            user_message=task.user_message,
            arguments=result["arguments"],
            provenance=decision_data.get("provenance", {}),
            allowed=result["allowed"],
            blocked_arguments=decision_data.get("blocked_arguments", []),
            reasons=decision_data.get("reasons", []),
            is_benign=is_benign,
            attack_type=attack_type,
            executed=result["executed"],
            execution_result=result.get("result"),
            latency_ms=result["latency_ms"],
        )
