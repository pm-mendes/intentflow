"""
ToolMediator — orchestre les composants IntentFlow.

Implémente l'Algorithm et les 7 baselines (B0-B6).

Structure d'une exécution :
1. Extraire l'intention utilisateur (si B4 ou B6)
2. Tracker la provenance de chaque argument (si B5 ou B6)
3. Valider contre les politiques (si B4, B5 ou B6)
4. Exécuter l'outil si autorisé
5. Logger la décision
"""

import time
from collections.abc import Callable
from typing import Any

from intentflow.intent.extractor import IntentExtractor
from intentflow.models.models import Intent, RAGDocument
from intentflow.policy.engine import PolicyEngine
from intentflow.policy.models import ValidationDecision
from intentflow.provenance.provenance_index import hash_arguments
from intentflow.provenance.tracker import ProvenanceTracker
from intentflow.tools.registry import TOOL_REGISTRY
from intentflow.utils.logger import logger


class ToolMediator:
    """
    Médiateur d'outils — point d'entrée principal du pipeline IntentFlow.

    Attributes:
        policy_engine: Moteur de politique pour la validation argument-level.
        intent_extractor: Extracteur d'intention utilisateur.
        documents: Corpus RAG par défaut.
    """

    BASELINES = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]

    def __init__(
        self,
        policy_engine: PolicyEngine,
        intent_extractor: IntentExtractor | None = None,
        documents: list[RAGDocument] | None = None,
    ) -> None:
        self.policy_engine = policy_engine
        self.intent_extractor = intent_extractor or IntentExtractor()
        self.documents = documents or []

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_message: str,
        tool_fn: Callable,
        baseline: str = "B6",
        context_documents: list[RAGDocument] | None = None,
        oracle_intent: Intent | None = None,
    ) -> dict[str, Any]:
        """
        Exécute un tool call avec la défense spécifiée.

        Args:
            tool_name: Nom de l'outil.
            arguments: Arguments proposés par le planner.
            user_message: Message original de l'utilisateur.
            tool_fn: Fonction à exécuter si autorisé.
            baseline: Mode de défense B0-B6.
            context_documents: Documents RAG pour ce call (override défaut).
            oracle_intent: Intent manuel pour mode oracle.

        Returns:
            Dict avec la décision, le résultat et les méta-données.
        """
        start = time.perf_counter()
        docs = context_documents or self.documents

        result: dict[str, Any] = {
            "tool": tool_name,
            "arguments": arguments,
            "user_message": user_message,
            "baseline": baseline,
            "allowed": False,
            "executed": False,
            "result": None,
            "decision": None,
            "latency_ms": 0.0,
        }

        # Sélection de la stratégie de défense
        try:
            if baseline == "B0":
                result["allowed"] = True

            elif baseline == "B1":
                # Prompt-only : on suppose que le LLM a été instruit de refuser
                # les injections. En pratique, efficacité limitée.
                result["allowed"] = True

            elif baseline == "B2":
                result["allowed"] = self._check_retrieval_filter(docs, arguments)

            elif baseline == "B3":
                result["allowed"] = tool_name in TOOL_REGISTRY

            elif baseline == "B4":
                # `mode` must be set explicitly: `extract()` defaults to
                # "llm" regardless of `oracle_intent`, so passing an oracle
                # intent here without also setting mode="oracle" silently
                # triggered a real (paid) LLM call — the same bug already
                # fixed for B6 below, just never fixed for B4.
                intent = self.intent_extractor.extract(
                    user_message,
                    mode="oracle" if oracle_intent else "llm",
                    oracle_intent=oracle_intent,
                )
                result["allowed"] = intent.action == tool_name
                result["intent"] = intent.model_dump()

            elif baseline == "B5":
                decision = self._validate_provenance_only(tool_name, arguments, user_message, docs)
                result["decision"] = decision.model_dump()
                result["allowed"] = decision.allowed

            elif baseline == "B6":
                decision = self._validate_intentflow_full(
                    tool_name, arguments, user_message, docs, oracle_intent
                )
                result["decision"] = decision.model_dump()
                result["allowed"] = decision.allowed

            else:
                raise ValueError(f"Unknown baseline: {baseline}")

        except Exception as e:
            logger.error(f"[Mediator] Error during {baseline} validation: {e}")
            result["allowed"] = False
            result["error"] = str(e)

        # Arguments effectifs : identiques aux arguments proposés, sauf en mode
        # Sanitize (B6/IF-*) où le PolicyEngine a réécrit certains d'entre eux.
        effective_args = dict(arguments)
        decision_payload = result.get("decision")
        if decision_payload:
            effective_args.update(decision_payload.get("sanitized_values", {}))
        result["effective_arguments"] = effective_args

        # Exécution de l'outil si autorisé (Allow ou Sanitize)
        result["executed"] = result["allowed"]

        # Protection TOCTOU (Section 4.4) : le hash des arguments validés par
        # le PolicyEngine (`decision.argument_hash`, calculé sur `tool` et
        # `effective_args` au moment de la décision) est re-vérifié juste
        # avant l'exécution. Un écart signifie que les arguments ont été
        # modifiés entre la décision et l'exécution — l'appel est alors
        # refusé plutôt qu'exécuté sur des valeurs qui n'ont pas été validées.
        # Avant Phase 0, `get_argument_hash`/`ProvenanceIndex` existait mais
        # n'avait aucun appelant : aucune vérification n'avait lieu ici.
        if result["executed"] and baseline == "B6" and decision_payload:
            expected_hash = decision_payload.get("argument_hash")
            if expected_hash is not None:
                actual_hash = hash_arguments({"tool": tool_name, "arguments": effective_args})
                if actual_hash != expected_hash:
                    result["executed"] = False
                    result["allowed"] = False
                    result["execution_status"] = "blocked_toctou_mismatch"
                    logger.error(
                        f"[Mediator] TOCTOU mismatch on {tool_name}: arguments changed "
                        f"between validation and execution — refusing to execute."
                    )

        if result["executed"]:
            try:
                result["result"] = tool_fn(**effective_args)
                result["execution_status"] = "success"
            except Exception as e:
                result["result"] = str(e)
                result["execution_status"] = "tool_error"
                logger.warning(f"[Mediator] Tool execution error: {e}")
        elif "execution_status" not in result:
            result["execution_status"] = "blocked"

        result["latency_ms"] = (time.perf_counter() - start) * 1000
        return result

    def _check_retrieval_filter(self, docs: list[RAGDocument], arguments: dict[str, Any]) -> bool:
        """
        B2 : Filtre de retrieval simple.
        Bloque si un argument provient d'un document connu comme injecté.
        En pratique, ce filtre suppose une connaissance a priori des docs malveillants.
        """
        for arg_value in arguments.values():
            if arg_value is None:
                continue
            value_str = str(arg_value).lower()
            for doc in docs:
                if doc.is_injected and value_str in doc.content.lower():
                    return False
        return True

    def _validate_provenance_only(
        self,
        tool: str,
        arguments: dict[str, Any],
        user_message: str,
        docs: list[RAGDocument],
    ) -> ValidationDecision:
        """
        B5 : Provenance-only (sans intent).

        Sémantique de B5 dans le contexte disambiguation :
        B5 autorise tout argument dont la provenance est un document de confiance
        (rag_document, rag_inferred, user_intent), et bloque UNIQUEMENT les arguments
        explicitement tracés vers un document injecté (injected_document).

        Cette sémantique est intentionnellement plus permissive que B6 :
        B5 ne peut pas distinguer "legal-team@company.com dans un doc interne autorisé"
        de "legal-team@company.com dans un doc interne NON autorisé par l'intent".
        C'est précisément le gap que B6 comble via l'intent conditioning.

        Différence avec le PolicyEngine complet :
        Le PolicyEngine vérifie l'origine ET la confiance ET la correspondance intent.
        B5 ici vérifie seulement : l'argument vient-il d'un document injecté ?
        Si non → Allow. Si oui → Block.
        """
        tracker = ProvenanceTracker(docs, intent=None)
        provenance = tracker.track_all_arguments(arguments, user_message)

        decision = ValidationDecision(
            allowed=True,
            tool=tool,
            arguments=arguments,
            provenance={k: v.model_dump() for k, v in provenance.items()},
        )

        for arg_name, arg_value in arguments.items():
            if arg_value is None:
                continue
            prov = provenance.get(arg_name)
            if prov and prov.origin == "injected_document":
                decision.decision = "block"
                decision.allowed = False
                decision.blocked_arguments.append(arg_name)
                decision.reasons.append(
                    f"BLOCKED: '{arg_name}' from injected_document " f"(source={prov.source_id})"
                )

        logger.info(
            f"[Policy/B5] {tool} → {'ALLOW' if decision.allowed else 'BLOCK'} "
            f"(blocked_args={decision.blocked_arguments})"
        )
        return decision

    def _validate_intentflow_full(
        self,
        tool: str,
        arguments: dict[str, Any],
        user_message: str,
        docs: list[RAGDocument],
        oracle_intent: Intent | None,
    ) -> ValidationDecision:
        """
        B6 : IntentFlow complet.
        Combine extraction d'intention + tracking de provenance + validation de politique.

        Différence clé avec B5 :
        B6 vérifie non seulement la provenance, mais aussi que chaque argument
        est cohérent avec l'intention utilisateur. Un argument tracé vers un document
        interne de confiance (rag_document) est quand même bloqué si l'utilisateur
        n'a pas explicitement autorisé cette valeur.
        Ex: legal-team@company.com est dans un doc interne → B5 l'accepte.
        Mais si l'utilisateur a dit "mon manager" → B6 le bloque car intent mismatch.
        """
        # Extraction d'intention
        intent = self.intent_extractor.extract(
            user_message,
            mode="oracle" if oracle_intent else "llm",
            oracle_intent=oracle_intent,
        )

        # Tracking de provenance avec l'intention comme référence
        tracker = ProvenanceTracker(docs, intent=intent)
        provenance = tracker.track_all_arguments(arguments, user_message)

        # Validation politique argument par argument (inclut intent conditioning)
        return self.policy_engine.validate(tool, arguments, provenance, intent=intent)
