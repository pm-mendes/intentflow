"""
intentflow/policy/decision_modes.py

Évaluation des trois modes de décision d'IntentFlow : Block, Confirm, Sanitize.

Pourquoi évaluer ces trois modes ?
Algorithm 1 mentionne Allow, Block, Sanitize, Confirm mais les reviews
remarquent que seuls Allow et Block sont évalués. Ce module implémente
et évalue les trois politiques de déploiement :

- Policy-Block    : bloque immédiatement tout argument non autorisé
- Policy-Confirm  : demande confirmation humaine pour les cas ambigus
- Policy-Sanitize : réécrit ou supprime l'argument non autorisé
- Policy-Hybrid   : Block sur injected, Confirm sur ambiguous, Sanitize sur payload

Métriques additionnelles :
- Confirm rate    : % de tâches nécessitant une confirmation humaine
- Sanitize rate   : % d'arguments réécrits
- User friction   : proxy du nombre d'interventions humaines requises
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from intentflow.models.models import Intent, ProvenanceSource
from intentflow.policy.models import ValidationDecision
from intentflow.utils.logger import logger

DecisionMode = Literal["block", "confirm", "sanitize", "hybrid"]


@dataclass
class MediationResult:
    """Résultat étendu incluant les décisions Confirm et Sanitize."""

    tool: str
    arguments: dict[str, Any]
    final_decision: Literal["allow", "block", "confirm", "sanitize"]
    blocked_arguments: list[str] = field(default_factory=list)
    confirmed_arguments: list[str] = field(default_factory=list)
    sanitized_arguments: list[str] = field(default_factory=list)
    sanitized_values: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    requires_human: bool = False
    latency_ms: float = 0.0

    @property
    def executed(self) -> bool:
        return self.final_decision in ("allow", "sanitize")

    @property
    def effective_arguments(self) -> dict[str, Any]:
        """Arguments effectifs après sanitization."""
        args = self.arguments.copy()
        args.update(self.sanitized_values)
        return args


# ============================================================
# Classificateur de risque d'argument
# Détermine si un argument doit être bloqué, confirmé ou sanitisé
# ============================================================

# Rôles à haute criticité → Block immédiat si non autorisé
HIGH_CRITICALITY_ROLES = {"recipient", "principal", "locator"}
# Rôles à criticité moyenne → Confirm si ambiguité
MEDIUM_CRITICALITY_ROLES = {"selector"}
# Rôles à faible criticité → Sanitize si non autorisé
LOW_CRITICALITY_ROLES = {"payload"}

# Mapping argument → rôle sémantique
ARGUMENT_ROLES = {
    "recipient": "recipient",
    "participants": "recipient",
    "attachment": "principal",
    "path": "locator",
    "url": "locator",
    "endpoint": "locator",
    "query": "selector",
    "body": "payload",
    "subject": "payload",
    "content": "payload",
    "description": "payload",
    "payload": "payload",
    "date": "payload",
}

# Valeurs de sanitization par défaut
SANITIZE_DEFAULTS = {
    "body": "[Content sanitized by IntentFlow]",
    "subject": "[Subject sanitized]",
    "content": "[Content sanitized by IntentFlow]",
    "description": "[Description sanitized by IntentFlow]",
    "payload": "[Payload sanitized]",
    "query": "SELECT 1",  # Requête neutre inoffensive
    "date": None,  # Supprimer la date corrompue
}


class DecisionModeEngine:
    """
    Moteur de médiation avec trois modes de décision.

    Implémente les variantes de l'Algorithm 1 :
    - Block    : MAX sécurité, friction élevée
    - Confirm  : équilibre, demande intervention humaine
    - Sanitize : MIN friction, réécrit les arguments problématiques
    - Hybrid   : Block sur critique, Confirm sur ambigu, Sanitize sur payload
    """

    def __init__(self, mode: DecisionMode = "block") -> None:
        self.mode = mode

    def mediate(
        self,
        tool: str,
        arguments: dict[str, Any],
        provenance: dict[str, ProvenanceSource],
        intent: Intent | None = None,
        policy_decision: ValidationDecision | None = None,
    ) -> MediationResult:
        """
        Applique le mode de décision sur un tool call.

        Args:
            tool: Nom de l'outil.
            arguments: Arguments proposés.
            provenance: Provenance de chaque argument.
            intent: Intention utilisateur extraite.
            policy_decision: Décision du PolicyEngine (si disponible).

        Returns:
            MediationResult avec la décision finale.
        """
        result = MediationResult(
            tool=tool,
            arguments=arguments,
            final_decision="allow",
        )

        # Identifier les arguments problématiques
        problematic = self._identify_problematic(arguments, provenance, policy_decision)

        if not problematic:
            result.final_decision = "allow"
            return result

        if self.mode == "block":
            return self._apply_block(result, problematic, arguments, provenance)
        elif self.mode == "confirm":
            return self._apply_confirm(result, problematic, arguments, provenance)
        elif self.mode == "sanitize":
            return self._apply_sanitize(result, problematic, arguments, provenance)
        elif self.mode == "hybrid":
            return self._apply_hybrid(result, problematic, arguments, provenance)
        else:
            raise ValueError(f"Unknown decision mode: {self.mode}")

    def _identify_problematic(
        self,
        arguments: dict[str, Any],
        provenance: dict[str, ProvenanceSource],
        policy_decision: ValidationDecision | None,
    ) -> list[str]:
        """Identifie les arguments qui violent la politique."""
        if policy_decision and policy_decision.blocked_arguments:
            return policy_decision.blocked_arguments

        # Fallback : identifier par provenance
        problematic = []
        for arg_name, arg_value in arguments.items():
            if arg_value is None:
                continue
            prov = provenance.get(arg_name)
            if prov and prov.origin == "injected_document":
                problematic.append(arg_name)
        return problematic

    def _apply_block(
        self,
        result: MediationResult,
        problematic: list[str],
        arguments: dict[str, Any],
        provenance: dict[str, ProvenanceSource],
    ) -> MediationResult:
        """Block : bloque tout l'appel dès qu'un argument est problématique."""
        result.final_decision = "block"
        result.blocked_arguments = problematic
        result.requires_human = False
        result.reasons = [f"BLOCK: argument '{arg}' violates policy" for arg in problematic]
        logger.info(f"[{self.mode.upper()}] {result.tool} → BLOCK {problematic}")
        return result

    def _apply_confirm(
        self,
        result: MediationResult,
        problematic: list[str],
        arguments: dict[str, Any],
        provenance: dict[str, ProvenanceSource],
    ) -> MediationResult:
        """
        Confirm : demande une confirmation humaine pour les arguments ambigus.
        Les arguments haute criticité sont bloqués sans confirmation.
        """
        high_crit = []
        needs_confirm = []

        for arg in problematic:
            role = ARGUMENT_ROLES.get(arg, "payload")
            if role in HIGH_CRITICALITY_ROLES:
                high_crit.append(arg)
            else:
                needs_confirm.append(arg)

        if high_crit:
            # Bloquer immédiatement les arguments critiques
            result.final_decision = "block"
            result.blocked_arguments = high_crit
            result.reasons = [f"BLOCK: high-criticality '{arg}'" for arg in high_crit]
        elif needs_confirm:
            # Demander confirmation pour les autres
            result.final_decision = "confirm"
            result.confirmed_arguments = needs_confirm
            result.requires_human = True
            result.reasons = [
                f"CONFIRM: argument '{arg}' requires human approval" for arg in needs_confirm
            ]
        else:
            result.final_decision = "allow"

        logger.info(
            f"[{self.mode.upper()}] {result.tool} → {result.final_decision.upper()} "
            f"(block={result.blocked_arguments}, confirm={result.confirmed_arguments})"
        )
        return result

    def _apply_sanitize(
        self,
        result: MediationResult,
        problematic: list[str],
        arguments: dict[str, Any],
        provenance: dict[str, ProvenanceSource],
    ) -> MediationResult:
        """
        Sanitize : réécrit les arguments problématiques avec des valeurs sûres.
        Les arguments haute criticité sans valeur de remplacement sont bloqués.
        """
        sanitized = {}
        blocked = []

        for arg in problematic:
            role = ARGUMENT_ROLES.get(arg, "payload")

            if role in HIGH_CRITICALITY_ROLES:
                # recipient, locator, principal → pas de valeur sûre par défaut
                blocked.append(arg)
                result.reasons.append(f"BLOCK: cannot sanitize high-criticality '{arg}'")
            elif arg in SANITIZE_DEFAULTS:
                # Réécrire avec valeur neutre
                default_val = SANITIZE_DEFAULTS[arg]
                if default_val is not None:
                    sanitized[arg] = default_val
                    result.sanitized_arguments.append(arg)
                    result.reasons.append(f"SANITIZE: '{arg}' replaced with safe default")
                else:
                    # Supprimer l'argument
                    sanitized[arg] = None
                    result.sanitized_arguments.append(arg)
                    result.reasons.append(f"SANITIZE: '{arg}' removed (null)")
            else:
                blocked.append(arg)
                result.reasons.append(f"BLOCK: no sanitization rule for '{arg}'")

        if blocked:
            result.final_decision = "block"
            result.blocked_arguments = blocked
        elif sanitized:
            result.final_decision = "sanitize"
            result.sanitized_values = sanitized
        else:
            result.final_decision = "allow"

        logger.info(
            f"[{self.mode.upper()}] {result.tool} → {result.final_decision.upper()} "
            f"(sanitized={result.sanitized_arguments}, blocked={result.blocked_arguments})"
        )
        return result

    def _apply_hybrid(
        self,
        result: MediationResult,
        problematic: list[str],
        arguments: dict[str, Any],
        provenance: dict[str, ProvenanceSource],
    ) -> MediationResult:
        """
        Hybrid : combinaison optimale.
        - injected_document + haute criticité → Block
        - injected_document + payload → Sanitize
        - rag_inferred + haute criticité → Confirm
        - rag_inferred + payload → Allow avec log
        """
        block_args = []
        confirm_args = []
        sanitize_args = {}

        for arg in problematic:
            role = ARGUMENT_ROLES.get(arg, "payload")
            prov = provenance.get(arg)
            origin = prov.origin if prov else "rag_inferred"

            if origin == "injected_document":
                if role in HIGH_CRITICALITY_ROLES:
                    block_args.append(arg)
                elif arg in SANITIZE_DEFAULTS and SANITIZE_DEFAULTS[arg] is not None:
                    sanitize_args[arg] = SANITIZE_DEFAULTS[arg]
                else:
                    block_args.append(arg)
            else:
                # Ambiguité (rag_inferred, rag_document non autorisé)
                if role in HIGH_CRITICALITY_ROLES:
                    confirm_args.append(arg)
                else:
                    # Payload non autorisé → sanitize
                    if arg in SANITIZE_DEFAULTS and SANITIZE_DEFAULTS[arg] is not None:
                        sanitize_args[arg] = SANITIZE_DEFAULTS[arg]
                    else:
                        confirm_args.append(arg)

        if block_args:
            result.final_decision = "block"
            result.blocked_arguments = block_args
            result.reasons = [f"BLOCK: '{a}' from injected source" for a in block_args]
        elif confirm_args:
            result.final_decision = "confirm"
            result.confirmed_arguments = confirm_args
            result.requires_human = True
            result.reasons = [f"CONFIRM: '{a}' ambiguous provenance" for a in confirm_args]
        elif sanitize_args:
            result.final_decision = "sanitize"
            result.sanitized_values = sanitize_args
            result.sanitized_arguments = list(sanitize_args.keys())
            result.reasons = [f"SANITIZE: '{a}' rewritten" for a in sanitize_args]
        else:
            result.final_decision = "allow"

        logger.info(
            f"[HYBRID] {result.tool} → {result.final_decision.upper()} "
            f"(block={block_args}, confirm={confirm_args}, sanitize={list(sanitize_args)})"
        )
        return result
