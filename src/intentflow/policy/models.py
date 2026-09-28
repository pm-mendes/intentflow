"""
Modèles pour les politiques de sécurité.

Chaque ArgumentPolicy définit :
- Quelles origines sont acceptables pour cet argument
- Quel niveau de confiance minimum est requis
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ArgumentPolicy(BaseModel):
    """Règle de validation pour un argument spécifique."""

    argument_name: str
    allowed_origins: list[str] = Field(
        default=["user_intent", "rag_document"], description="Origines de provenance acceptables"
    )
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confiance minimale de provenance requise"
    )
    role: str = Field(
        default="payload",
        description="Rôle sémantique : recipient | payload | selector | locator | amount | principal",
    )


class ToolPolicy(BaseModel):
    """Politique de sécurité complète pour un outil."""

    tool_name: str
    enabled: bool = True
    arguments: list[ArgumentPolicy] = Field(default_factory=list)


class ValidationDecision(BaseModel):
    """
    Résultat de la validation d'un appel d'outil.

    `allowed` reste le champ historique (True ssi le tool call s'exécute, donc
    pour decision in {"allow", "sanitize"}) — conservé pour compatibilité avec
    le code existant qui ne lit que ce booléen. `decision` porte la sémantique
    complète à 4 voies d'Algorithm 1 (Block ≻ Confirm ≻ Sanitize ≻ Allow),
    câblée depuis Phase 0 (auparavant seul `PolicyEngine.validate()` binaire
    Allow/Block était appelé par `ToolMediator`; `DecisionModeEngine` existait
    mais n'était utilisé que par un script d'expérience isolé).
    """

    decision: Literal["allow", "block", "confirm", "sanitize"] = "allow"
    allowed: bool
    tool: str
    arguments: dict
    blocked_arguments: list[str] = Field(default_factory=list)
    confirmed_arguments: list[str] = Field(
        default_factory=list, description="Arguments nécessitant une confirmation humaine."
    )
    sanitized_arguments: list[str] = Field(
        default_factory=list, description="Arguments réécrits avec une valeur neutre."
    )
    sanitized_values: dict[str, Any] = Field(
        default_factory=dict, description="Valeurs de remplacement appliquées (sanitize mode)."
    )
    requires_human: bool = Field(default=False, description="True si decision == 'confirm'.")
    reasons: list[str] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)
    argument_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 des arguments validés (canonical(tool,args)), calculé quand "
            "decision in {allow, sanitize}. Le mediator re-vérifie ce hash juste "
            "avant l'exécution de l'outil (protection TOCTOU, Section 4.4)."
        ),
    )
