"""
Modèles de données centraux d'IntentFlow.

Ces classes définissent les structures échangées entre tous les composants.
Changer un modèle ici affecte immédiatement toute la chaîne — ce qui est voulu :
cela force à réfléchir aux impacts avant de modifier une structure.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Intent(BaseModel):
    """
    Intention extraite du message utilisateur.

    L'intention est le contrat entre l'utilisateur et le pipeline de sécurité.
    Elle définit CE QUE l'utilisateur a autorisé, pas ce que le LLM veut faire.
    """

    action: str = Field(..., description="Outil principal autorisé par l'utilisateur")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Entités extraites du message (destinataires, dates, etc.)",
    )
    context: str = Field(default="", description="Contexte implicite inféré")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confiance d'extraction [0,1]"
    )
    extraction_mode: Literal["oracle", "llm", "heuristic"] = Field(
        default="llm",
        description="Mode d'extraction : oracle (manuel), llm (API), heuristic (fallback)",
    )
    resolved_endpoint: str | None = Field(
        default=None,
        description=(
            "Endpoint canonique que l'utilisateur a réellement autorisé, résolu via le "
            "registre d'endpoints/alias (intent/endpoint_extractor.py) sur le message "
            "utilisateur — indépendant du champ libre 'endpoint' de `parameters`, qui peut "
            "être un texte non canonicalisé produit par le LLM. C'est ce champ que "
            "PolicyEngine compare à l'argument `run_query.endpoint` proposé (Phase 0.3/0.4)."
        ),
    )
    endpoint_resolution_method: (
        Literal["registry", "registry_low_confidence", "unresolved"] | None
    ) = Field(
        default=None, description="Comment resolved_endpoint a été obtenu, ou None si non tenté."
    )


class ProvenanceLabel(str):
    """
    Labels de provenance formant une lattice d'intégrité.

    Ordre croissant de confiance (Eq. 1 du papier) :
    untrusted_external ⊑ user_provided ⊑ trusted_internal ⊑ trusted_system
    """

    pass


PROVENANCE_LATTICE = [
    "injected_document",  # Document manifestement malveillant
    "untrusted_external",  # Source externe non vérifiée
    "user_provided",  # Fourni directement par l'utilisateur
    "rag_document",  # Document interne récupéré par RAG
    "trusted_internal",  # Source interne de confiance
    "trusted_system",  # Système (planner, orchestrateur)
    "user_intent",  # Extrait directement de l'intention utilisateur
]


class ProvenanceSource(BaseModel):
    """
    Source de la valeur d'un argument.

    Permet de tracer chaque argument jusqu'à son origine dans le pipeline.
    C'est la pièce centrale du mécanisme d'information-flow control.
    """

    origin: Literal[
        "user_intent",
        "rag_document",
        "injected_document",
        "untrusted_external",
        "trusted_internal",
        "rag_inferred",
        "mixed_provenance_external",
    ]
    source_id: str | None = Field(None, description="Identifiant du document source")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extraction_method: str = Field(default="direct", description="Méthode de détection")


class ToolCall(BaseModel):
    """Appel d'outil proposé par le planner LLM."""

    tool: str = Field(..., description="Nom de l'outil")
    arguments: dict[str, Any] = Field(default_factory=dict)


class RAGDocument(BaseModel):
    """
    Document du corpus RAG.

    is_injected = True signale un document délibérément malveillant,
    utilisé pour générer les scénarios d'attaque. Dans un déploiement réel,
    ce flag serait inféré par le tracker de provenance, pas connu a priori.
    """

    doc_id: str
    content: str
    is_injected: bool = Field(
        default=False, description="True si document malveillant (pour expériences)"
    )
    injection_type: str | None = Field(
        None, description="Type d'injection : recipient, url, path, query"
    )
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_domain: str = Field(default="internal", description="Domaine source du document")


class BenignTask(BaseModel):
    """
    Tâche bénigne : l'utilisateur envoie une requête légitime
    et les documents RAG sont tous bénins.
    """

    task_id: str
    user_message: str = Field(..., description="Message original de l'utilisateur")
    expected_tool: str
    expected_arguments: dict[str, Any]
    context_documents: list[str] = Field(
        default_factory=list, description="IDs des documents RAG pertinents"
    )
    scenario_family: str = Field(
        default="generic", description="Famille de scénario : email, calendar, file, query"
    )


class AttackedTask(BenignTask):
    """
    Tâche attaquée : mêmes requête et outil légitimes, mais
    un ou plusieurs arguments sont corrompus par contenu injecté.

    Hérite de BenignTask pour faciliter la comparaison bénin/attaqué.
    """

    attack_type: str = Field(
        ..., description="Type d'attaque : recipient_hijack, path_traversal, etc."
    )
    corrupted_arguments: dict[str, Any] = Field(
        ..., description="Arguments remplacés par des valeurs malveillantes"
    )
    attack_payload: str = Field(default="", description="Contenu injecté dans le document")
    expected_block: bool = Field(
        default=True,
        description="IntentFlow devrait-il bloquer cette tâche ? (pour calcul des métriques)",
    )


class MediationDecision(BaseModel):
    """
    Décision du médiateur pour un appel d'outil.

    Stockée dans les logs JSONL — structure la plus importante
    pour calculer les métriques PCR, TMR, ASR, BTSR, PBR, FBR.
    """

    task_id: str
    baseline: str
    tool: str
    user_message: str
    arguments: dict[str, Any]
    provenance: dict[str, Any] = Field(default_factory=dict)

    # Décision
    allowed: bool
    blocked_arguments: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    # Contexte
    is_benign: bool
    attack_type: str | None = None
    executed: bool = False
    execution_result: Any | None = None

    # Performance
    latency_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
