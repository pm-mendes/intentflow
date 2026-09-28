"""
Configuration centralisée du projet IntentFlow.
- Valide automatiquement les types des variables d'environnement
- Fournit des valeurs par défaut documentées
- Lève des erreurs claires si une variable obligatoire est manquante
- Permet de surcharger la config via env vars ou fichier .env
"""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Paramètres globaux du projet, chargés depuis .env."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0

    # Logging
    log_level: str = "INFO"
    log_dir: str = "./data/results"

    # Expériences
    random_seed: int = 42
    num_tasks_benign: int = 10
    num_tasks_attacked: int = 10
    injected_document_ratio: float = 0.3

    # RAG
    rag_top_k: int = 3
    rag_confidence_threshold: float = 0.5

    # Pipeline de sécurité
    enable_provenance_tracking: bool = True
    enable_intent_validation: bool = True
    block_on_policy_violation: bool = True

    # Timeouts
    intent_extraction_timeout: int = 5000
    tool_execution_timeout: int = 10000


# Singleton — importé partout dans le projet
settings = Settings()
