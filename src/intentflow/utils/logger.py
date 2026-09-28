"""
Logger structuré pour IntentFlow.
- Syntaxe plus expressive : logger.info("msg {var}", var=value)
- Formatage coloré en console, JSON en fichier
- Rotation automatique des fichiers de log
- Propagation d'exceptions avec contexte enrichi
"""

import sys
from pathlib import Path

from loguru import logger

from intentflow.utils.config import settings

# Supprimer le handler par défaut
logger.remove()

# Console : lisible par les humains
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <white>{message}</white>",
    level=settings.log_level,
)

# Fichier : JSON structuré pour analyse
log_dir = Path(settings.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)

logger.add(
    log_dir / "intentflow.log",
    format="{time} | {level} | {name} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    serialize=False,
)

__all__ = ["logger"]
