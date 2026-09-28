"""
Outils agentic simulés pour les expériences IntentFlow.

Ces outils ne font pas d'appels réseau réels : ils simulent le comportement
d'outils réels (validation des arguments, retour de résultats structurés)
sans effets de bord externes.

Important pour la publication MIT : n'importe qui peut exécuter les expériences
sans credentials externes ni configuration réseau.
"""

import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from intentflow.utils.logger import logger


def send_email(
    recipient: str,
    subject: str,
    body: str,
    attachment: str | None = None,
) -> dict[str, Any]:
    """
    Simule l'envoi d'un email.

    Valide le format de l'adresse email — une adresse invalide est le signe
    d'un argument corrompu par injection.
    """
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", recipient):
        raise ValueError(f"Invalid recipient email: '{recipient}'")
    logger.info(f"[TOOL] send_email → {recipient}")
    return {
        "status": "sent",
        "recipient": recipient,
        "subject": subject,
        "has_attachment": attachment is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


def create_calendar_event(
    date: str,
    participants: list[str],
    description: str,
) -> dict[str, Any]:
    """Simule la création d'un événement calendrier."""
    if not participants:
        raise ValueError("At least one participant required")
    logger.info(f"[TOOL] create_calendar_event on {date} with {participants}")
    return {
        "status": "created",
        "date": date,
        "participant_count": len(participants),
        "participants": participants,
    }


def write_file(path: str, content: str) -> dict[str, Any]:
    """
    Simule l'écriture d'un fichier.

    Vérifie que le chemin ne contient pas de path traversal ('../').
    """
    if not path or ".." in path:
        raise ValueError(f"Dangerous or invalid file path: '{path}'")
    logger.info(f"[TOOL] write_file → {path}")
    return {
        "status": "written",
        "path": path,
        "size_bytes": len(content.encode("utf-8")),
    }


def post_url(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Simule un POST HTTP vers une URL."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL scheme: '{url}'")
    logger.info(f"[TOOL] post_url → {url}")
    return {"status": "posted", "url": url, "http_status": 200}


def run_query(query: str, endpoint: str) -> dict[str, Any]:
    """Simule l'exécution d'une requête sur un endpoint."""
    if not query or not endpoint:
        raise ValueError("query and endpoint are required")
    logger.info(f"[TOOL] run_query on {endpoint}: {query[:50]}")
    return {"status": "executed", "endpoint": endpoint, "rows_returned": 42}


# Registre central — utilisé par le mediator et le runner de baselines
TOOL_REGISTRY: dict[str, Callable] = {
    "send_email": send_email,
    "create_calendar_event": create_calendar_event,
    "write_file": write_file,
    "post_url": post_url,
    "run_query": run_query,
}


def get_tool(tool_name: str) -> Callable | None:
    """Retourne la fonction d'outil par son nom."""
    return TOOL_REGISTRY.get(tool_name)
