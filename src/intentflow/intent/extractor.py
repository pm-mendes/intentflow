"""
Extracteur d'intention utilisateur.

Deux modes :
- LLM (gpt-4o-mini) : production, expériences principales
- Heuristic : fallback sans API, utile pour les tests

Le prompt système est conçu pour forcer une sortie JSON stricte,
sans preamble ni markdown, pour permettre un parsing fiable.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from intentflow.intent.alias_directory import AliasDirectory
from intentflow.intent.endpoint_extractor import DEFAULT_REGISTRY, normalize_url, registry_lookup
from intentflow.models.models import Intent
from intentflow.normalization import canonicalize_sql, normalize_date
from intentflow.utils.logger import logger

load_dotenv()

INTENT_SYSTEM_PROMPT = """You are a security-aware intent extractor for an agentic RAG system.

Available tools:
- send_email(recipient, subject, body, attachment)
- create_calendar_event(date, participants, description)
- write_file(path, content)
- post_url(url, payload)
- run_query(query, endpoint)

Your task: extract what the USER explicitly authorized.
Be conservative: only include entities clearly stated in the message.
Do NOT infer recipients, URLs, or paths not mentioned by the user.
Only fill a field that is relevant to the extracted action's own arguments
(e.g. "participants"/"date" only for create_calendar_event, "query"/"endpoint"
only for run_query) — leave every other field null.
For "query": copy it ONLY if the user wrote an explicit SQL statement or
filter predicate verbatim. Never paraphrase a natural-language request
("get the sales numbers") into SQL yourself — that is null, not a guess.

Output ONLY valid JSON (no markdown, no preamble):
{
  "action": "<tool_name>",
  "parameters": {
    "recipient": "<email_if_stated or null>",
    "date": "<date_if_stated or null>",
    "participants": ["<email_if_stated>"],
    "path": "<path_if_stated or null>",
    "url": "<url_if_stated or null>",
    "query": "<verbatim SQL/predicate if explicitly stated, else null>",
    "endpoint": "<endpoint_if_stated or null>"
  },
  "context": "<brief implicit context>",
  "confidence": 0.9
}"""

# Phase 1 (point 2.4) : schéma déterministe des paramètres par outil, dérivé
# de la signature réelle de `tools/registry.py`. Le prompt ci-dessus liste
# TOUS les champs possibles dans un unique schéma JSON plat (nécessaire pour
# forcer une sortie structurée) — en pratique, gpt-4o-mini a tendance à
# remplir un champ non pertinent pour l'action extraite plutôt que le laisser
# `null` (ex.: "query" rempli sur un message `write_file`). C'était l'une des
# causes mesurées des F1 très bas sur `participants`/`selector` (respectivement
# 14 et 46 faux positifs sur 200 messages avant ce correctif — la plupart des
# messages n'appelant ni `create_calendar_event` ni `run_query`). Un champ
# hors du schéma de l'action extraite est donc mis à `null` après extraction :
# une contrainte typée sur le champ, pas un texte libre.
TOOL_PARAMETER_SCHEMA: dict[str, set[str]] = {
    "send_email": {"recipient", "subject", "body", "attachment"},
    "create_calendar_event": {"date", "participants", "description"},
    "write_file": {"path", "content"},
    "post_url": {"url", "payload"},
    "run_query": {"query", "endpoint"},
}

# The flat JSON schema in INTENT_SYSTEM_PROMPT's "parameters" object above —
# every field a real intent extraction could ever produce (payload fields
# like subject/body/content/description/attachment/payload are never part of
# it). Used to build oracle `Intent.parameters` from a task's
# `expected_arguments` without blindly copying payload fields into it: doing
# that caused a real cross-field false-authorization bug (`path` matching an
# unrelated `content` value that happened to share a substring — found
# running Disambiguation-v2 in oracle mode for the first time, Phase 2).
INTENT_SCHEMA_FIELDS: frozenset[str] = frozenset(
    {"recipient", "date", "participants", "path", "url", "query", "endpoint"}
)


class IntentExtractor:
    """
    Extrait l'intention utilisateur depuis un message en langage naturel.

    Attributes:
        model: Modèle OpenAI à utiliser.
        temperature: 0.0 pour des sorties déterministes (important pour reproductibilité).
        use_llm: Si False, utilise uniquement l'extraction heuristique.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        use_llm: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
        json_mode: bool = False,
        endpoint_registry: dict[str, str] | None = None,
        alias_directory: AliasDirectory | None = None,
    ) -> None:
        """
        base_url/api_key: override the OpenAI endpoint, e.g. to point at a
        self-hosted OpenAI-compatible server (Ollama: http://localhost:11434/v1,
        api_key="ollama") instead of api.openai.com.
        json_mode: request response_format={"type": "json_object"}. Off by
        default to preserve existing production behavior (gpt-4o-mini follows
        the prompt's JSON-only instruction reliably without it); weaker
        open-weight models via a self-hosted endpoint often need it to avoid
        emitting prose instead of JSON.
        """
        self.model = model
        self.temperature = temperature
        self.use_llm = use_llm
        self.json_mode = json_mode
        # Phase 0.3/0.4: registre d'endpoints + annuaire d'alias — auparavant
        # définis dans intent/endpoint_extractor.py et intent/alias_directory.py
        # mais jamais importés ici ; l'extraction réelle passait uniquement par
        # le champ JSON libre du prompt LLM ci-dessus (`parameters.endpoint`),
        # jamais canonicalisé ni vérifié contre un registre.
        self.endpoint_registry = endpoint_registry or DEFAULT_REGISTRY
        self.alias_directory = alias_directory or AliasDirectory()

        if use_llm:
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not resolved_key:
                logger.warning(
                    "OPENAI_API_KEY not set. IntentExtractor will use heuristic fallback."
                )
                self.use_llm = False
            else:
                self.client = OpenAI(api_key=resolved_key, base_url=base_url)

    def extract(
        self,
        user_message: str,
        mode: str = "llm",
        oracle_intent: Intent | None = None,
    ) -> Intent:
        """
        Extrait l'intention depuis un message utilisateur.

        Args:
            user_message: Message brut de l'utilisateur.
            mode: "oracle" | "llm" | "heuristic"
            oracle_intent: Intent manuel (utilisé si mode="oracle").

        Returns:
            Intent structuré.
        """
        if mode == "oracle" and oracle_intent is not None:
            logger.debug(f"[Intent] Oracle mode for: {user_message[:50]}...")
            oracle_intent.extraction_mode = "oracle"
            # N'enrichir que si le benchmark n'a pas déjà fixé un endpoint
            # résolu explicitement — laisse les jeux de données contrôler le
            # gold standard quand ils le souhaitent (ex.: Disambiguation-v2).
            if oracle_intent.resolved_endpoint is None:
                self._enrich_with_registries(oracle_intent, user_message)
            return oracle_intent

        if mode == "llm" and self.use_llm:
            intent = self._extract_with_llm(user_message)
        else:
            # Fallback heuristique
            intent = self._extract_heuristic(user_message)

        self._mask_parameters_to_schema(intent)
        self._enrich_with_registries(intent, user_message)
        return intent

    def _mask_parameters_to_schema(self, intent: Intent) -> None:
        """
        Null out any parameter that isn't part of `intent.action`'s tool
        schema (`TOOL_PARAMETER_SCHEMA`) — e.g. a `write_file` action should
        never carry a `query`/`participants` value. Only applied to LLM/
        heuristic extractions, not oracle (a benchmark's gold `Intent` is
        trusted as-is). An unrecognized action is left untouched: we don't
        guess a schema for a tool we don't know.
        """
        schema = TOOL_PARAMETER_SCHEMA.get(intent.action)
        if schema is None:
            return
        for key in list(intent.parameters.keys()):
            if key not in schema and intent.parameters[key] is not None:
                logger.debug(
                    f"[Intent] Masking out-of-schema parameter '{key}' for action "
                    f"'{intent.action}' (was {intent.parameters[key]!r})"
                )
                intent.parameters[key] = None

    def _enrich_with_registries(self, intent: Intent, user_message: str) -> None:
        """
        Résout `intent.resolved_endpoint` et complète `intent.parameters`
        (recipient/participants/path) via le registre d'endpoints et
        l'annuaire d'alias — en place, additif uniquement (n'écrase jamais un
        champ déjà extrait par le LLM/l'heuristique/l'oracle).

        Ne fait aucun appel LLM supplémentaire (registry_lookup et
        AliasDirectory.resolve sont des lookups en mémoire) : le coût/latence
        de l'extraction d'intention n'augmente pas.
        """
        endpoint, confidence = registry_lookup(user_message, self.endpoint_registry)
        if endpoint:
            intent.resolved_endpoint = normalize_url(endpoint)
            intent.endpoint_resolution_method = (
                "registry" if confidence >= 0.90 else "registry_low_confidence"
            )
        else:
            intent.resolved_endpoint = None
            intent.endpoint_resolution_method = "unresolved"

        enriched_params, resolutions = self.alias_directory.resolve_intent_fields(
            user_message, intent.parameters
        )
        intent.parameters = enriched_params
        for res in resolutions:
            logger.debug(f"[Intent] Alias resolved: '{res.alias}' -> {res.resolved_value}")

        # Phase 1 (point 2.4) : extraction typée pour date/participants/selector
        # au lieu de conserver le texte libre produit par le LLM tel quel.
        # Champ non résolu -> laissé tel quel (fail-closed : on ne devine pas),
        # mais canonicalisé quand c'est possible, pour que le tracking de
        # provenance (comparaison à l'argument proposé) et l'évaluation par
        # champ (metrics/field_comparison.py) ne traitent pas un simple
        # changement de format comme un désaccord ou un mismatch d'autorité.
        date_value = intent.parameters.get("date")
        if date_value:
            canonical_date = normalize_date(str(date_value))
            if canonical_date:
                intent.parameters["date"] = canonical_date

        participants = intent.parameters.get("participants")
        if isinstance(participants, str) and participants.strip():
            # A less structured extractor (heuristic, or a weaker LLM) may
            # return "alice@x.com and bob@x.com" instead of a JSON list.
            for sep in (" and ", ","):
                if sep in participants:
                    intent.parameters["participants"] = [
                        p.strip() for p in participants.split(sep) if p.strip()
                    ]
                    break

        selector_value = intent.parameters.get("query")
        if selector_value:
            canonical_sql = canonicalize_sql(str(selector_value))
            if canonical_sql:
                intent.parameters["query"] = canonical_sql

    def _extract_with_llm(self, user_message: str) -> Intent:
        """Extraction via OpenAI API."""
        try:
            kwargs = {}
            if self.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                **kwargs,
            )
            raw = response.choices[0].message.content.strip()

            # Nettoyer d'éventuels backticks markdown
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            return Intent(
                # `.get("action", "unknown")` only helps when the key is
                # absent; the LLM can also emit `"action": null` explicitly
                # (observed in practice), which `.get` happily returns as
                # None and crashes `Intent`'s validation (action: str).
                action=data.get("action") or "unknown",
                parameters={k: v for k, v in data.get("parameters", {}).items() if v is not None},
                context=data.get("context", ""),
                confidence=float(data.get("confidence", 0.8)),
                extraction_mode="llm",
            )
        except (OpenAIError, json.JSONDecodeError, KeyError, ValidationError) as e:
            logger.warning(f"[Intent] LLM extraction failed ({e}), using heuristic fallback.")
            return self._extract_heuristic(user_message)

    def _extract_heuristic(self, user_message: str) -> Intent:
        """
        Extraction heuristique basée sur des mots-clés.

        Utilisé comme fallback et dans les tests (sans API).
        Suffisant pour valider la logique de politique indépendamment du LLM.
        """
        msg = user_message.lower()

        # Détection de l'action
        if any(kw in msg for kw in ["send email", "email to", "mail to"]):
            action = "send_email"
        elif any(kw in msg for kw in ["calendar", "meeting", "schedule", "event"]):
            action = "create_calendar_event"
        elif any(kw in msg for kw in ["write file", "save file", "create file"]):
            action = "write_file"
        elif any(kw in msg for kw in ["post to", "upload", "http", "url"]):
            action = "post_url"
        elif any(kw in msg for kw in ["query", "database", "search", "run"]):
            action = "run_query"
        else:
            action = "unknown"

        # Extraction simple d'entités (email)
        import re

        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", user_message)
        params: dict = {}
        if emails:
            params["recipient"] = emails[0]
            if len(emails) > 1:
                params["participants"] = emails

        return Intent(
            action=action,
            parameters=params,
            context=user_message[:100],
            confidence=0.4,
            extraction_mode="heuristic",
        )
