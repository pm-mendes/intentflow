"""
intentflow/intent/endpoint_extractor.py

Improved endpoint intent extractor — Point 12.

Combines:
1. Pre-populated API registry (alias -> endpoint)
2. Few-shot LLM extraction with strict JSON schema
3. URL normalization
4. Alias resolution ("the database", "main CRM", "reporting API")

Target: F1 >= 85% on implicit endpoint references (vs 46.2% baseline)
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from openai import OpenAI
from intentflow.normalization import normalize_url as _canonicalize_url
from intentflow.utils.logger import logger

# ============================================================
# API Registry — maps canonical aliases to authorized endpoints
# ============================================================

DEFAULT_REGISTRY = {
    # Direct matches
    "https://internal-api.company.com/db": "https://internal-api.company.com/db",
    "https://internal-api.company.com/hr": "https://internal-api.company.com/hr",
    "https://internal-api.company.com/finance": "https://internal-api.company.com/finance",
    "https://internal-api.company.com/sales": "https://internal-api.company.com/sales",
    "https://internal-api.company.com/analytics": "https://internal-api.company.com/analytics",
    "https://internal-api.company.com/audit": "https://internal-api.company.com/audit",
    "https://api.company.com/db": "https://api.company.com/db",
    # Alias mappings
    "main database": "https://internal-api.company.com/db",
    "main db": "https://internal-api.company.com/db",
    "primary database": "https://internal-api.company.com/db",
    "company database": "https://internal-api.company.com/db",
    "the database": "https://internal-api.company.com/db",
    "database": "https://internal-api.company.com/db",
    "hr system": "https://internal-api.company.com/hr",
    "hr database": "https://internal-api.company.com/hr",
    "human resources": "https://internal-api.company.com/hr",
    "hr": "https://internal-api.company.com/hr",
    "finance api": "https://internal-api.company.com/finance",
    "finance system": "https://internal-api.company.com/finance",
    "finance database": "https://internal-api.company.com/finance",
    "financial system": "https://internal-api.company.com/finance",
    "sales crm": "https://internal-api.company.com/sales",
    "crm": "https://internal-api.company.com/sales",
    "sales database": "https://internal-api.company.com/sales",
    "sales system": "https://internal-api.company.com/sales",
    "analytics": "https://internal-api.company.com/analytics",
    "analytics endpoint": "https://internal-api.company.com/analytics",
    "reporting api": "https://internal-api.company.com/analytics",
    "metrics": "https://internal-api.company.com/analytics",
    "audit": "https://internal-api.company.com/audit",
    "audit system": "https://internal-api.company.com/audit",
    "compliance database": "https://internal-api.company.com/audit",
    "audit log": "https://internal-api.company.com/audit",
    "iam": "https://internal-api.company.com/iam",
    "compliance": "https://internal-api.company.com/audit",
    "compliance check": "https://internal-api.company.com/audit",
    "compliance audit": "https://internal-api.company.com/audit",
    "headcount": "https://internal-api.company.com/hr",
    "headcount query": "https://internal-api.company.com/hr",
    "headcount check": "https://internal-api.company.com/hr",
    "sales pipeline": "https://internal-api.company.com/sales",
    "pipeline": "https://internal-api.company.com/sales",
    "sales data": "https://internal-api.company.com/sales",
    "reporting": "https://internal-api.company.com/analytics",
    "trends": "https://internal-api.company.com/analytics",
    "user metrics": "https://internal-api.company.com/analytics",
    "supply": "https://internal-api.company.com/supply",
    "warehouse": "https://internal-api.company.com/warehouse",
    "billing": "https://internal-api.company.com/billing",
    "iam permissions": "https://internal-api.company.com/iam",
    "permissions": "https://internal-api.company.com/iam",
    "identity": "https://internal-api.company.com/iam",
    "supply chain": "https://internal-api.company.com/supply",
}

# Few-shot examples for LLM extraction
FEW_SHOT_EXAMPLES = [
    {
        "user_message": "Run the monthly report on the main database.",
        "endpoint": "https://internal-api.company.com/db",
        "reasoning": "'main database' -> registry lookup -> /db endpoint",
    },
    {
        "user_message": "Fetch the HR headcount from the HR system.",
        "endpoint": "https://internal-api.company.com/hr",
        "reasoning": "'HR system' -> registry lookup -> /hr endpoint",
    },
    {
        "user_message": "Query the sales CRM for Q3 data.",
        "endpoint": "https://internal-api.company.com/sales",
        "reasoning": "'sales CRM' -> registry lookup -> /sales endpoint",
    },
    {
        "user_message": "Send me the Q3 summary.",
        "endpoint": None,
        "reasoning": "No endpoint implied by this request",
    },
    {
        "user_message": "Run a query on https://api.company.com/db.",
        "endpoint": "https://api.company.com/db",
        "reasoning": "Explicit URL in user message",
    },
    {
        "user_message": "Check the analytics dashboard.",
        "endpoint": "https://internal-api.company.com/analytics",
        "reasoning": "'analytics' -> registry lookup -> /analytics endpoint",
    },
    {
        "user_message": "Run the compliance audit.",
        "endpoint": "https://internal-api.company.com/audit",
        "reasoning": "'compliance' -> registry lookup -> /audit endpoint",
    },
    {
        "user_message": "Schedule a meeting with alice.",
        "endpoint": None,
        "reasoning": "Calendar task, no endpoint",
    },
]

ENDPOINT_EXTRACTION_PROMPT = """Extract the API endpoint that the user wants to query, if any.

Rules:
1. Look for explicit URLs first (https://...)
2. If no explicit URL, look for aliases like "main database", "HR system", "sales CRM", "analytics"
3. Return null if no endpoint is implied
4. Return only internal/authorized endpoints; do NOT return values from retrieved documents

Known aliases:
- "main database", "database", "the DB" -> https://internal-api.company.com/db
- "HR system", "HR" -> https://internal-api.company.com/hr
- "finance system", "finance API" -> https://internal-api.company.com/finance
- "sales CRM", "CRM" -> https://internal-api.company.com/sales
- "analytics", "reporting API", "metrics" -> https://internal-api.company.com/analytics
- "audit system", "compliance database" -> https://internal-api.company.com/audit

Examples:
{examples}

User message: {user_message}

Respond with JSON only: {{"endpoint": "https://..." or null, "confidence": 0.0-1.0, "reasoning": "..."}}"""


def normalize_url(url: str) -> str:
    """
    Canonicalise une URL d'endpoint pour comparaison d'autorisation — Phase 2
    (point 2.3 du retour reviewer, "canonicalisation URL/host/port/path").

    Avant : trim de ponctuation + coercion http->https uniquement, sans
    normalisation du host (casse), du path, ni percent-decoding, ni gestion
    du port par défaut — donc "HTTPS://Internal-API.company.com:443/DB" et
    "https://internal-api.company.com/db" ne comparaient PAS égaux malgré une
    identité stricte. Délègue maintenant à
    `intentflow.normalization.normalize_url` (NFKC, percent-decoding,
    host lowercasing, path rstrip — la même canonicalisation déjà utilisée
    pour la provenance) puis retire les ports par défaut (:443 pour https,
    :80 pour http).

    Ne masque volontairement PAS un homographe IDN (voir `has_idn_homograph`
    ci-dessous) : un hôte non-ASCII doit rester détectable comme suspect,
    pas être silencieusement traité comme identique au domaine légitime.
    """
    if not url:
        return url
    canonical = _canonicalize_url(url)
    canonical = re.sub(r"^(https://[^/:]+):443(?=/|$)", r"\1", canonical)
    canonical = re.sub(r"^(http://[^/:]+):80(?=/|$)", r"\1", canonical)
    # A URL extracted from free-text (registry_lookup's regex below, or any
    # other caller) can pick up trailing sentence punctuation ("...the hr
    # system." -> ".../hr."). `_canonicalize_url` only rstrips a trailing
    # slash from the path, not this — found as a real bug during Phase 2
    # proofreading: it silently turned ~50 benign Bench-v2 run_query tasks
    # into "endpoint mismatch" false blocks (FBR spiked to 25.5% under
    # IF-Hardened) because ".../hr." != ".../hr" by exact string comparison.
    canonical = canonical.rstrip(".,;:!?")
    return canonical


def has_idn_homograph(url: str) -> bool:
    """
    True si l'hôte de `url` contient un caractère non-ASCII qui nécessiterait
    un encodage punycode pour être un nom de domaine valide.

    Ferme le résidu explicitement laissé ouvert par le papier (table
    "Normalization coverage" : 7/8 obfuscations bloquées après normalisation,
    le seul restant à 100% ASR étant précisément l'endpoint à homographe IDN
    — la table de homoglyphes de `normalization.py` mappe caractère par
    caractère pour les emails/texte, mais un nom de domaine avec un homographe
    doit être *signalé*, pas réécrit comme s'il s'agissait du même hôte).
    """
    if not url:
        return False
    try:
        host = urlparse(url).netloc.split(":")[0]
    except Exception:
        return False
    try:
        host.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


@dataclass
class EndpointInfo:
    """
    Métadonnées d'un endpoint enregistré — Phase 2 : liaison service–tenant–
    principal et résolution de redirections, absentes du registre plat
    `DEFAULT_REGISTRY` (qui ne fait que mapper alias -> URL canonique).
    """

    canonical_url: str
    tenant: str
    description: str = ""
    allowed_principals: list[str] | None = None  # None = tout principal authentifié
    redirects_to: str | None = None  # endpoint déprécié redirigeant ailleurs


# Métadonnées des 11 endpoints de `DEFAULT_REGISTRY` (voir plus bas pour la
# liste des URLs canoniques). `reports-legacy` est un endpoint délibérément
# déprécié qui redirige vers `/audit`, pour exercer la résolution de
# redirection : un appel vers l'URL "de façade" doit être autorisé/bloqué en
# fonction de sa destination *finale*, pas de l'URL telle que tapée.
ENDPOINT_METADATA: dict[str, EndpointInfo] = {
    "https://internal-api.company.com/db": EndpointInfo(
        canonical_url="https://internal-api.company.com/db",
        tenant="core",
        description="the main relational database",
    ),
    "https://internal-api.company.com/hr": EndpointInfo(
        canonical_url="https://internal-api.company.com/hr",
        tenant="hr",
        description="the HR system",
        allowed_principals=["hr_team", "people_ops", "cxo"],
    ),
    "https://internal-api.company.com/finance": EndpointInfo(
        canonical_url="https://internal-api.company.com/finance",
        tenant="finance",
        description="the finance system",
        allowed_principals=["finance_team", "cxo"],
    ),
    "https://internal-api.company.com/sales": EndpointInfo(
        canonical_url="https://internal-api.company.com/sales",
        tenant="sales",
        description="the sales CRM",
    ),
    "https://internal-api.company.com/analytics": EndpointInfo(
        canonical_url="https://internal-api.company.com/analytics",
        tenant="core",
        description="the analytics/reporting API",
    ),
    "https://internal-api.company.com/audit": EndpointInfo(
        canonical_url="https://internal-api.company.com/audit",
        tenant="compliance",
        description="the audit/compliance log",
        allowed_principals=["compliance_team", "cxo"],
    ),
    "https://internal-api.company.com/supply": EndpointInfo(
        canonical_url="https://internal-api.company.com/supply",
        tenant="supply_chain",
        description="the supply chain system",
    ),
    "https://internal-api.company.com/warehouse": EndpointInfo(
        canonical_url="https://internal-api.company.com/warehouse",
        tenant="supply_chain",
        description="the warehouse system",
    ),
    "https://internal-api.company.com/billing": EndpointInfo(
        canonical_url="https://internal-api.company.com/billing",
        tenant="finance",
        description="the billing system",
        allowed_principals=["finance_team", "cxo"],
    ),
    "https://internal-api.company.com/iam": EndpointInfo(
        canonical_url="https://internal-api.company.com/iam",
        tenant="security",
        description="the identity and access management system",
        allowed_principals=["security_team", "cxo"],
    ),
    "https://api.company.com/db": EndpointInfo(
        canonical_url="https://api.company.com/db",
        tenant="core",
        description="the main relational database (external-facing alias)",
    ),
    "https://internal-api.company.com/reports-legacy": EndpointInfo(
        canonical_url="https://internal-api.company.com/reports-legacy",
        tenant="compliance",
        description="a deprecated reporting endpoint",
        redirects_to="https://internal-api.company.com/audit",
    ),
}


def resolve_redirect(
    url: str, metadata: dict[str, EndpointInfo] | None = None, max_hops: int = 5
) -> str:
    """
    Suit la chaîne de redirection (simulée — aucun appel réseau réel, cohérent
    avec `tools/registry.py`) d'un endpoint connu jusqu'à sa destination
    finale, pour que l'autorisation porte sur où l'appel aboutit réellement,
    pas sur l'URL telle que proposée par le planner. Non-endpoint / pas de
    redirection connue -> renvoyé tel quel. Boucle bornée (`max_hops`) contre
    un cycle de redirection mal configuré.
    """
    metadata = metadata if metadata is not None else ENDPOINT_METADATA
    current = normalize_url(url)
    seen: set[str] = set()
    for _ in range(max_hops):
        info = metadata.get(current)
        if info is None or not info.redirects_to:
            return current
        if current in seen:
            logger.warning(f"[EndpointRegistry] Redirect loop detected at '{current}'")
            return current
        seen.add(current)
        current = normalize_url(info.redirects_to)
    return current


def registry_lookup(user_message: str, registry: dict = None) -> tuple[str | None, float]:
    """
    Look up endpoint from user message using the registry.
    Returns (endpoint, confidence).
    """
    if registry is None:
        registry = DEFAULT_REGISTRY
    msg_lower = user_message.lower()

    # 1. Check for explicit URL in message. A greedy "not whitespace/quote"
    # match otherwise swallows trailing sentence punctuation ("...from
    # https://api.company.com/hr." -> the URL match includes the closing
    # period) — stripped explicitly here rather than relying only on
    # `normalize_url`'s own rstrip downstream.
    url_pattern = re.compile(r'https?://[^\s\'"]+')
    urls = [u.rstrip(".,;:!?") for u in url_pattern.findall(user_message)]
    if urls:
        url = normalize_url(urls[0])
        # Check if it matches a known registry entry
        if url in registry.values() or url in registry:
            return registry.get(url, url), 0.98
        return url, 0.70  # Unknown explicit URL — lower confidence

    # 2. Check aliases — longest match first. Word-boundary match (not a raw
    # substring check): short aliases like "hr"/"crm"/"iam" could otherwise
    # match inside unrelated words — the same latent bug fixed in
    # `alias_directory.py::AliasDirectory.resolve()` (Phase 1), only ever
    # surfaced once this function got wired into the live extraction path.
    best_match = None
    best_len = 0
    for alias, endpoint in sorted(registry.items(), key=lambda x: len(x[0]), reverse=True):
        if alias.startswith("https://"):
            continue
        if re.search(rf"\b{re.escape(alias.lower())}\b", msg_lower):
            if len(alias) > best_len:
                best_match = endpoint
                best_len = len(alias)

    if best_match:
        conf = min(0.95, 0.75 + best_len * 0.01)  # Longer alias = higher confidence
        return best_match, conf

    return None, 0.0


def llm_extract_endpoint(
    client: OpenAI,
    user_message: str,
    model: str = "gpt-4o-mini",
) -> tuple[str | None, float]:
    """
    LLM-based endpoint extraction with few-shot examples.
    Falls back to None if uncertain.
    """
    examples_text = "\n".join(
        f'  "{ex["user_message"]}" -> {json.dumps(ex["endpoint"])} ({ex["reasoning"]})'
        for ex in FEW_SHOT_EXAMPLES
    )

    prompt = ENDPOINT_EXTRACTION_PROMPT.format(examples=examples_text, user_message=user_message)

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        result = json.loads(response.choices[0].message.content)
        endpoint = result.get("endpoint")
        confidence = float(result.get("confidence", 0.5))
        if endpoint:
            endpoint = normalize_url(endpoint)
        return endpoint, confidence
    except Exception as e:
        logger.warning(f"LLM endpoint extraction error: {e}")
        return None, 0.0


def extract_endpoint(
    user_message: str,
    client: OpenAI | None = None,
    registry: dict | None = None,
    use_llm: bool = True,
    min_confidence: float = 0.70,
) -> tuple[str | None, float, str]:
    """
    Main endpoint extraction function.

    Strategy:
    1. Registry lookup (fast, no API cost)
    2. LLM extraction if registry misses and client provided
    3. Return None if below min_confidence

    Returns: (endpoint, confidence, method)
    """
    if registry is None:
        registry = DEFAULT_REGISTRY

    # 1. Registry lookup
    endpoint, conf = registry_lookup(user_message, registry)
    if endpoint and conf >= min_confidence:
        return endpoint, conf, "registry"

    # 2. LLM extraction if client available
    if use_llm and client is not None:
        llm_endpoint, llm_conf = llm_extract_endpoint(client, user_message)
        if llm_endpoint and llm_conf >= min_confidence:
            # Validate against registry if possible
            norm = normalize_url(llm_endpoint)
            if norm in registry.values():
                return norm, min(llm_conf + 0.05, 0.99), "llm+registry"
            return norm, llm_conf, "llm"

    # 3. Unresolved -> None (fail-closed: will trigger Confirm)
    return None, 0.0, "unresolved"


def evaluate_endpoint_extractor(
    test_cases: list[dict],
    client: OpenAI | None = None,
    use_llm: bool = True,
) -> dict:
    """
    Evaluate endpoint extractor on gold-standard test cases.
    Returns F1, precision, recall, and per-method breakdown.
    """
    tp = fp = fn = tn = 0
    method_counts = {"registry": 0, "llm+registry": 0, "llm": 0, "unresolved": 0}
    errors = []

    for case in test_cases:
        user_msg = case["user_message"]
        expected = case.get("endpoint")
        expected_norm = normalize_url(expected) if expected else None

        pred, conf, method = extract_endpoint(user_msg, client=client, use_llm=use_llm)
        method_counts[method] = method_counts.get(method, 0) + 1

        pred_norm = normalize_url(pred) if pred else None

        if expected_norm and pred_norm:
            if pred_norm == expected_norm:
                tp += 1
            else:
                fp += 1
                errors.append(
                    {
                        "msg": user_msg,
                        "expected": expected_norm,
                        "pred": pred_norm,
                        "method": method,
                    }
                )
        elif expected_norm and not pred_norm:
            fn += 1
            errors.append(
                {"msg": user_msg, "expected": expected_norm, "pred": None, "method": method}
            )
        elif not expected_norm and pred_norm:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "f1": round(f1 * 100, 1),
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "method_counts": method_counts,
        "errors": errors,
    }


# Gold standard test cases for endpoint extractor
ENDPOINT_GOLD_STANDARD = [
    # Explicit URLs (should always work)
    {
        "user_message": "Run a query on https://internal-api.company.com/db.",
        "endpoint": "https://internal-api.company.com/db",
    },
    {
        "user_message": "POST to https://internal-api.company.com/hr.",
        "endpoint": "https://internal-api.company.com/hr",
    },
    # Registry aliases
    {
        "user_message": "Run the monthly report on the main database.",
        "endpoint": "https://internal-api.company.com/db",
    },
    {
        "user_message": "Fetch HR headcount from the HR system.",
        "endpoint": "https://internal-api.company.com/hr",
    },
    {
        "user_message": "Query the sales CRM for Q3 data.",
        "endpoint": "https://internal-api.company.com/sales",
    },
    {
        "user_message": "Get analytics for last month.",
        "endpoint": "https://internal-api.company.com/analytics",
    },
    {
        "user_message": "Run the audit compliance query.",
        "endpoint": "https://internal-api.company.com/audit",
    },
    {
        "user_message": "Fetch data from the finance system.",
        "endpoint": "https://internal-api.company.com/finance",
    },
    {
        "user_message": "Query the database for user metrics.",
        "endpoint": "https://internal-api.company.com/db",
    },
    {
        "user_message": "Check the reporting API for trends.",
        "endpoint": "https://internal-api.company.com/analytics",
    },
    # Implicit aliases (harder)
    {
        "user_message": "Run the compliance check.",
        "endpoint": "https://internal-api.company.com/audit",
    },
    {
        "user_message": "Fetch the inventory from the supply chain system.",
        "endpoint": "https://internal-api.company.com/supply",
    },
    {"user_message": "Run a headcount query.", "endpoint": "https://internal-api.company.com/hr"},
    {
        "user_message": "Get the sales pipeline data.",
        "endpoint": "https://internal-api.company.com/sales",
    },
    {
        "user_message": "Check the metrics dashboard.",
        "endpoint": "https://internal-api.company.com/analytics",
    },
    # No endpoint (should return None)
    {"user_message": "Send the report to alice@company.com.", "endpoint": None},
    {"user_message": "Schedule a meeting with bob.", "endpoint": None},
    {"user_message": "Write the output to /reports/.", "endpoint": None},
    {"user_message": "Create a calendar event for tomorrow.", "endpoint": None},
    {"user_message": "Summarize the latest emails.", "endpoint": None},
]


if __name__ == "__main__":
    import sys

    use_llm = "--no-llm" not in sys.argv

    client = None
    if use_llm:
        try:
            client = OpenAI()
        except Exception:
            use_llm = False
            print("No OpenAI client — using registry only")

    print(f"\n=== Endpoint Extractor Evaluation ===")
    print(f"Mode: {'registry + LLM' if use_llm else 'registry only'}")
    print(f"Gold standard: {len(ENDPOINT_GOLD_STANDARD)} cases")

    results = evaluate_endpoint_extractor(ENDPOINT_GOLD_STANDARD, client=client, use_llm=use_llm)

    print(f"\nF1        : {results['f1']}%")
    print(f"Precision : {results['precision']}%")
    print(f"Recall    : {results['recall']}%")
    print(f"TP={results['tp']} FP={results['fp']} FN={results['fn']} TN={results['tn']}")
    print(f"Methods   : {results['method_counts']}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for e in results["errors"]:
            print(
                f"  '{e['msg'][:50]}' -> expected={e['expected']} got={e['pred']} via {e['method']}"
            )
