"""
intentflow/metrics/field_comparison.py

Comparaison sémantique par champ pour l'évaluation de l'extracteur d'intention
— Phase 1 (point 2.4 du retour reviewer).

Avant cette phase, `metrics/intent_eval_extended.py::compute_f1` comparait
TOUS les champs par égalité de chaîne exacte (insensible à la casse) :
`str(gold).lower() == str(pred).lower()`. C'est correct pour des champs
copiés mot pour mot (action, recipient, path, endpoint), mais c'est un
artefact de mesure — pas une vraie faiblesse d'extraction — pour :

- `date` : "2026-08-15" vs "August 15, 2026" vs "15/08/2026" sont la même
  date mais ne matchent jamais en comparaison de chaîne (F1 mesuré = 52.5%).
- `participants` : une liste correcte mais dans un ordre différent, ou
  partiellement couverte, échoue entièrement (F1 mesuré = 33.3%).
- `selector` (SQL) : `SELECT * FROM x` et `select * from x  ` sont identiques
  structurellement mais diffèrent en tant que chaînes (F1 mesuré = 4.1%,
  quasi garanti proche de zéro par construction de la métrique).

Ce module fournit une égalité par champ : sémantique pour date/participants/
selector, toujours une égalité de chaîne stricte pour les champs porteurs
d'autorité (recipient, path, endpoint, principal) — on ne veut PAS d'égalité
"floue" là où une différence est précisément ce qui doit faire échouer
l'autorisation.
"""

from __future__ import annotations

from intentflow.normalization import canonicalize_sql, normalize_date

# Champs comparés par égalité de chaîne stricte (insensible à la casse) —
# une différence ici EST une erreur d'extraction, pas un artefact de format.
STRICT_FIELDS = {"action", "recipient", "path", "endpoint", "principal"}


def _as_list(value: object) -> list[str]:
    """Coerce a participants-like value (list, or a comma/and-separated
    string produced by a less structured extractor) into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None]
    text = str(value)
    for sep in (" and ", ","):
        if sep in text:
            return [p.strip() for p in text.split(sep) if p.strip()]
    return [text.strip()] if text.strip() else []


def dates_equal(gold: object, pred: object) -> bool:
    """
    Compare two date-like values semantically via `normalization.normalize_date`
    instead of string equality — "2026-08-15", "August 15, 2026" and
    "15/08/2026" all normalize to the same ISO date and should score as a
    match. An unparseable value never matches (fails closed, not open).
    """
    if gold is None or pred is None:
        return False
    g = normalize_date(str(gold))
    p = normalize_date(str(pred))
    if g is None or p is None:
        return False
    return g == p


def participants_equal(gold: object, pred: object) -> bool:
    """
    Compare participant lists as sets (case-insensitive), independent of
    order or of whether the extractor returned a list or a delimited string.
    Recall on a *subset* still counts as a mismatch — this loosens formatting,
    not correctness: a proposed participant list must cover exactly the same
    set of people as gold, not a superset or a subset.
    """
    g = {p.lower() for p in _as_list(gold)}
    p = {p.lower() for p in _as_list(pred)}
    if not g and not p:
        return False  # both empty is handled by the tp/fp/fn caller, not here
    return g == p


def sql_equal(gold: object, pred: object) -> bool:
    """
    Compare two SQL strings structurally (AST equivalence via
    `normalization.canonicalize_sql`) instead of exact text — "SELECT * FROM x"
    and "select * from x" (or with different whitespace/quoting) are the same
    query. Falls back to a normalized-string comparison (case/whitespace
    -insensitive) if sqlglot can't parse either side, or isn't installed.
    """
    if gold is None or pred is None:
        return False
    gold_s, pred_s = str(gold), str(pred)

    g_canon = canonicalize_sql(gold_s)
    p_canon = canonicalize_sql(pred_s)
    if g_canon is not None and p_canon is not None:
        return g_canon == p_canon

    normalize = lambda s: " ".join(s.lower().split()).rstrip(";")  # noqa: E731
    return normalize(gold_s) == normalize(pred_s)


def values_equal(field: str, gold: object, pred: object) -> bool:
    """Dispatch to the right equality function for this field name."""
    if field == "date":
        return dates_equal(gold, pred)
    if field == "participants":
        return participants_equal(gold, pred)
    if field == "selector":
        return sql_equal(gold, pred)
    # STRICT_FIELDS and anything else: exact, case-insensitive string match.
    return str(gold).strip().lower() == str(pred).strip().lower()
