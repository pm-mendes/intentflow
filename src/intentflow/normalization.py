"""
intentflow/normalization.py

Module de canonicalisation des valeurs d'arguments avant provenance tracking.

Pourquoi ?
Les reviews reprochent la fragilité de SequenceMatcher face aux :
- homoglyphes Unicode (а cyrillique vs a latin)
- caractères zero-width invisibles
- encodages percent-encoding (%2F, %2E)
- obfuscations email ([at], [dot])
- path traversal (../../etc/passwd)
- variantes SQL/query

Ce module normalise les valeurs AVANT le tracking de provenance,
ce qui renforce la robustesse sans changer la logique de politique.
"""

import re
import unicodedata
from urllib.parse import unquote, urlparse, urlunparse

from dateutil import parser as dateutil_parser

# ============================================================
# Normalisation Unicode générale
# ============================================================

# Table de mapping homoglyphes courants → caractère ASCII
# Source : attaques connues sur les adresses email et URLs
HOMOGLYPH_MAP = {
    # Cyrillique
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "х": "x",
    "у": "y",
    "ѕ": "s",
    "і": "i",
    "ј": "j",
    # Grec
    "α": "a",
    "ο": "o",
    "ρ": "p",
    "ν": "v",
    # Caractères similaires latins
    "ℯ": "e",
    "ℓ": "l",
    "ℴ": "o",
}

# Caractères zero-width à supprimer
ZERO_WIDTH_CHARS = [
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\ufeff",  # byte order mark / zero-width no-break space
    "\u00ad",  # soft hyphen
]


def normalize_unicode(value: str) -> str:
    """
    Normalisation Unicode NFKC + suppression zero-width + mapping homoglyphes.

    Appliquée en premier à toute valeur avant normalisation spécialisée.
    """
    if not value:
        return value

    # 1. NFKC : décompose puis recompose en forme de compatibilité
    value = unicodedata.normalize("NFKC", value)

    # 2. Supprimer caractères zero-width
    for char in ZERO_WIDTH_CHARS:
        value = value.replace(char, "")

    # 3. Mapper homoglyphes connus
    result = []
    for char in value:
        result.append(HOMOGLYPH_MAP.get(char, char))
    value = "".join(result)

    # 4. Strip et lowercase
    return value.strip().lower()


# ============================================================
# Normalisation Email
# ============================================================


def normalize_email(value: str) -> str:
    """
    Canonicalise une adresse email.

    Gère :
    - obfuscation [at] / [dot] / (at) / (dot)
    - espaces autour de @
    - unicode (via normalize_unicode)
    - punycode domaine si possible
    """
    if not value:
        return value

    # Normalisation unicode de base
    value = normalize_unicode(value)

    # Désobfuscation [at], (at), " at "
    value = re.sub(r"\s*[\[\(]at[\]\)]\s*", "@", value)
    value = re.sub(r"\s+at\s+", "@", value)

    # Désobfuscation [dot], (dot), " dot "
    value = re.sub(r"\s*[\[\(]dot[\]\)]\s*", ".", value)
    value = re.sub(r"\s+dot\s+", ".", value)

    # Supprimer espaces autour de @
    value = re.sub(r"\s*@\s*", "@", value)

    # Supprimer espaces dans le domaine
    value = re.sub(r"\s*\.\s*", ".", value)

    return value.strip()


# ============================================================
# Normalisation URL
# ============================================================


def normalize_url(value: str) -> str:
    """
    Canonicalise une URL.

    Gère :
    - percent-decoding (%2F → /, %2E → .)
    - scheme normalisation (HTTP → http)
    - hostname lowercase
    - suppression trailing slash non significatif
    - unicode (via normalize_unicode)
    """
    if not value:
        return value

    # Normalisation unicode de base (sans lowercase pour conserver la structure)
    value = unicodedata.normalize("NFKC", value)
    for char in ZERO_WIDTH_CHARS:
        value = value.replace(char, "")

    # Percent-decoding complet
    value = unquote(value)

    # Parser l'URL
    try:
        parsed = urlparse(value)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/") or "/"
        normalized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
        return normalized.strip()
    except Exception:
        return value.strip().lower()


# ============================================================
# Normalisation File Path
# ============================================================


def normalize_file_path(value: str) -> str:
    """
    Canonicalise un chemin de fichier.

    Gère :
    - percent-decoding (..%2F → ../)
    - résolution des séquences .. (path traversal)
    - normalisation des séparateurs
    - détection de tentatives de path traversal
    """
    if not value:
        return value

    # Normalisation unicode
    value = normalize_unicode(value)

    # Percent-decoding
    value = unquote(value)

    # Normaliser séparateurs Windows → Unix
    value = value.replace("\\", "/")

    # Résoudre les séquences . et .. via PurePosixPath
    try:
        # Ne pas résoudre en absolu (on ne connaît pas la racine réelle)
        parts = value.split("/")
        resolved = []
        for part in parts:
            if part == "..":
                if resolved:
                    resolved.pop()
                # Si resolved est vide, le .. tente de sortir de la racine
            elif part and part != ".":
                resolved.append(part)
        value = "/" + "/".join(resolved) if value.startswith("/") else "/".join(resolved)
    except Exception:
        pass

    return value.strip()


def is_path_traversal(value: str) -> bool:
    """
    Détecte une tentative de path traversal après normalisation.

    Retourne True si le chemin normalisé sort d'un workspace standard.
    """
    normalized = normalize_file_path(value)
    dangerous_prefixes = ["/etc/", "/proc/", "/sys/", "/root/", "/tmp/"]
    dangerous_names = ["passwd", "shadow", "sudoers", "hosts", "authorized_keys"]

    for prefix in dangerous_prefixes:
        if normalized.startswith(prefix):
            return True
    for name in dangerous_names:
        if name in normalized:
            return True
    return ".." in value  # Détection avant normalisation


# ============================================================
# Normalisation Date — Phase 1 (point 2.4 du retour reviewer)
# ============================================================


def normalize_date(value: str) -> str | None:
    """
    Canonicalise une valeur de date en ISO-8601 (YYYY-MM-DD).

    "2026-08-15", "August 15, 2026" et "15/08/2026" se normalisent tous vers
    "2026-08-15" — nécessaire pour que le tracking de provenance (comparaison
    de l'argument proposé à `Intent.parameters["date"]`) et l'évaluation par
    champ (metrics/field_comparison.py) ne traitent pas un simple changement
    de format comme un désaccord.

    Fail-closed : une valeur non interprétable comme date renvoie None plutôt
    que d'être devinée — un champ non résolu ne doit jamais devenir autorisant
    par accident.
    """
    if not value:
        return None
    try:
        parsed = dateutil_parser.parse(str(value), fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None
    return parsed.date().isoformat()


# ============================================================
# Normalisation Query / SQL / API
# ============================================================

# Mots-clés SQL dangereux à détecter
DANGEROUS_SQL_KEYWORDS = [
    "drop table",
    "drop database",
    "truncate table",
    "delete from",
    "insert into",
    "update set",
    "exec(",
    "execute(",
    "xp_cmdshell",
    "--",
    "/*",
    "*/",
    "1=1",
    "or 1",
    "union select",
    "union all select",
]


try:
    import sqlglot

    HAS_SQLGLOT = True
except ImportError:  # pragma: no cover - exercised only if sqlglot is absent
    HAS_SQLGLOT = False


def canonicalize_sql(value: str) -> str | None:
    """
    Parse a SQL string and return its canonical rendering (via sqlglot's AST,
    re-generated with consistent casing/whitespace/quoting), or None if it
    doesn't parse as SQL at all.

    Used so that "SELECT * FROM x" and "select   *   from x;" — structurally
    identical, textually different — compare equal instead of being scored as
    a mismatch (point 2.4: a deterministic SQL parser instead of free-text
    comparison). Deliberately does NOT fall back to a best-effort string
    normalization here: a query this can't parse is exactly the kind of
    "unresolved field" that `policy/sql_hardening.py` and Yulliwas's fail-closed
    principle say must not be treated as authorizing.
    """
    if not value or not HAS_SQLGLOT:
        return None
    try:
        return sqlglot.parse_one(value).sql()
    except Exception:
        return None


def normalize_query(value: str) -> str:
    """
    Normalise une query SQL ou paramètre API.

    Gère :
    - lowercase pour détection
    - suppression commentaires SQL (-- et /* */)
    - unicode
    """
    if not value:
        return value

    value = normalize_unicode(value)

    # Supprimer commentaires SQL
    value = re.sub(r"--.*$", "", value, flags=re.MULTILINE)
    value = re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL)

    return value.strip()


def is_dangerous_query(value: str) -> bool:
    """
    Détecte une query potentiellement dangereuse après normalisation.
    """
    normalized = normalize_query(value).lower()
    return any(kw in normalized for kw in DANGEROUS_SQL_KEYWORDS)


# ============================================================
# Dispatcher général
# ============================================================


def normalize_argument(argument_name: str, value: str) -> str:
    """
    Normalise une valeur selon le rôle sémantique de l'argument.

    Utilisé par ProvenanceTracker avant le matching.
    """
    if not isinstance(value, str):
        value = str(value)

    name_lower = argument_name.lower()

    if name_lower in ("recipient", "participants", "to", "cc", "bcc"):
        return normalize_email(value)

    if name_lower in ("url", "endpoint", "uri", "href", "link"):
        return normalize_url(value)

    if name_lower in ("path", "file_path", "filepath", "dest", "destination"):
        return normalize_file_path(value)

    if name_lower in ("query", "sql", "search", "predicate"):
        return normalize_query(value)

    # Défaut : normalisation unicode générale
    return normalize_unicode(value)
