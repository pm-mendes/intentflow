"""
Tracker de provenance des arguments.

Implémente la propagation de labels de provenance décrite
dans la Section 4.3 du papier (lattice d'intégrité).

Algorithme de tracking (par ordre de priorité) :
1. Vérifier si la valeur correspond à l'intention utilisateur → user_intent
2. Vérifier si la valeur est dans le message utilisateur → user_intent
3. Chercher dans les documents (index Aho-Corasick, Phase 0.5) :
   3a. Span mixte (forwarded/quoted/external) qui recouvre la valeur → mixed_provenance_external
   3b. Document injecté → injected_document
   3c. Document bénin → rag_document
4. Fallback → rag_inferred (provenance inconnue)

Phase 0 : le scan naïf O(n) par `difflib.SequenceMatcher` sur tout le corpus a
été remplacé par `provenance.provenance_index.ProvenanceIndex` (Aho-Corasick) —
avant cette phase, l'index existait mais n'était jamais utilisé par ce module
(uniquement benchmarké en isolation). La détection de provenance mixte au
niveau "span" (`provenance.span_provenance`) est câblée de la même façon —
avant, elle n'était exercée que par un script d'expérience à part.

Même constat, découvert en continuant le câblage (Phase 1) : `normalization.py`
(NFKC, homoglyphes, zero-width, percent-decoding, désobfuscation email/URL/
path/SQL — exactement les mécanismes que le papier revendique en Section 4.3,
"12% ASR on bypass attacks") n'était utilisé que par le script isolé
`experiments/normalization_ablation.py`. Ce tracker comparait les valeurs avec
un simple `.lower().strip()` : un homoglyphe cyrillique ou un zero-width space
suffisait donc à échapper au matching normal, ce qui n'avait jamais été
exercé par le chemin B6 réel. Toutes les comparaisons ci-dessous (valeur vs
intent, vs message, vs documents) passent maintenant par `normalize_argument`/
`normalize_unicode`.
"""

from difflib import SequenceMatcher

from intentflow.models.models import Intent, ProvenanceSource, RAGDocument
from intentflow.normalization import normalize_argument, normalize_date, normalize_unicode
from intentflow.provenance.provenance_index import ProvenanceIndex
from intentflow.provenance.span_provenance import (
    SpanProvenanceResult,
    check_argument_against_spans,
    detect_spans,
)
from intentflow.utils.logger import logger

# Quels paramètres d'intent (`Intent.parameters`, schéma JSON du prompt LLM :
# recipient/date/participants/path/url/query/endpoint) un argument peut
# légitimement matcher pour hériter de `user_intent`. Un argument absent de
# cette table (payload : subject/body/content/description/attachment/payload)
# n'a pas d'équivalent déclaré — il ne peut pas obtenir `user_intent` par
# correspondance de paramètre (voir `_matches_intent_parameters`).
_INTENT_FIELD_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "recipient": ("recipient", "participants"),
    "participants": ("participants", "recipient"),
    "path": ("path",),
    "url": ("url", "endpoint"),
    "endpoint": ("endpoint", "url"),
    "query": ("query",),
    "date": ("date",),
}

# Trust ranking for aggregating a list-valued argument's provenance down to a
# single origin (lower = less trusted). Discovered via the AgentDojo subset
# (Point 9): a list argument (e.g. `participants`) was stringified whole
# (`str(["alice@company.com", "attacker@evil.io"])`) before comparison, so an
# attacker email appended alongside one legitimate item let the *substring*
# check (`param_str in value_str` below) match on the legitimate item and
# grant `user_intent` to the entire list — attacker value included. Each item
# must now be tracked independently and the whole list inherits its
# least-trusted item's origin, never better.
_ORIGIN_TRUST_RANK: dict[str, int] = {
    "injected_document": 0,
    "rag_inferred": 1,
    "mixed_provenance_external": 2,
    "rag_document": 3,
    "user_intent": 4,
}


class ProvenanceTracker:
    """
    Trace la provenance de chaque argument d'un tool call.

    La précision du tracking est intentionnellement conservatrice :
    en cas de doute, l'origine est marquée 'rag_inferred' (bas niveau de confiance).
    Il vaut mieux sur-bloquer que laisser passer un argument corrompu.
    """

    SIMILARITY_THRESHOLD = 0.7  # Seuil pour la correspondance approximative

    def __init__(
        self,
        documents: list[RAGDocument],
        intent: Intent | None = None,
    ) -> None:
        self.documents = documents
        self.intent = intent
        self._index = ProvenanceIndex()
        self._index.build(documents)
        # Span analysis is O(len(doc)) regex per doc; cache per doc_id since
        # the same tracker instance tracks every argument of a call against
        # the same fixed document set.
        self._span_cache: dict[str, SpanProvenanceResult] = {}
        self._normalized_content_cache: dict[str, str] = {}

    def _spans_for(self, doc: RAGDocument) -> SpanProvenanceResult:
        cached = self._span_cache.get(doc.doc_id)
        if cached is None:
            cached = detect_spans(doc)
            self._span_cache[doc.doc_id] = cached
        return cached

    def _normalized_content(self, doc: RAGDocument) -> str:
        """Document content normalized the same way as tracked values, so an
        obfuscated value (homoglyph/zero-width/percent-encoded) still matches
        the source document it was copied from."""
        cached = self._normalized_content_cache.get(doc.doc_id)
        if cached is None:
            cached = normalize_unicode(doc.content)
            self._normalized_content_cache[doc.doc_id] = cached
        return cached

    def track_all_arguments(
        self,
        arguments: dict[str, any],
        user_message: str | None = None,
    ) -> dict[str, ProvenanceSource]:
        """
        Trace la provenance de tous les arguments d'un tool call.

        Returns:
            Dictionnaire {argument_name: ProvenanceSource}
        """
        return {
            arg_name: self.track_argument(arg_name, arg_value, user_message)
            for arg_name, arg_value in arguments.items()
            if arg_value is not None
        }

    def track_argument(
        self,
        argument_name: str,
        value: any,
        user_message: str | None = None,
    ) -> ProvenanceSource:
        """
        Détermine la provenance d'un argument spécifique.

        La logique suit une cascade de vérifications du plus fiable au moins fiable.
        """
        if value is None:
            return ProvenanceSource(origin="rag_inferred", confidence=0.0)

        if isinstance(value, (list, tuple)):
            # Track each item independently rather than stringifying the whole
            # list, then take the least-trusted item's origin for the list as
            # a whole: one unauthorized item must not be laundered by the
            # presence of legitimate ones alongside it (see
            # `_ORIGIN_TRUST_RANK` above).
            if not value:
                return ProvenanceSource(origin="rag_inferred", confidence=0.0)
            item_results = [
                self.track_argument(argument_name, item, user_message) for item in value
            ]
            worst = min(
                item_results, key=lambda p: _ORIGIN_TRUST_RANK.get(p.origin, -1)
            )
            return ProvenanceSource(
                origin=worst.origin,
                source_id=worst.source_id,
                confidence=min(p.confidence for p in item_results),
                extraction_method=f"list_aggregate:{worst.extraction_method}",
            )

        value_str = normalize_argument(argument_name, value)

        # 1. Vérifier contre l'intention extraite
        if self.intent and self._matches_intent_parameters(argument_name, value_str):
            logger.debug(f"[Provenance] {argument_name}={value!r} → user_intent (intent match)")
            return ProvenanceSource(
                origin="user_intent",
                confidence=1.0,
                extraction_method="intent_match",
            )

        # 2. Vérifier contre le message utilisateur brut
        if user_message and value_str in normalize_unicode(user_message):
            logger.debug(
                f"[Provenance] {argument_name}={value!r} → user_intent (message substring)"
            )
            return ProvenanceSource(
                origin="user_intent",
                confidence=0.9,
                extraction_method="message_substring",
            )

        # 3. Chercher dans les documents RAG (index Aho-Corasick, O(len(value)))
        doc_match = self._search_in_documents(value_str)
        if doc_match:
            doc, is_injected, similarity = doc_match
            confidence = min(0.99, 0.6 + similarity * 0.39)

            # 3a. Provenance mixte au niveau span : une valeur externe encapsulée
            # dans un document par ailleurs interne/non-injecté (forwarded block,
            # citation, URL/email externe) ne doit pas hériter silencieusement de
            # la confiance "rag_document" du document englobant.
            if not is_injected:
                span_result = self._spans_for(doc)
                span_origin, span_confidence = check_argument_against_spans(value_str, span_result)
                if span_origin == "mixed_provenance_external":
                    logger.debug(
                        f"[Provenance] {argument_name}={value!r} → mixed_provenance_external "
                        f"(doc={doc.doc_id}, span-flagged)"
                    )
                    return ProvenanceSource(
                        origin="mixed_provenance_external",
                        source_id=doc.doc_id,
                        confidence=span_confidence,
                        extraction_method="span_provenance",
                    )

            origin = "injected_document" if is_injected else "rag_document"
            logger.debug(
                f"[Provenance] {argument_name}={value!r} → {origin} "
                f"(doc={doc.doc_id}, sim={similarity:.2f})"
            )
            return ProvenanceSource(
                origin=origin,
                source_id=doc.doc_id,
                confidence=confidence,
                extraction_method="rag_retrieval",
            )

        # 4. Fallback : provenance inconnue
        logger.debug(f"[Provenance] {argument_name}={value!r} → rag_inferred (no match)")
        return ProvenanceSource(origin="rag_inferred", confidence=0.2)

    def _matches_intent_parameters(self, argument_name: str, value_str: str) -> bool:
        """
        Vérifie si la valeur correspond à un paramètre de l'intention.

        Ne compare `value_str` qu'aux paramètres d'intention correspondant à
        `argument_name` (`_INTENT_FIELD_EQUIVALENTS`), pas à n'importe quel
        paramètre du contrat — trouvé en relançant Disambiguation-v2 pour la
        première fois en mode oracle réel (Phase 2/2.3) : un argument `path`
        corrompu contenant la sous-chaîne "output" matchait le paramètre
        `content`="output" (non lié) et héritait à tort de `user_intent`,
        autorisant silencieusement une redirection de chemin. Un argument
        purement "payload" (subject/body/content/description/attachment) n'a
        pas d'équivalent déclaré : il ne peut plus obtenir `user_intent` par
        ce biais (il le reste via la correspondance au message brut, étape 2,
        ou via la provenance documentaire — sa politique n'exige de toute
        façon pas `user_intent`, voir `policy/engine.py::load_default_policies`).
        """
        if not self.intent or not self.intent.parameters:
            return False

        allowed_fields = _INTENT_FIELD_EQUIVALENTS.get(argument_name)
        if not allowed_fields:
            return False

        for param_name, param_value in self.intent.parameters.items():
            if param_name not in allowed_fields or param_value is None:
                continue
            param_str = normalize_unicode(str(param_value))
            # Correspondance exacte ou inclusion
            if param_str == value_str or param_str in value_str or value_str in param_str:
                return True
            # Correspondance dans les listes (ex: participants)
            if isinstance(param_value, list):
                for item in param_value:
                    if normalize_unicode(str(item)) == value_str:
                        return True
            # Phase 1 : "date" est comparé sémantiquement (ISO vs texte
            # naturel vs slash) en plus de la comparaison de chaîne ci-dessus
            # — même correction que `metrics/field_comparison.py::dates_equal`,
            # sinon un "2026-08-15" (intent, normalisé en Phase 0.3) et un
            # "August 15, 2026" (argument synthétisé par le planner) ne
            # matcheraient jamais bien qu'ils désignent le même jour.
            if param_name == "date":
                normalized_param_date = normalize_date(str(param_value))
                if normalized_param_date and normalized_param_date == normalize_date(value_str):
                    return True
        return False

    def _search_in_documents(self, value_str: str) -> tuple[RAGDocument, bool, float] | None:
        """
        Cherche la valeur dans le corpus de documents.

        Phase 0.5 : la recherche par sous-chaîne exacte passe désormais par
        `ProvenanceIndex.lookup()` (Aho-Corasick, O(len(value_str)) amorti)
        au lieu d'un scan O(n_docs) — voir `provenance_index.py` pour le
        benchmark (38 890x à 10 000 docs). La similarité fuzzy reste un
        fallback en dernier recours, mais n'est exécutée que si l'index n'a
        rien trouvé, et seulement sur les documents restants (au lieu de
        toujours parcourir tout le corpus).
        """
        # 1. Correspondance exacte via l'index inversé (priorité absolue).
        # `value_str` est déjà normalisé (normalize_argument) ; le contenu du
        # document l'est de la même façon pour que la comparaison reste valide
        # face aux homoglyphes/zero-width/percent-encoding.
        candidates = self._index.lookup(value_str)
        for doc in candidates:
            if value_str in self._normalized_content(doc):
                return (doc, doc.is_injected, 1.0)

        # 2. Fallback fuzzy (valeurs légèrement obfusquées/reformulées,
        # non capturées par la tokenisation de l'index)
        best_match = None
        best_score = self.SIMILARITY_THRESHOLD

        for doc in self.documents:
            doc_lower = self._normalized_content(doc)
            words = doc_lower.split()
            for i in range(len(words)):
                window = " ".join(words[i : i + len(value_str.split()) + 2])
                score = SequenceMatcher(None, value_str, window).ratio()
                if score > best_score:
                    best_score = score
                    best_match = (doc, doc.is_injected, score)

        return best_match
