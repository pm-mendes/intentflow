"""
Moteur de politique argument-level.

Implémente la Table de politiques décrite dans la Section 4.5 du papier.

Principe de conception :
- Les arguments "recipient", "path", "url", "endpoint" ne peuvent venir que
  de l'intention utilisateur (très haute confiance requise).
- Les arguments "body", "content", "description", "subject", "query" sont des
  payloads synthétisés : ils peuvent venir du RAG ou être inférés (rag_inferred)
  car le LLM les génère souvent en résumant le contexte sans copier mot pour mot.
- Un argument d'un document INJECTÉ est toujours bloqué, quelle que soit la règle.

Phase 0 (câblage code <-> papier) : `validate()` intègre maintenant, dans le
chemin réel emprunté par B6, trois mécanismes qui existaient déjà dans le
dépôt mais n'étaient jamais appelés par ce module avant cette phase :
- `policy/sql_hardening.py::analyze_sql` (durcissement SQL du rôle "selector").
- `intent/endpoint_extractor.py` (registre d'endpoints, via `Intent.resolved_endpoint`
  calculé par `intent/extractor.py`), pour authentifier `run_query.endpoint`.
- `policy/decision_modes.py` (rôles/criticité), pour porter la décision à 4
  voies Allow/Block/Confirm/Sanitize au lieu du seul Allow/Block d'avant.

IF-Core / IF-Hardened / IF-Confirm sont désormais des configurations nommées
de `PolicyEngine` (classmethods ci-dessous) — auparavant ces trois variantes
étaient trois implémentations ad hoc dupliquées dans
`experiments/if_core_vs_hardened.py`, `experiments/e2e_agent_eval_v2.py` et
`experiments/run_adaptive_eval.py`.
"""

from typing import Any, Literal

from intentflow.models.models import Intent, ProvenanceSource
from intentflow.policy.decision_modes import HIGH_CRITICALITY_ROLES, SANITIZE_DEFAULTS
from intentflow.policy.models import ArgumentPolicy, ToolPolicy, ValidationDecision
from intentflow.policy.sql_hardening import analyze_sql
from intentflow.provenance.provenance_index import hash_arguments
from intentflow.utils.logger import logger

DecisionMode = Literal["block", "confirm", "sanitize", "hybrid"]


class PolicyEngine:
    """
    Valide les appels d'outils contre les politiques de sécurité.

    Les politiques par défaut encodent les règles décrites dans le papier.
    Elles peuvent être surchargées via configs/policies.yaml pour des déploiements
    avec des contraintes différentes.

    Args:
        policies: table de politiques par outil (voir `load_default_policies`).
        decision_mode: comment traiter un argument "problématique" (origine non
            autorisée, confiance insuffisante, endpoint non résolu, SQL suspect)
            qui n'est pas un blocage absolu (injection, SQL dangereux) :
            "block" (tout problématique -> Block, comportement historique),
            "confirm" (critique -> Block, sinon -> Confirm),
            "sanitize" (critique -> Block, sinon réécrit si un défaut sûr existe),
            "hybrid" (Block sur injecté/critique-non-ambigu, Confirm sur
            ambigu critique, Sanitize sur payload — recommandé, IF-Confirm).
        enable_sql_hardening: appelle `sql_hardening.analyze_sql` sur les
            arguments de rôle "selector" (ex. `run_query.query`).
        enable_endpoint_resolution: authentifie `run_query.endpoint` contre
            `Intent.resolved_endpoint` (registre + alias) au lieu de se fier
            uniquement à l'origine de provenance.
        fail_closed_unknown_tool: si True, un outil sans politique est bloqué
            (au lieu d'être autorisé par défaut — le fail-open historique,
            Section 2.3 du retour reviewer).
    """

    def __init__(
        self,
        policies: dict[str, ToolPolicy],
        decision_mode: DecisionMode = "block",
        enable_sql_hardening: bool = False,
        enable_endpoint_resolution: bool = False,
        fail_closed_unknown_tool: bool = False,
    ) -> None:
        self.policies = policies
        self.decision_mode = decision_mode
        self.enable_sql_hardening = enable_sql_hardening
        self.enable_endpoint_resolution = enable_endpoint_resolution
        self.fail_closed_unknown_tool = fail_closed_unknown_tool

    @classmethod
    def load_default_policies(cls) -> "PolicyEngine":
        """
        Charge les politiques par défaut décrites dans le papier (Section 4.5).

        Rationale des politiques par rôle d'argument :

        RECIPIENT / LOCATOR / PRINCIPAL (haute criticité) :
        - send_email.recipient   → user_intent uniquement (conf ≥ 0.85)
        - send_email.attachment  → user_intent uniquement (conf ≥ 0.90)
        - write_file.path        → user_intent uniquement (conf ≥ 0.90)
        - run_query.endpoint     → user_intent uniquement (conf ≥ 0.90)
        - create_calendar_event.participants → user_intent uniquement (conf ≥ 0.80)

        PAYLOAD (faible criticité) :
        - send_email.subject/body → user_intent | rag_document | rag_inferred
          Justification : le LLM synthétise le sujet et le corps depuis le contexte
          sans nécessairement copier mot pour mot un document. rag_inferred est
          acceptable car ces arguments ne contrôlent pas la destination.
        - create_calendar_event.description → idem
        - write_file.content               → idem
        - run_query.query                  → idem (le SELECT est construit par le LLM)

        LOCATOR EXTERNE (criticité moyenne) :
        - post_url.url → user_intent | rag_document (conf ≥ 0.75)
          On tolère rag_document pour les URLs dans des docs internes de confiance,
          mais pas rag_inferred (une URL inconnue ne doit pas passer).
        """
        policies = {
            "send_email": ToolPolicy(
                tool_name="send_email",
                arguments=[
                    ArgumentPolicy(
                        argument_name="recipient",
                        allowed_origins=["user_intent"],
                        min_confidence=0.85,
                        role="recipient",
                    ),
                    ArgumentPolicy(
                        argument_name="subject",
                        # Payload synthétisé : rag_inferred autorisé
                        allowed_origins=["user_intent", "rag_document", "rag_inferred"],
                        min_confidence=0.0,
                        role="payload",
                    ),
                    ArgumentPolicy(
                        argument_name="body",
                        # Payload synthétisé : rag_inferred autorisé
                        allowed_origins=["user_intent", "rag_document", "rag_inferred"],
                        min_confidence=0.0,
                        role="payload",
                    ),
                    ArgumentPolicy(
                        argument_name="attachment",
                        allowed_origins=["user_intent"],
                        min_confidence=0.90,
                        role="principal",
                    ),
                ],
            ),
            "create_calendar_event": ToolPolicy(
                tool_name="create_calendar_event",
                arguments=[
                    ArgumentPolicy(
                        argument_name="date",
                        allowed_origins=["user_intent", "rag_document"],
                        min_confidence=0.5,
                        role="payload",
                    ),
                    ArgumentPolicy(
                        argument_name="participants",
                        allowed_origins=["user_intent"],
                        min_confidence=0.80,
                        role="recipient",
                    ),
                    ArgumentPolicy(
                        argument_name="description",
                        # Payload synthétisé : rag_inferred autorisé
                        allowed_origins=["user_intent", "rag_document", "rag_inferred"],
                        min_confidence=0.0,
                        role="payload",
                    ),
                ],
            ),
            "write_file": ToolPolicy(
                tool_name="write_file",
                arguments=[
                    ArgumentPolicy(
                        argument_name="path",
                        allowed_origins=["user_intent"],
                        min_confidence=0.90,
                        role="locator",
                    ),
                    ArgumentPolicy(
                        argument_name="content",
                        # Payload synthétisé : rag_inferred autorisé
                        allowed_origins=["user_intent", "rag_document", "rag_inferred"],
                        min_confidence=0.0,
                        role="payload",
                    ),
                ],
            ),
            "post_url": ToolPolicy(
                tool_name="post_url",
                arguments=[
                    ArgumentPolicy(
                        argument_name="url",
                        # URL externe : rag_document toléré (doc interne) mais pas rag_inferred
                        allowed_origins=["user_intent", "rag_document"],
                        min_confidence=0.75,
                        role="locator",
                    ),
                    ArgumentPolicy(
                        argument_name="payload",
                        allowed_origins=["user_intent", "rag_document", "rag_inferred"],
                        min_confidence=0.0,
                        role="payload",
                    ),
                ],
            ),
            "run_query": ToolPolicy(
                tool_name="run_query",
                arguments=[
                    ArgumentPolicy(
                        argument_name="query",
                        # Le SELECT est construit par le LLM : rag_inferred autorisé
                        allowed_origins=["user_intent", "rag_document", "rag_inferred"],
                        min_confidence=0.0,
                        role="selector",
                    ),
                    ArgumentPolicy(
                        argument_name="endpoint",
                        allowed_origins=["user_intent", "rag_document"],
                        min_confidence=0.85,
                        role="locator",
                    ),
                ],
            ),
        }
        return cls(policies)

    @classmethod
    def if_core(cls) -> "PolicyEngine":
        """
        IF-Core : intent + provenance + rôles, sans durcissement additionnel.
        Alias explicite de `load_default_policies()` — même politique, même
        comportement (decision_mode="block", pas de SQL hardening, pas de
        résolution d'endpoint, fail-open sur outil inconnu) ; nommé pour
        correspondre à la variante système du papier plutôt que de forcer les
        scripts d'expérience à connaître l'implémentation par défaut.
        """
        return cls.load_default_policies()

    @classmethod
    def if_hardened(cls) -> "PolicyEngine":
        """
        IF-Hardened : IF-Core + registre d'endpoints + durcissement SQL AST +
        provenance mixte au niveau span (câblée dans `ProvenanceTracker`, donc
        automatique dès que cette instance est utilisée) + fail-closed sur
        outil/endpoint inconnu. Décisions toujours binaires Allow/Block
        (pas de Confirm) : un endpoint non résolu est bloqué plutôt
        qu'autorisé — c'est la correction du bug fail-open de la Section 2.3,
        mais sans le mode Confirm ce réglage sur-bloque (voir l'ablation
        endpoint dans `experiments/run_endpoint_ablation_n30.py`).
        """
        engine = cls.load_default_policies()
        engine.decision_mode = "block"
        engine.enable_sql_hardening = True
        engine.enable_endpoint_resolution = True
        engine.fail_closed_unknown_tool = True
        return engine

    @classmethod
    def if_confirm(cls) -> "PolicyEngine":
        """
        IF-Confirm (recommandé pour la production) : IF-Hardened + un
        argument ambigu mais non manifestement malveillant (endpoint non
        résolu, origine/confiance insuffisante sur un rôle critique, requête
        SQL "suspecte" plutôt que dangereuse) est routé vers Confirm au lieu
        d'un Block silencieux — seuls l'injection avérée, un mismatch
        d'endpoint avéré (valeur ≠ endpoint réellement autorisé par l'intent)
        et le SQL manifestement dangereux restent des Block absolus.
        """
        engine = cls.load_default_policies()
        engine.decision_mode = "hybrid"
        engine.enable_sql_hardening = True
        engine.enable_endpoint_resolution = True
        engine.fail_closed_unknown_tool = True
        return engine

    def validate(
        self,
        tool: str,
        arguments: dict[str, Any],
        provenance: dict[str, ProvenanceSource],
        intent: Intent | None = None,
    ) -> ValidationDecision:
        """
        Valide un appel d'outil argument par argument.

        Logique de décision :
        1. Outil inconnu → `fail_closed_unknown_tool` décide (Block ou Allow-par-défaut).
        2. Outil désactivé → bloquer.
        3. Pour chaque argument, dans l'ordre :
           a. Origine injected_document → Block absolu (invariant utilisé par
              `scripts/logs_to_sigma.py`, ne dépend jamais de `decision_mode`).
           b. Rôle "selector" + SQL hardening actif → `analyze_sql` : dangereux
              → Block absolu ; suspect → candidat "problématique".
           c. `run_query.endpoint` + résolution d'endpoint active → comparé à
              `intent.resolved_endpoint` : mismatch avéré → Block absolu
              (trusted-but-unauthorized, Section 2.3) ; non résolu → candidat
              "problématique endpoint" (traité différemment du (d) ci-dessous :
              toujours Confirm-éligible en mode confirm/hybrid, même si
              "locator" est un rôle à haute criticité).
           d. Pas de provenance / origine non autorisée / confiance insuffisante
              → candidat "problématique générique", résolu selon `decision_mode`
              et la criticité du rôle (Section 4.5 / `decision_modes.py`).
        4. Agrégation : Block ≻ Confirm ≻ Sanitize ≻ Allow (Algorithm 1).

        Returns:
            ValidationDecision avec `decision`/`allowed` et le détail des
            arguments bloqués/à confirmer/sanitisés.
        """
        decision = ValidationDecision(
            decision="allow",
            allowed=True,
            tool=tool,
            arguments=arguments,
            provenance={k: v.model_dump() for k, v in provenance.items()},
        )

        # Vérifier que l'outil est connu et activé
        if tool not in self.policies:
            if self.fail_closed_unknown_tool:
                decision.decision = "block"
                decision.allowed = False
                decision.reasons.append(f"BLOCKED: tool '{tool}' unknown (fail-closed)")
            else:
                logger.warning(f"[Policy] Tool '{tool}' not in policies → allow (unknown tool)")
                decision.reasons.append(f"Tool '{tool}' not in policies (allowed by default)")
            return decision

        if not self.policies[tool].enabled:
            decision.decision = "block"
            decision.allowed = False
            decision.reasons.append(f"Tool '{tool}' is disabled by policy")
            return decision

        tool_policy = self.policies[tool]
        arg_policies = {p.argument_name: p for p in tool_policy.arguments}

        block_args: list[str] = []
        endpoint_confirm_candidates: list[str] = []
        generic_confirm_candidates: list[str] = []

        for arg_name, arg_value in arguments.items():
            if arg_value is None:
                continue

            arg_prov = provenance.get(arg_name)

            # (a) Règle prioritaire absolue : tout argument injecté est bloqué
            if arg_prov and arg_prov.origin == "injected_document":
                block_args.append(arg_name)
                decision.reasons.append(
                    f"BLOCKED: argument '{arg_name}' comes from injected document "
                    f"(source={arg_prov.source_id})"
                )
                continue

            if arg_name not in arg_policies:
                continue  # Pas de politique pour cet argument → autorisé

            arg_policy = arg_policies[arg_name]
            role = arg_policy.role

            # (b) Durcissement SQL — indépendant de la provenance : une requête
            # dangereuse reste dangereuse même si son origine est autorisée.
            if self.enable_sql_hardening and role == "selector":
                sql_result = analyze_sql(str(arg_value))
                if sql_result.decision == "block":
                    block_args.append(arg_name)
                    decision.reasons.append(
                        f"BLOCKED: '{arg_name}' SQL hardening: {sql_result.reason}"
                    )
                    continue
                if sql_result.decision == "confirm":
                    generic_confirm_candidates.append(arg_name)
                    decision.reasons.append(
                        f"FLAGGED: '{arg_name}' SQL hardening suspicious: {sql_result.reason}"
                    )
                    continue

            # (c) Authentification de l'endpoint contre l'intent résolu —
            # remplace la simple vérification d'origine pour ce champ
            # spécifique : un endpoint peut être "trusted" (rag_document) tout
            # en étant différent de celui que l'utilisateur a demandé.
            if self.enable_endpoint_resolution and tool == "run_query" and arg_name == "endpoint":
                status, human_reason = self._check_endpoint_authorization(arg_value, intent)
                if status == "mismatch":
                    block_args.append(arg_name)
                    decision.reasons.append(f"BLOCKED: '{arg_name}' — {human_reason}")
                    continue
                if status == "unresolved":
                    endpoint_confirm_candidates.append(arg_name)
                    decision.reasons.append(f"FLAGGED: '{arg_name}' — {human_reason}")
                    continue
                # status == "authorized": retombe sur la vérification standard ci-dessous.

            # (d) Pas de provenance et argument avec politique → bloquer par précaution
            if arg_prov is None:
                block_args.append(arg_name)
                decision.reasons.append(f"BLOCKED: no provenance info for argument '{arg_name}'")
                continue

            if arg_prov.origin not in arg_policy.allowed_origins:
                generic_confirm_candidates.append(arg_name)
                decision.reasons.append(
                    f"FLAGGED: '{arg_name}' origin='{arg_prov.origin}' "
                    f"not in allowed={arg_policy.allowed_origins}"
                )
                continue

            if arg_prov.confidence < arg_policy.min_confidence:
                generic_confirm_candidates.append(arg_name)
                decision.reasons.append(
                    f"FLAGGED: '{arg_name}' confidence={arg_prov.confidence:.2f} "
                    f"< required={arg_policy.min_confidence}"
                )

        self._resolve_decision(
            decision,
            arg_policies,
            block_args,
            endpoint_confirm_candidates,
            generic_confirm_candidates,
        )

        if decision.decision in ("allow", "sanitize"):
            effective_args = dict(arguments)
            effective_args.update(decision.sanitized_values)
            decision.argument_hash = hash_arguments({"tool": tool, "arguments": effective_args})

        logger.info(
            f"[Policy] {tool} → {decision.decision.upper()} "
            f"(blocked={decision.blocked_arguments}, confirm={decision.confirmed_arguments}, "
            f"sanitize={decision.sanitized_arguments})"
        )
        return decision

    def _check_endpoint_authorization(
        self, arg_value: Any, intent: Intent | None
    ) -> tuple[Literal["authorized", "mismatch", "unresolved"], str]:
        """
        Compare l'endpoint proposé au canonique résolu depuis l'intention
        utilisateur (`Intent.resolved_endpoint`, câblé en Phase 0.3 dans
        `intent/extractor.py` via le registre `intent/endpoint_extractor.py`).

        Renvoie `(statut, message)` — le message est écrit pour être
        compréhensible par un humain (Section 2.3 : "mode Confirm uniquement
        lorsque l'utilisateur peut comprendre l'effet concret"), pas un code
        interne. Phase 2 :
        - un hôte à homographe IDN est un mismatch immédiat, jamais un simple
          "non résolu" (résidu explicitement laissé ouvert par le papier) ;
        - la résolution de redirection (`resolve_redirect`) s'applique aux
          deux côtés avant comparaison, pour qu'un endpoint-façade déprécié
          soit jugé sur sa destination réelle.
        """
        from intentflow.intent.endpoint_extractor import has_idn_homograph, resolve_redirect

        arg_str = str(arg_value)
        if has_idn_homograph(arg_str):
            return (
                "mismatch",
                f"'{arg_str}' uses a non-ASCII hostname characteristic of an IDN "
                f"homograph attack (a look-alike character substituted into a "
                f"trusted domain) — blocked regardless of provenance.",
            )

        if intent is None or intent.resolved_endpoint is None:
            return (
                "unresolved",
                f"the endpoint '{arg_str}' could not be matched to anything you "
                f"authorized in your request — confirm if this is the service you "
                f"meant, otherwise the call is refused.",
            )

        resolved_arg = resolve_redirect(arg_str)
        resolved_intent = resolve_redirect(intent.resolved_endpoint)
        if resolved_arg == resolved_intent:
            return "authorized", "endpoint matches the one you authorized"

        via_redirect = f" (via a redirect from '{arg_str}')" if resolved_arg != arg_str else ""
        return (
            "mismatch",
            f"the call would actually reach '{resolved_arg}'{via_redirect}, not "
            f"'{resolved_intent}' as you authorized — blocked as trusted-but-unauthorized.",
        )

    def _resolve_decision(
        self,
        decision: ValidationDecision,
        arg_policies: dict[str, ArgumentPolicy],
        block_args: list[str],
        endpoint_confirm_candidates: list[str],
        generic_confirm_candidates: list[str],
    ) -> None:
        """
        Applique `decision_mode` aux candidats "problématiques" (ni absolus,
        ni déjà autorisés) puis agrège selon la priorité Block ≻ Confirm ≻
        Sanitize ≻ Allow (Algorithm 1).
        """
        confirm_args: list[str] = []
        sanitize_values: dict[str, Any] = {}
        mode = self.decision_mode

        # Endpoint non résolu : toujours Confirm-éligible en mode confirm/hybrid,
        # même si "locator" est classé à haute criticité pour le cas générique
        # (un endpoint ambigu se prête à une confirmation humaine compréhensible ;
        # un mismatch avéré est déjà passé par `block_args` plus haut).
        for arg in endpoint_confirm_candidates:
            if mode in ("confirm", "hybrid"):
                confirm_args.append(arg)
            else:  # "block" ou "sanitize" : pas de valeur sûre pour un endpoint
                block_args.append(arg)

        for arg in generic_confirm_candidates:
            role = arg_policies[arg].role if arg in arg_policies else "payload"
            high_crit = role in HIGH_CRITICALITY_ROLES
            safe_default = SANITIZE_DEFAULTS.get(arg)

            if mode == "block":
                block_args.append(arg)
            elif mode == "confirm":
                (block_args if high_crit else confirm_args).append(arg)
            elif mode == "sanitize":
                if high_crit:
                    block_args.append(arg)
                elif safe_default is not None:
                    sanitize_values[arg] = safe_default
                else:
                    block_args.append(arg)
            elif mode == "hybrid":
                if high_crit:
                    confirm_args.append(arg)
                elif safe_default is not None:
                    sanitize_values[arg] = safe_default
                else:
                    confirm_args.append(arg)
            else:
                raise ValueError(f"Unknown decision mode: {mode}")

        block_args = sorted(set(block_args))
        confirm_args = sorted(set(confirm_args) - set(block_args))

        if block_args:
            decision.decision = "block"
            decision.allowed = False
            decision.blocked_arguments = block_args
        elif confirm_args:
            decision.decision = "confirm"
            decision.allowed = False
            decision.confirmed_arguments = confirm_args
            decision.requires_human = True
        elif sanitize_values:
            decision.decision = "sanitize"
            decision.allowed = True
            decision.sanitized_values = sanitize_values
            decision.sanitized_arguments = sorted(sanitize_values.keys())
        else:
            decision.decision = "allow"
            decision.allowed = True
