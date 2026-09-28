"""
intentflow/intent/alias_directory.py

Mock directory for alias resolution — Point 18.

Maps user aliases to authorized values:
- "my manager" -> manager@company.com
- "legal team" -> [legal-team@company.com]
- "the database" -> https://internal-api.company.com/db

Plugs into the intent extractor to:
1. Resolve aliases before policy check
2. Add resolved_alias and resolution_confidence to intent contract
3. Unresolved alias -> Confirm
4. Resolved alias -> strict comparison against proposed argument
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from intentflow.utils.logger import logger


@dataclass
class AliasResolution:
    alias: str
    resolved_value: str | list[str]
    confidence: float
    source: str  # "directory" | "llm" | "unresolved"
    field_type: str  # "email" | "email_list" | "endpoint" | "path"


# ============================================================
# Default mock directory
# ============================================================

DEFAULT_DIRECTORY = {
    "user": {
        "value": "alice@company.com",
        "field_type": "email",
        "aliases": ["me", "myself", "the user", "I"],
    },
    "manager": {
        "value": "manager@company.com",
        "field_type": "email",
        "aliases": ["my manager", "my boss", "the manager", "direct manager",
                    "line manager", "my supervisor"],
    },
    "legal_team": {
        "value": ["legal-team@company.com"],
        "field_type": "email_list",
        "aliases": ["legal team", "legal", "the legal team", "legal department",
                    "lawyers", "law team", "legal counsel"],
    },
    "finance_team": {
        "value": ["finance@company.com"],
        "field_type": "email_list",
        "aliases": ["finance team", "finance", "the finance team",
                    "finance department", "accounting", "accounts"],
    },
    "hr_team": {
        "value": ["hr@company.com"],
        "field_type": "email_list",
        "aliases": ["hr team", "hr", "human resources", "the hr team",
                    "people team", "people ops"],
    },
    "security_team": {
        "value": ["security@company.com"],
        "field_type": "email_list",
        "aliases": ["security team", "security", "infosec", "the security team",
                    "ciso", "cyber team"],
    },
    "ceo": {
        "value": "ceo@company.com",
        "field_type": "email",
        "aliases": ["the ceo", "ceo", "chief executive", "chief executive officer"],
    },
    "cto": {
        "value": "cto@company.com",
        "field_type": "email",
        "aliases": ["the cto", "cto", "chief technology officer", "tech lead"],
    },
    "alice": {
        "value": "alice@company.com",
        "field_type": "email",
        "aliases": ["alice", "alice from the team", "project lead alice"],
    },
    "bob": {
        "value": "bob@company.com",
        "field_type": "email",
        "aliases": ["bob", "bob from the team", "bob the developer"],
    },
    "project_lead": {
        "value": "alice@company.com",
        "field_type": "email",
        "aliases": ["project lead", "the project lead", "project manager",
                    "pm", "the pm"],
    },
    "reporting_db": {
        "value": "https://internal-api.company.com/db",
        "field_type": "endpoint",
        "aliases": ["reporting database", "reporting db", "main database",
                    "main db", "the database", "database", "primary database",
                    "company database", "our database"],
    },
    "hr_system": {
        "value": "https://internal-api.company.com/hr",
        "field_type": "endpoint",
        "aliases": ["hr system", "hr database", "hr endpoint", "people system"],
    },
    "finance_api": {
        "value": "https://internal-api.company.com/finance",
        "field_type": "endpoint",
        "aliases": ["finance api", "finance system", "finance database",
                    "financial system", "finance endpoint"],
    },
    "reports_folder": {
        "value": "/reports/",
        "field_type": "path",
        "aliases": ["reports folder", "the reports folder", "report directory",
                    "output folder", "reports directory"],
    },
}


class AliasDirectory:
    """
    Resolves user aliases to authorized values.

    Usage:
        directory = AliasDirectory()
        resolution = directory.resolve("my manager", field_type="email")
        if resolution.source == "directory":
            # Use resolution.resolved_value for intent contract
        elif resolution.source == "unresolved":
            # Trigger Confirm
    """

    def __init__(self, directory: dict = None):
        self.directory = directory or DEFAULT_DIRECTORY
        # Build reverse index: alias_lower -> entry_key
        self._alias_index: dict[str, str] = {}
        for entry_key, entry in self.directory.items():
            for alias in entry.get("aliases", []):
                self._alias_index[alias.lower()] = entry_key

    def resolve(
        self,
        user_message: str,
        field_type: str = "email",
    ) -> AliasResolution | None:
        """
        Resolve an alias from a user message for a given field type.
        Returns AliasResolution or None if no alias found.
        """
        msg_lower = user_message.lower()

        # Longest alias match first
        best_alias = None
        best_key = None
        best_len = 0

        for alias_lower, entry_key in self._alias_index.items():
            entry = self.directory[entry_key]
            # Only match if field type is compatible
            entry_type = entry.get("field_type", "email")
            if field_type == "email" and entry_type not in ("email", "email_list"):
                continue
            if field_type == "endpoint" and entry_type != "endpoint":
                continue
            if field_type == "path" and entry_type != "path":
                continue

            # Word-boundary match, not a raw substring check: short aliases
            # like "I"/"me"/"hr" would otherwise match inside unrelated words
            # ("Write", "meeting", "chr..."), a latent bug only exposed once
            # this resolver got wired into the live extraction path (Phase 0) —
            # `AliasDirectory.resolve()`'s own `__main__` self-test never
            # exercised single-letter aliases against realistic prose.
            if (
                re.search(rf"\b{re.escape(alias_lower)}\b", msg_lower)
                and len(alias_lower) > best_len
            ):
                best_alias = alias_lower
                best_key = entry_key
                best_len = len(alias_lower)

        if best_key:
            entry = self.directory[best_key]
            value = entry["value"]
            conf = min(0.95, 0.75 + best_len * 0.02)
            return AliasResolution(
                alias=best_alias,
                resolved_value=value,
                confidence=conf,
                source="directory",
                field_type=entry["field_type"],
            )

        return None

    def resolve_intent_fields(
        self,
        user_message: str,
        intent_parameters: dict,
    ) -> tuple[dict, list[AliasResolution]]:
        """
        Enrich intent parameters with alias resolutions.
        Returns (enriched_parameters, resolutions).
        """
        enriched = dict(intent_parameters)
        resolutions = []

        # Try to resolve recipient/participants if missing or vague
        if not intent_parameters.get("recipient"):
            res = self.resolve(user_message, field_type="email")
            if res and res.source == "directory":
                if isinstance(res.resolved_value, list):
                    enriched["participants"] = res.resolved_value
                else:
                    enriched["recipient"] = res.resolved_value
                resolutions.append(res)
                logger.info(f"Alias resolved: '{res.alias}' -> {res.resolved_value}")

        # Try to resolve endpoint if missing
        if not intent_parameters.get("endpoint"):
            res = self.resolve(user_message, field_type="endpoint")
            if res and res.source == "directory":
                enriched["endpoint"] = res.resolved_value
                resolutions.append(res)
                logger.info(f"Endpoint alias resolved: '{res.alias}' -> {res.resolved_value}")

        # Try to resolve path if missing
        if not intent_parameters.get("path"):
            res = self.resolve(user_message, field_type="path")
            if res and res.source == "directory":
                enriched["path"] = res.resolved_value
                resolutions.append(res)
                logger.info(f"Path alias resolved: '{res.alias}' -> {res.resolved_value}")

        return enriched, resolutions


def save_directory_json(output_path: str = "artifact/policies/alias_directory.json"):
    """Export directory as JSON for the artifact."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    export = {}
    for key, entry in DEFAULT_DIRECTORY.items():
        export[key] = {
            "value": entry["value"],
            "field_type": entry["field_type"],
            "aliases": entry["aliases"],
        }
    with open(output_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Directory exported: {output_path}")
    print(f"Entries: {len(export)}")
    total_aliases = sum(len(e["aliases"]) for e in export.values())
    print(f"Total aliases: {total_aliases}")


if __name__ == "__main__":
    directory = AliasDirectory()

    print("\n=== Alias Directory Resolution Tests ===\n")
    test_cases = [
        ("Send the report to my manager.", "email", "manager@company.com"),
        ("Notify the legal team about the contract.", "email", ["legal-team@company.com"]),
        ("Forward to the finance team.", "email", ["finance@company.com"]),
        ("Run the query on the main database.", "endpoint", "https://internal-api.company.com/db"),
        ("Query the reporting database.", "endpoint", "https://internal-api.company.com/db"),
        ("Save to the reports folder.", "path", "/reports/"),
        ("Email alice about the update.", "email", "alice@company.com"),
        ("Send to the CEO.", "email", "ceo@company.com"),
        ("Query the HR system.", "endpoint", "https://internal-api.company.com/hr"),
        ("Send to an unknown person.", "email", None),  # unresolved
    ]

    passed = failed = 0
    for msg, field_type, expected in test_cases:
        res = directory.resolve(msg, field_type=field_type)
        actual = res.resolved_value if res else None
        ok = actual == expected
        status = "OK" if ok else "FAIL"
        if ok: passed += 1
        else: failed += 1
        print(f"  [{status}] '{msg[:45]:<45}' -> {actual}")

    print(f"\n{passed}/{passed+failed} passed")

    save_directory_json()