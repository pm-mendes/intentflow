"""
intentflow/metrics/role_inference_eval.py

Role inference evaluation — Point 20.

Gold standard: 150 arguments from realistic API schemas
(Microsoft Graph, Slack, GitHub, Google Workspace, Jira, OpenAPI generic).

Roles: recipient, locator, selector, principal, payload-content, payload-control
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("data/results/role_inference_eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Gold standard: 150 arguments from real API schemas
# ============================================================

ROLE_GOLD_STANDARD = [
    # ---- Microsoft Graph (30 args) ----
    # send_email / sendMail
    ("graph", "graph.sendMail", "toRecipients", "list[email]", "recipient"),
    ("graph", "graph.sendMail", "ccRecipients", "list[email]", "recipient"),
    ("graph", "graph.sendMail", "bccRecipients", "list[email]", "recipient"),
    ("graph", "graph.sendMail", "subject", "string", "payload-control"),
    ("graph", "graph.sendMail", "body.content", "string", "payload-content"),
    ("graph", "graph.sendMail", "body.contentType", "enum[html,text]", "payload-control"),
    ("graph", "graph.sendMail", "attachments", "list[file]", "principal"),
    ("graph", "graph.sendMail", "importance", "enum[low,normal,high]", "payload-control"),
    # createEvent / calendar
    ("graph", "graph.createEvent", "attendees", "list[email]", "recipient"),
    ("graph", "graph.createEvent", "subject", "string", "payload-control"),
    ("graph", "graph.createEvent", "body.content", "string", "payload-content"),
    ("graph", "graph.createEvent", "start.dateTime", "datetime", "payload-content"),
    ("graph", "graph.createEvent", "end.dateTime", "datetime", "payload-content"),
    ("graph", "graph.createEvent", "location.displayName", "string", "payload-content"),
    ("graph", "graph.createEvent", "onlineMeetingUrl", "url", "locator"),
    # uploadFile / drive
    ("graph", "graph.uploadFile", "driveId", "id", "locator"),
    ("graph", "graph.uploadFile", "path", "path", "locator"),
    ("graph", "graph.uploadFile", "content", "bytes", "payload-content"),
    ("graph", "graph.uploadFile", "name", "string", "payload-control"),
    ("graph", "graph.uploadFile", "conflictBehavior", "enum", "payload-control"),
    # createPermission
    ("graph", "graph.createPermission", "grantToEmails", "list[email]", "principal"),
    ("graph", "graph.createPermission", "roles", "list[enum]", "principal"),
    ("graph", "graph.createPermission", "driveItemId", "id", "locator"),
    ("graph", "graph.createPermission", "requireSignIn", "bool", "payload-control"),
    ("graph", "graph.createPermission", "sendInvitation", "bool", "payload-control"),
    # createSharingLink
    ("graph", "graph.createSharingLink", "type", "enum[view,edit]", "principal"),
    ("graph", "graph.createSharingLink", "scope", "enum[anonymous,organization]", "principal"),
    ("graph", "graph.createSharingLink", "driveItemId", "id", "locator"),
    ("graph", "graph.createSharingLink", "expirationDateTime", "datetime", "payload-content"),
    ("graph", "graph.createSharingLink", "retainInheritedPermissions", "bool", "payload-control"),

    # ---- Slack API (25 args) ----
    ("slack", "slack.chat.postMessage", "channel", "channel_id", "recipient"),
    ("slack", "slack.chat.postMessage", "text", "string", "payload-content"),
    ("slack", "slack.chat.postMessage", "blocks", "list[block]", "payload-content"),
    ("slack", "slack.chat.postMessage", "thread_ts", "timestamp", "locator"),
    ("slack", "slack.chat.postMessage", "username", "string", "payload-control"),
    ("slack", "slack.chat.postMessage", "icon_emoji", "string", "payload-control"),
    ("slack", "slack.chat.postMessage", "attachments", "list", "payload-content"),
    ("slack", "slack.files.upload", "channels", "list[channel_id]", "recipient"),
    ("slack", "slack.files.upload", "content", "string", "payload-content"),
    ("slack", "slack.files.upload", "filename", "string", "payload-control"),
    ("slack", "slack.conversations.invite", "channel", "channel_id", "locator"),
    ("slack", "slack.conversations.invite", "users", "list[user_id]", "recipient"),
    ("slack", "slack.admin.users.invite", "email", "email", "recipient"),
    ("slack", "slack.admin.users.invite", "channel_ids", "list[channel_id]", "principal"),
    ("slack", "slack.admin.users.setAdmin", "team_id", "id", "principal"),
    ("slack", "slack.admin.users.setAdmin", "user_id", "user_id", "principal"),
    ("slack", "slack.bookmarks.add", "channel_id", "channel_id", "locator"),
    ("slack", "slack.bookmarks.add", "link", "url", "locator"),
    ("slack", "slack.bookmarks.add", "title", "string", "payload-control"),
    ("slack", "slack.search.messages", "query", "string", "selector"),
    ("slack", "slack.search.messages", "count", "int", "payload-control"),
    ("slack", "slack.search.messages", "sort", "enum", "selector"),
    ("slack", "slack.pins.add", "channel", "channel_id", "locator"),
    ("slack", "slack.pins.add", "timestamp", "timestamp", "locator"),
    ("slack", "slack.reactions.add", "name", "string", "payload-control"),

    # ---- GitHub API (25 args) ----
    ("github", "github.issues.create", "owner", "string", "locator"),
    ("github", "github.issues.create", "repo", "string", "locator"),
    ("github", "github.issues.create", "title", "string", "payload-control"),
    ("github", "github.issues.create", "body", "string", "payload-content"),
    ("github", "github.issues.create", "assignees", "list[username]", "recipient"),
    ("github", "github.issues.create", "labels", "list[string]", "payload-control"),
    ("github", "github.issues.create", "milestone", "int", "locator"),
    ("github", "github.pulls.create", "head", "string", "locator"),
    ("github", "github.pulls.create", "base", "string", "locator"),
    ("github", "github.pulls.create", "title", "string", "payload-control"),
    ("github", "github.pulls.create", "body", "string", "payload-content"),
    ("github", "github.pulls.create", "reviewers", "list[username]", "recipient"),
    ("github", "github.repos.addCollaborator", "username", "string", "principal"),
    ("github", "github.repos.addCollaborator", "permission", "enum", "principal"),
    ("github", "github.repos.addCollaborator", "repo", "string", "locator"),
    ("github", "github.gists.create", "files", "dict", "payload-content"),
    ("github", "github.gists.create", "public", "bool", "payload-control"),
    ("github", "github.gists.create", "description", "string", "payload-control"),
    ("github", "github.search.code", "q", "string", "selector"),
    ("github", "github.search.issues", "q", "string", "selector"),
    ("github", "github.search.repos", "q", "string", "selector"),
    ("github", "github.actions.createWorkflowDispatch", "ref", "string", "locator"),
    ("github", "github.actions.createWorkflowDispatch", "inputs", "dict", "payload-content"),
    ("github", "github.teams.addMember", "team_id", "int", "locator"),
    ("github", "github.teams.addMember", "username", "string", "principal"),

    # ---- Google Workspace (25 args) ----
    ("google", "gmail.users.messages.send", "to", "list[email]", "recipient"),
    ("google", "gmail.users.messages.send", "cc", "list[email]", "recipient"),
    ("google", "gmail.users.messages.send", "bcc", "list[email]", "recipient"),
    ("google", "gmail.users.messages.send", "subject", "string", "payload-control"),
    ("google", "gmail.users.messages.send", "body", "string", "payload-content"),
    ("google", "gmail.users.messages.send", "attachments", "list[file]", "principal"),
    ("google", "calendar.events.insert", "attendees", "list[email]", "recipient"),
    ("google", "calendar.events.insert", "summary", "string", "payload-control"),
    ("google", "calendar.events.insert", "description", "string", "payload-content"),
    ("google", "calendar.events.insert", "location", "string", "locator"),
    ("google", "calendar.events.insert", "start.dateTime", "datetime", "payload-content"),
    ("google", "drive.files.create", "name", "string", "payload-control"),
    ("google", "drive.files.create", "content", "bytes", "payload-content"),
    ("google", "drive.files.create", "parents", "list[folder_id]", "locator"),
    ("google", "drive.permissions.create", "emailAddress", "email", "principal"),
    ("google", "drive.permissions.create", "role", "enum", "principal"),
    ("google", "drive.permissions.create", "fileId", "id", "locator"),
    ("google", "sheets.spreadsheets.values.update", "range", "string", "selector"),
    ("google", "sheets.spreadsheets.values.update", "values", "list", "payload-content"),
    ("google", "sheets.spreadsheets.values.update", "spreadsheetId", "id", "locator"),
    ("google", "admin.directory.users.insert", "primaryEmail", "email", "principal"),
    ("google", "admin.directory.users.insert", "password", "string", "principal"),
    ("google", "admin.directory.members.insert", "email", "email", "principal"),
    ("google", "admin.directory.members.insert", "role", "enum", "principal"),
    ("google", "admin.directory.members.insert", "groupKey", "email", "locator"),

    # ---- Jira (25 args) ----
    ("jira", "jira.issues.create", "project.key", "string", "locator"),
    ("jira", "jira.issues.create", "issuetype.name", "string", "payload-control"),
    ("jira", "jira.issues.create", "summary", "string", "payload-control"),
    ("jira", "jira.issues.create", "description", "string", "payload-content"),
    ("jira", "jira.issues.create", "assignee.accountId", "user_id", "recipient"),
    ("jira", "jira.issues.create", "reporter.accountId", "user_id", "principal"),
    ("jira", "jira.issues.create", "labels", "list[string]", "payload-control"),
    ("jira", "jira.issues.create", "priority.name", "enum", "payload-control"),
    ("jira", "jira.issues.addWatcher", "issueIdOrKey", "string", "locator"),
    ("jira", "jira.issues.addWatcher", "accountId", "user_id", "principal"),
    ("jira", "jira.issues.addComment", "issueIdOrKey", "string", "locator"),
    ("jira", "jira.issues.addComment", "body", "string", "payload-content"),
    ("jira", "jira.issues.addComment", "visibility.type", "enum", "principal"),
    ("jira", "jira.issues.search", "jql", "string", "selector"),
    ("jira", "jira.issues.search", "maxResults", "int", "payload-control"),
    ("jira", "jira.issues.search", "fields", "list[string]", "selector"),
    ("jira", "jira.projects.setPermission", "projectKey", "string", "locator"),
    ("jira", "jira.projects.setPermission", "accountId", "user_id", "principal"),
    ("jira", "jira.projects.setPermission", "permissionSchemeId", "int", "principal"),
    ("jira", "jira.attachments.add", "issueIdOrKey", "string", "locator"),
    ("jira", "jira.attachments.add", "file", "bytes", "payload-content"),
    ("jira", "jira.users.invite", "emailAddress", "email", "recipient"),
    ("jira", "jira.users.invite", "productAccess", "list[string]", "principal"),
    ("jira", "jira.webhooks.register", "url", "url", "locator"),
    ("jira", "jira.webhooks.register", "events", "list[string]", "selector"),

    # ---- Generic OpenAPI (20 args) ----
    ("openapi", "generic.send_notification", "to", "email", "recipient"),
    ("openapi", "generic.send_notification", "message", "string", "payload-content"),
    ("openapi", "generic.send_notification", "subject", "string", "payload-control"),
    ("openapi", "generic.upload_file", "destination", "path", "locator"),
    ("openapi", "generic.upload_file", "content", "bytes", "payload-content"),
    ("openapi", "generic.upload_file", "permissions", "list[user]", "principal"),
    ("openapi", "generic.run_query", "sql", "string", "selector"),
    ("openapi", "generic.run_query", "database_url", "url", "locator"),
    ("openapi", "generic.run_query", "timeout", "int", "payload-control"),
    ("openapi", "generic.grant_access", "user_email", "email", "principal"),
    ("openapi", "generic.grant_access", "resource_id", "id", "locator"),
    ("openapi", "generic.grant_access", "role", "enum", "principal"),
    ("openapi", "generic.create_webhook", "callback_url", "url", "locator"),
    ("openapi", "generic.create_webhook", "events", "list[string]", "selector"),
    ("openapi", "generic.create_webhook", "secret", "string", "principal"),
    ("openapi", "generic.post_message", "channel_id", "id", "recipient"),
    ("openapi", "generic.post_message", "content", "string", "payload-content"),
    ("openapi", "generic.post_message", "reply_to", "id", "locator"),
    ("openapi", "generic.search", "query", "string", "selector"),
    ("openapi", "generic.search", "filters", "dict", "selector"),
]


def infer_role_heuristic(tool: str, arg_name: str, arg_type: str) -> str:
    """Simple heuristic role inference based on argument name and type."""
    name = arg_name.lower().replace(".", "_")
    typ = arg_type.lower()

    # Recipient patterns
    if any(k in name for k in ("to", "recipient", "attendee", "assignee",
                                "cc", "bcc", "email", "user", "channel",
                                "invite", "watcher", "reviewer")):
        if "email" in typ or "user" in typ or "list" in typ:
            return "recipient"

    # Principal patterns
    if any(k in name for k in ("permission", "role", "grant", "admin",
                                "collaborator", "member", "access", "principal",
                                "scope", "auth")):
        return "principal"

    # Locator patterns
    if any(k in name for k in ("path", "url", "id", "key", "repo", "drive",
                                "folder", "location", "ref", "branch", "file_id",
                                "parent", "milestone", "thread", "timestamp",
                                "spreadsheet", "project", "issue", "group",
                                "webhook", "resource", "database", "destination",
                                "callback")):
        if "url" in typ or "path" in typ or "id" in typ or "string" in typ:
            return "locator"

    # Selector patterns
    if any(k in name for k in ("query", "jql", "sql", "q", "search",
                                "filter", "sort", "range", "field", "event")):
        return "selector"

    # Payload-control patterns
    if any(k in name for k in ("subject", "title", "name", "type", "label",
                                "priority", "importance", "content_type",
                                "visibility", "public", "status", "format",
                                "issuetype", "conflict", "retain", "sign",
                                "timeout", "count", "max")):
        return "payload-control"

    # Default: payload-content
    return "payload-content"


def evaluate_role_inference() -> dict:
    """Evaluate heuristic role inference against gold standard."""
    from collections import Counter

    roles = ["recipient", "locator", "selector", "principal",
             "payload-content", "payload-control"]

    # Per-role TP/FP/FN
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # Per-connector
    connector_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for connector, tool, arg_name, arg_type, gold_role in ROLE_GOLD_STANDARD:
        pred_role = infer_role_heuristic(tool, arg_name, arg_type)
        correct = pred_role == gold_role
        connector_stats[connector]["total"] += 1
        if correct:
            connector_stats[connector]["correct"] += 1
            tp[gold_role] += 1
        else:
            fp[pred_role] += 1
            fn[gold_role] += 1

    results = {}
    for role in roles:
        p = tp[role] / (tp[role] + fp[role]) if (tp[role] + fp[role]) > 0 else 0
        r = tp[role] / (tp[role] + fn[role]) if (tp[role] + fn[role]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[role] = {
            "precision": round(p * 100, 1),
            "recall": round(r * 100, 1),
            "f1": round(f1 * 100, 1),
            "tp": tp[role], "fp": fp[role], "fn": fn[role],
        }

    # Oracle vs inferred ASR impact (simulated)
    # With oracle roles: ASR matches main results (24% on Bench-v2)
    # With inferred roles: some authority-bearing args misclassified as payload -> higher ASR
    n_misclassified_critical = sum(
        1 for _, _, arg_name, arg_type, gold_role in ROLE_GOLD_STANDARD
        if gold_role in ("recipient", "principal", "locator")
        and infer_role_heuristic("", arg_name, arg_type)
           not in ("recipient", "principal", "locator")
    )
    n_critical = sum(
        1 for _, _, _, _, gold_role in ROLE_GOLD_STANDARD
        if gold_role in ("recipient", "principal", "locator")
    )

    return {
        "per_role": results,
        "connector_stats": {k: v for k, v in connector_stats.items()},
        "n_total": len(ROLE_GOLD_STANDARD),
        "n_critical_misclassified": n_misclassified_critical,
        "n_critical": n_critical,
        "critical_miss_rate": round(n_misclassified_critical / n_critical * 100, 1)
                              if n_critical > 0 else 0,
    }


def main():
    results = evaluate_role_inference()

    print(f"\n{'='*70}")
    print(f"ROLE INFERENCE EVALUATION (N={results['n_total']} args, 6 connectors)")
    print(f"{'='*70}")
    print(f"\n{'Role':<20} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 50)

    rows = []
    for role, m in results["per_role"].items():
        print(f"{role:<20} {m['precision']:>9.1f}% {m['recall']:>7.1f}% {m['f1']:>7.1f}%")
        rows.append({"role": role, **m})

    macro_f1 = sum(r["f1"] for r in rows) / len(rows)
    print(f"\n{'Macro avg':<20} {'':>10} {'':>8} {macro_f1:>7.1f}%")

    print(f"\nConnector accuracy:")
    for conn, stats in results["connector_stats"].items():
        acc = stats["correct"] / stats["total"] * 100
        print(f"  {conn:<12} {stats['correct']}/{stats['total']}={acc:.0f}%")

    print(f"\nCritical role misclassification:")
    print(f"  {results['n_critical_misclassified']}/{results['n_critical']} "
          f"authority-bearing args misclassified = "
          f"{results['critical_miss_rate']}% miss rate")
    print(f"  Impact: misclassified authority args treated as payload "
          f"-> potential ASR increase")

    # Save
    with open(RESULTS_DIR / "role_inference.csv", "w", newline="") as f:
        writer = csv.DictWriter(f,
            fieldnames=["role", "precision", "recall", "f1", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)

    with open(RESULTS_DIR / "role_inference_full.json", "w") as f:
        json.dump(results, f, indent=2)

    from intentflow.utils.logger import logger
    logger.info(f"Results saved to {RESULTS_DIR}/")
    return results


if __name__ == "__main__":
    main()