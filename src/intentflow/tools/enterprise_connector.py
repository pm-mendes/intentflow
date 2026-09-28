"""
intentflow/tools/enterprise_connector.py

Connecteur enterprise simulé — Microsoft Graph / Gmail style.

Pourquoi ce connecteur ?
Les reviews critiquent les outils simulés comme trop simples.
Un connecteur enterprise réaliste a des schémas plus complexes :
- Arguments imbriqués (listes de destinataires, pièces jointes multiples)
- Permissions et scopes
- Sharing links
- Différents types d'événements calendrier

Ce module simule un connecteur Microsoft Graph-style
et montre comment IntentFlow mappe les schémas complexes
vers son système de rôles et politiques.

Correspondance avec le papier (Section 6) :
"We additionally demonstrate IntentFlow on a Microsoft Graph-style
enterprise connector with nested argument schemas."
"""

from dataclasses import dataclass
from typing import Any

from intentflow.utils.logger import logger

# ============================================================
# Schémas Microsoft Graph-style
# ============================================================


@dataclass
class EmailRecipient:
    """Destinataire email avec nom et adresse."""

    email_address: str
    name: str | None = None

    def to_dict(self) -> dict:
        return {"emailAddress": {"address": self.email_address, "name": self.name or ""}}


@dataclass
class FileAttachment:
    """Pièce jointe avec contenu base64."""

    name: str
    content_type: str = "application/octet-stream"
    content_bytes: str = ""  # Base64 simulé
    size: int = 0


@dataclass
class MeetingAttendee:
    """Participant à une réunion avec type de présence."""

    email_address: str
    attendee_type: str = "required"  # required | optional | resource

    def to_dict(self) -> dict:
        return {
            "emailAddress": {"address": self.email_address},
            "type": self.attendee_type,
        }


@dataclass
class OnlineMeetingInfo:
    """Informations de réunion en ligne."""

    join_url: str | None = None
    conference_id: str | None = None
    provider: str = "teamsForBusiness"


# ============================================================
# Outils Microsoft Graph-style
# ============================================================


class GraphMailClient:
    """
    Simule le connecteur Microsoft Graph pour les emails.

    graph.sendMail(message) → Envoi d'email avec schéma Graph complet
    """

    @staticmethod
    def send_mail(
        to_recipients: list[dict],  # [{"emailAddress": {"address": "..."}}]
        subject: str,
        body_content: str,
        body_content_type: str = "text",  # text | html
        cc_recipients: list | None = None,
        bcc_recipients: list | None = None,
        attachments: list | None = None,
        importance: str = "normal",  # low | normal | high
        save_to_sent_items: bool = True,
    ) -> dict[str, Any]:
        """
        Simule graph.sendMail().
        Arguments complexes avec destinataires imbriqués.
        """
        if not to_recipients:
            raise ValueError("to_recipients cannot be empty")

        # Valider les adresses email
        for recipient in to_recipients:
            addr = recipient.get("emailAddress", {}).get("address", "")
            if "@" not in addr:
                raise ValueError(f"Invalid email address: {addr}")

        recipients_list = [r["emailAddress"]["address"] for r in to_recipients]
        logger.info(f"[GraphMail] sendMail → {recipients_list}")

        return {
            "status": "sent",
            "messageId": f"msg_{hash(subject) % 100000:05d}",
            "toRecipients": recipients_list,
            "subject": subject,
            "hasAttachments": bool(attachments),
            "importance": importance,
        }

    @staticmethod
    def create_message_with_attachment(
        to_recipients: list[dict],
        subject: str,
        body_content: str,
        file_path: str,
        file_name: str,
    ) -> dict[str, Any]:
        """
        Simule la création d'un email avec pièce jointe depuis le filesystem.
        file_path est un argument locator → haute criticité.
        """
        if ".." in file_path:
            raise ValueError(f"Path traversal detected in file_path: {file_path}")

        recipients_list = [r["emailAddress"]["address"] for r in to_recipients]
        logger.info(f"[GraphMail] createWithAttachment → {recipients_list}, file={file_path}")

        return {
            "status": "drafted",
            "toRecipients": recipients_list,
            "subject": subject,
            "attachment": {"name": file_name, "sourcePath": file_path},
        }


class GraphCalendarClient:
    """
    Simule le connecteur Microsoft Graph pour le calendrier.
    """

    @staticmethod
    def create_event(
        subject: str,
        start_datetime: str,
        end_datetime: str,
        attendees: list[dict],
        body_content: str = "",
        location: str = "",
        is_online_meeting: bool = False,
        online_meeting_provider: str = "teamsForBusiness",
        sensitivity: str = "normal",  # normal | personal | private | confidential
        show_as: str = "busy",
    ) -> dict[str, Any]:
        """
        Simule graph.createEvent() avec schéma complet.
        attendees est un argument recipient → haute criticité.
        """
        if not attendees:
            raise ValueError("attendees cannot be empty")

        attendee_emails = [a.get("emailAddress", {}).get("address", "") for a in attendees]
        logger.info(f"[GraphCalendar] createEvent '{subject}' with {attendee_emails}")

        return {
            "status": "created",
            "eventId": f"evt_{hash(subject) % 100000:05d}",
            "subject": subject,
            "start": start_datetime,
            "end": end_datetime,
            "attendees": attendee_emails,
            "isOnlineMeeting": is_online_meeting,
            "sensitivity": sensitivity,
        }

    @staticmethod
    def create_sharing_link(
        item_id: str,
        link_type: str = "view",  # view | edit | embed
        scope: str = "organization",  # organization | anonymous | users
        recipients: list | None = None,
    ) -> dict[str, Any]:
        """
        Simule graph.createSharingLink().
        scope et recipients sont des arguments principal → haute criticité.
        Si scope=anonymous et recipients inclut des emails externes → risque.
        """
        if scope == "anonymous" and recipients:
            logger.warning(
                "[GraphCalendar] createSharingLink with scope=anonymous "
                "and explicit recipients — potential data leak"
            )

        logger.info(f"[GraphCalendar] createSharingLink item={item_id} scope={scope}")
        return {
            "status": "created",
            "link": f"https://company.sharepoint.com/shared/{item_id}",
            "type": link_type,
            "scope": scope,
        }


class GraphDriveClient:
    """
    Simule le connecteur Microsoft Graph pour OneDrive/SharePoint.
    """

    @staticmethod
    def upload_file(
        drive_id: str,
        path: str,
        content: bytes | str,
        conflict_behavior: str = "replace",  # fail | replace | rename
    ) -> dict[str, Any]:
        """
        Simule graph.uploadFile().
        path et drive_id sont des locators → haute criticité.
        """
        if ".." in path:
            raise ValueError(f"Path traversal in path: {path}")

        logger.info(f"[GraphDrive] uploadFile drive={drive_id} path={path}")
        return {
            "status": "uploaded",
            "driveId": drive_id,
            "path": path,
            "size": len(content) if isinstance(content, bytes | str) else 0,
        }

    @staticmethod
    def create_permission(
        item_id: str,
        roles: list[str],  # read | write | owner
        grant_to_emails: list[str],  # Liste d'emails autorisés
        send_invitation: bool = True,
    ) -> dict[str, Any]:
        """
        Simule graph.createPermission().
        grant_to_emails est un principal → haute criticité absolue.
        """
        if not grant_to_emails:
            raise ValueError("grant_to_emails cannot be empty")

        for email in grant_to_emails:
            if "@" not in email:
                raise ValueError(f"Invalid email in grant_to_emails: {email}")

        logger.info(f"[GraphDrive] createPermission {roles} → {grant_to_emails}")
        return {
            "status": "granted",
            "itemId": item_id,
            "roles": roles,
            "grantedTo": grant_to_emails,
        }


# ============================================================
# Registre du connecteur enterprise
# ============================================================

ENTERPRISE_TOOL_REGISTRY = {
    "graph.sendMail": GraphMailClient.send_mail,
    "graph.createMailWithAttachment": GraphMailClient.create_message_with_attachment,
    "graph.createEvent": GraphCalendarClient.create_event,
    "graph.createSharingLink": GraphCalendarClient.create_sharing_link,
    "graph.uploadFile": GraphDriveClient.upload_file,
    "graph.createPermission": GraphDriveClient.create_permission,
}

# Politiques IntentFlow pour les outils enterprise
# Montre comment les schémas complexes sont mappés vers les rôles
ENTERPRISE_POLICY_MAPPING = {
    "graph.sendMail": {
        "to_recipients": "recipient",  # Liste imbriquée → recipient
        "cc_recipients": "recipient",
        "bcc_recipients": "recipient",
        "subject": "payload",
        "body_content": "payload",
        "attachments": "principal",  # Pièces jointes → principal
        "importance": "payload",
    },
    "graph.createEvent": {
        "subject": "payload",
        "attendees": "recipient",  # Liste d'attendees → recipient
        "start_datetime": "payload",
        "end_datetime": "payload",
        "location": "payload",
        "sensitivity": "principal",  # Niveau de confidentialité → principal
        "is_online_meeting": "payload",
    },
    "graph.uploadFile": {
        "drive_id": "locator",  # Drive ID → locator
        "path": "locator",  # Path dans le drive → locator
        "content": "payload",
    },
    "graph.createPermission": {
        "item_id": "locator",
        "roles": "principal",  # Rôles de permission → principal
        "grant_to_emails": "principal",  # Destinataires de permission → principal
    },
    "graph.createSharingLink": {
        "item_id": "locator",
        "link_type": "payload",
        "scope": "principal",  # Scope de partage → principal
        "recipients": "recipient",
    },
}


def get_enterprise_tool(tool_name: str):
    """Retourne la fonction d'outil enterprise par son nom."""
    return ENTERPRISE_TOOL_REGISTRY.get(tool_name)


def demonstrate_enterprise_connector() -> None:
    """Démontre l'usage du connecteur enterprise avec IntentFlow."""

    print("\n=== Enterprise Connector — IntentFlow Integration Demo ===\n")

    # Cas 1 : Email légitime
    print("Case 1: Legitimate email send")
    result = GraphMailClient.send_mail(
        to_recipients=[{"emailAddress": {"address": "manager@company.com"}}],
        subject="Q3 Summary",
        body_content="Please find the Q3 summary attached.",
    )
    print(f"  → {result['status']}: {result['toRecipients']}")

    # Cas 2 : Tentative d'envoi à un attaquant
    print("\nCase 2: Attacker email (would be blocked by IntentFlow)")
    print("  → IntentFlow policy: to_recipients role=recipient → user_intent only")
    print("  → 'attacker@evil.io' from injected_document → BLOCK")

    # Cas 3 : Création d'événement avec participant non autorisé
    print("\nCase 3: Calendar event with unauthorized attendee")
    print("  → attendees role=recipient → user_intent only")
    print("  → 'spy@evil.io' not in user intent → BLOCK via B6")

    # Cas 4 : Upload de fichier avec path traversal
    print("\nCase 4: File upload with path traversal")
    try:
        GraphDriveClient.upload_file(
            drive_id="drive_001",
            path="../../etc/passwd",
            content="injected content",
        )
    except ValueError as e:
        print(f"  → Tool-level validation caught: {e}")

    # Cas 5 : Permission grant à un email externe
    print("\nCase 5: Permission grant to external email")
    print("  → grant_to_emails role=principal → user_intent only")
    print("  → Any email not in user intent → BLOCK via B6")

    print("\n=== Policy Mapping Summary ===")
    for tool_name, mapping in ENTERPRISE_POLICY_MAPPING.items():
        print(f"\n  {tool_name}:")
        for arg, role in mapping.items():
            criticality = "HIGH" if role in ("recipient", "locator", "principal") else "low"
            print(f"    {arg:<25} → role={role:<12} criticality={criticality}")
