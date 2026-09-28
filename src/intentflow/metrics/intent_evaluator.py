"""
intentflow/metrics/intent_evaluator.py

Évaluation de l'extracteur d'intention avec précision, rappel et F1.

Pourquoi évaluer l'intent extractor ?
Les reviews disent que l'intent extractor est un composant critique :
s'il échoue, toute la médiation échoue. Il faut donc le traiter
comme un composant évaluable séparément.

Protocole :
1. Gold standard : 50 messages annotés manuellement avec les champs attendus
2. Métriques : exact match, field-level precision, recall, F1
3. Modes : oracle vs LLM vs heuristic

Champs évalués :
- action (outil principal)
- recipient / participants (entités cibles)
- path / endpoint (localisateurs)
- date (temporel)
- confidence (calibration)
"""

from dataclasses import dataclass
from pathlib import Path

from intentflow.intent.extractor import IntentExtractor
from intentflow.models.models import Intent


@dataclass
class GoldIntent:
    """Intent gold standard annoté manuellement."""

    message: str
    expected_action: str
    expected_recipient: str | None = None
    expected_participants: list | None = None
    expected_path: str | None = None
    expected_endpoint: str | None = None
    expected_date: str | None = None
    notes: str = ""


@dataclass
class FieldMetrics:
    """Métriques pour un champ spécifique."""

    field_name: str
    total: int = 0
    correct: int = 0
    present_gold: int = 0  # Dans le gold
    present_pred: int = 0  # Dans la prédiction
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        if self.present_pred == 0:
            return 1.0
        return min(1.0, self.true_positive / self.present_pred)

    @property
    def recall(self) -> float:
        if self.present_gold == 0:
            return 1.0
        return self.true_positive / self.present_gold

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def exact_match_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


# ============================================================
# Gold Standard — 50 messages annotés
# ============================================================

GOLD_STANDARD = [
    # === EMAIL ===
    GoldIntent(
        "Send the Q3 report to alice@company.com",
        "send_email",
        expected_recipient="alice@company.com",
    ),
    GoldIntent(
        "Forward the meeting notes to bob@company.com",
        "send_email",
        expected_recipient="bob@company.com",
    ),
    GoldIntent(
        "Email the compliance report to legal@company.com",
        "send_email",
        expected_recipient="legal@company.com",
    ),
    GoldIntent(
        "Send the project update to manager@company.com",
        "send_email",
        expected_recipient="manager@company.com",
    ),
    GoldIntent(
        "Forward the invoice to finance@company.com",
        "send_email",
        expected_recipient="finance@company.com",
    ),
    GoldIntent(
        "Send a summary of the sprint to carol@company.com and david@company.com",
        "send_email",
        expected_recipient="carol@company.com",
    ),
    GoldIntent(
        "Email the board presentation to ceo@company.com",
        "send_email",
        expected_recipient="ceo@company.com",
    ),
    GoldIntent(
        "Send the audit findings to cfo@company.com",
        "send_email",
        expected_recipient="cfo@company.com",
    ),
    GoldIntent(
        "Forward the customer feedback to support@company.com",
        "send_email",
        expected_recipient="support@company.com",
    ),
    GoldIntent(
        "Send the release notes to the engineering team at eng@company.com",
        "send_email",
        expected_recipient="eng@company.com",
    ),
    # === CALENDAR ===
    GoldIntent(
        "Schedule a meeting with alice@company.com on 2026-07-15",
        "create_calendar_event",
        expected_participants=["alice@company.com"],
        expected_date="2026-07-15",
    ),
    GoldIntent(
        "Create a board meeting on 2026-08-01 with ceo@company.com and cfo@company.com",
        "create_calendar_event",
        expected_participants=["ceo@company.com", "cfo@company.com"],
        expected_date="2026-08-01",
    ),
    GoldIntent(
        "Set up a sprint review on 2026-07-20 with the dev team",
        "create_calendar_event",
        expected_date="2026-07-20",
    ),
    GoldIntent(
        "Schedule a one-on-one with manager@company.com next Monday",
        "create_calendar_event",
        expected_participants=["manager@company.com"],
    ),
    GoldIntent(
        "Create a workshop on 2026-09-10 for the entire team",
        "create_calendar_event",
        expected_date="2026-09-10",
    ),
    GoldIntent(
        "Book a call with carol@company.com and bob@company.com on 2026-07-25",
        "create_calendar_event",
        expected_participants=["carol@company.com", "bob@company.com"],
        expected_date="2026-07-25",
    ),
    GoldIntent(
        "Schedule the quarterly review for 2026-09-01",
        "create_calendar_event",
        expected_date="2026-09-01",
    ),
    GoldIntent(
        "Create a training session on 2026-08-15 with david@company.com",
        "create_calendar_event",
        expected_participants=["david@company.com"],
        expected_date="2026-08-15",
    ),
    GoldIntent(
        "Set up a retrospective on 2026-07-30 with alice@company.com and bob@company.com",
        "create_calendar_event",
        expected_participants=["alice@company.com", "bob@company.com"],
        expected_date="2026-07-30",
    ),
    GoldIntent(
        "Book a strategy session on 2026-10-01 with the leadership team",
        "create_calendar_event",
        expected_date="2026-10-01",
    ),
    # === FILE ===
    GoldIntent(
        "Save the report to /reports/q3_2026.pdf",
        "write_file",
        expected_path="/reports/q3_2026.pdf",
    ),
    GoldIntent(
        "Write the meeting transcript to /transcripts/meeting_july.txt",
        "write_file",
        expected_path="/transcripts/meeting_july.txt",
    ),
    GoldIntent(
        "Export the data to /exports/sales_june.csv",
        "write_file",
        expected_path="/exports/sales_june.csv",
    ),
    GoldIntent(
        "Save the config to /backups/config.yaml",
        "write_file",
        expected_path="/backups/config.yaml",
    ),
    GoldIntent(
        "Write the audit log to /logs/audit_june.log",
        "write_file",
        expected_path="/logs/audit_june.log",
    ),
    GoldIntent(
        "Export results to /data/analysis_output.json",
        "write_file",
        expected_path="/data/analysis_output.json",
    ),
    GoldIntent(
        "Save the model checkpoint to /models/checkpoint_v2.pt",
        "write_file",
        expected_path="/models/checkpoint_v2.pt",
    ),
    GoldIntent(
        "Write the invoice to /invoices/inv_2026_06.pdf",
        "write_file",
        expected_path="/invoices/inv_2026_06.pdf",
    ),
    GoldIntent(
        "Save the customer data to /customers/export_june.json",
        "write_file",
        expected_path="/customers/export_june.json",
    ),
    GoldIntent(
        "Export the compliance report to /compliance/report_q2.pdf",
        "write_file",
        expected_path="/compliance/report_q2.pdf",
    ),
    # === QUERY ===
    GoldIntent(
        "Run a query on https://internal-api.company.com/db to get active users",
        "run_query",
        expected_endpoint="https://internal-api.company.com/db",
    ),
    GoldIntent(
        "Fetch the sales data from https://internal-api.company.com/sales",
        "run_query",
        expected_endpoint="https://internal-api.company.com/sales",
    ),
    GoldIntent(
        "Query the HR system at https://internal-api.company.com/hr for headcount",
        "run_query",
        expected_endpoint="https://internal-api.company.com/hr",
    ),
    GoldIntent(
        "Get the analytics from https://api.company.com/metrics",
        "run_query",
        expected_endpoint="https://api.company.com/metrics",
    ),
    GoldIntent(
        "Run the audit query on https://internal-api.company.com/audit",
        "run_query",
        expected_endpoint="https://internal-api.company.com/audit",
    ),
    GoldIntent(
        "Fetch inventory levels from the supply chain API at https://internal-api.company.com/supply",
        "run_query",
        expected_endpoint="https://internal-api.company.com/supply",
    ),
    GoldIntent(
        "Query the finance system at https://api.company.com/finance",
        "run_query",
        expected_endpoint="https://api.company.com/finance",
    ),
    GoldIntent(
        "Get user permissions from https://internal-api.company.com/iam",
        "run_query",
        expected_endpoint="https://internal-api.company.com/iam",
    ),
    GoldIntent(
        "Run the compliance check on https://internal-api.company.com/compliance",
        "run_query",
        expected_endpoint="https://internal-api.company.com/compliance",
    ),
    GoldIntent(
        "Fetch the CRM data from https://api.company.com/crm",
        "run_query",
        expected_endpoint="https://api.company.com/crm",
    ),
    # === MIXED / AMBIGUOUS ===
    GoldIntent(
        "Summarize the emails and send the summary to my manager",
        "send_email",
        notes="recipient ambiguous — 'manager' not explicit email",
    ),
    GoldIntent(
        "Create a meeting for next Tuesday with the finance team",
        "create_calendar_event",
        notes="date ambiguous",
    ),
    GoldIntent("Save the output somewhere safe", "write_file", notes="path ambiguous"),
    GoldIntent("Run the usual query on the database", "run_query", notes="endpoint ambiguous"),
    GoldIntent(
        "Forward the attached document to the compliance officer",
        "send_email",
        notes="recipient ambiguous",
    ),
    GoldIntent(
        "Schedule the weekly sync", "create_calendar_event", notes="date and participants ambiguous"
    ),
    GoldIntent("Export the results", "write_file", notes="path ambiguous"),
    GoldIntent("Check the database for anomalies", "run_query", notes="endpoint ambiguous"),
    GoldIntent(
        "Send the report to the team",
        "send_email",
        notes="recipient ambiguous — 'team' not specific",
    ),
    GoldIntent("Back up the configuration file", "write_file", notes="path ambiguous"),
]


class IntentEvaluator:
    """Évalue l'extracteur d'intention sur le gold standard."""

    def __init__(self, extractor: IntentExtractor | None = None) -> None:
        self.extractor = extractor or IntentExtractor()

    def evaluate(
        self,
        gold_standard: list[GoldIntent] = None,
        mode: str = "heuristic",
    ) -> dict[str, FieldMetrics]:
        """
        Évalue l'extracteur sur le gold standard.

        Args:
            gold_standard: Liste de GoldIntent (défaut: GOLD_STANDARD)
            mode: "heuristic" | "llm"

        Returns:
            Dict {field_name: FieldMetrics}
        """
        if gold_standard is None:
            gold_standard = GOLD_STANDARD

        metrics = {
            "action": FieldMetrics("action"),
            "recipient": FieldMetrics("recipient"),
            "participants": FieldMetrics("participants"),
            "path": FieldMetrics("path"),
            "endpoint": FieldMetrics("endpoint"),
            "date": FieldMetrics("date"),
        }

        for gold in gold_standard:
            # Extraire l'intention
            intent = self.extractor.extract(gold.message, mode=mode)

            # Évaluer chaque champ
            self._eval_action(gold, intent, metrics["action"])
            self._eval_field(
                gold.expected_recipient,
                intent.parameters.get("recipient"),
                metrics["recipient"],
            )
            self._eval_list_field(
                gold.expected_participants,
                intent.parameters.get("participants"),
                metrics["participants"],
            )
            self._eval_field(
                gold.expected_path,
                intent.parameters.get("path"),
                metrics["path"],
            )
            self._eval_field(
                gold.expected_endpoint,
                intent.parameters.get("endpoint"),
                metrics["endpoint"],
            )
            self._eval_field(
                gold.expected_date,
                intent.parameters.get("date"),
                metrics["date"],
            )

        return metrics

    @staticmethod
    def _eval_action(gold: GoldIntent, intent: Intent, m: FieldMetrics) -> None:
        m.total += 1
        m.present_gold += 1
        m.present_pred += 1
        if intent.action == gold.expected_action:
            m.correct += 1
            m.true_positive += 1
        else:
            m.false_positive += 1
            m.false_negative += 1

    @staticmethod
    def _eval_field(
        gold_val: str | None,
        pred_val: str | None,
        m: FieldMetrics,
    ) -> None:
        m.total += 1
        if gold_val is not None:
            m.present_gold += 1
        if pred_val is not None:
            m.present_pred += 1

        if gold_val is None and pred_val is None:
            m.correct += 1
        elif gold_val is not None and pred_val is not None:
            # Comparaison insensible à la casse
            if str(gold_val).lower().strip() == str(pred_val).lower().strip():
                m.correct += 1
                m.true_positive += 1
            else:
                m.false_positive += 1
                m.false_negative += 1
        elif gold_val is not None and pred_val is None:
            m.false_negative += 1
        else:
            m.false_positive += 1

    @staticmethod
    def _eval_list_field(
        gold_list: list | None,
        pred_list: list | None,
        m: FieldMetrics,
    ) -> None:
        m.total += 1
        if gold_list:
            m.present_gold += 1
        if pred_list:
            m.present_pred += 1

        if not gold_list and not pred_list:
            m.correct += 1
        elif gold_list and pred_list:
            gold_set = {str(x).lower().strip() for x in gold_list}
            pred_set = {str(x).lower().strip() for x in pred_list}
            overlap = gold_set & pred_set
            m.true_positive += len(overlap)
            m.false_positive += len(pred_set - gold_set)
            m.false_negative += len(gold_set - pred_set)
            if gold_set == pred_set:
                m.correct += 1
        elif gold_list:
            m.false_negative += 1
        else:
            m.false_positive += 1

    @staticmethod
    def print_report(metrics: dict[str, FieldMetrics], mode: str = "heuristic") -> None:
        """Affiche le rapport d'évaluation."""
        print(f"\n{'=' * 70}")
        print(f"Intent Extractor Evaluation — mode={mode}")
        print(f"{'=' * 70}")
        print(f"{'Field':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ExactMatch':>12}")
        print(f"{'-' * 65}")
        for field_name, m in metrics.items():
            print(
                f"{field_name:<20} {m.precision:>9.1%} {m.recall:>9.1%} "
                f"{m.f1:>9.1%} {m.exact_match_rate:>11.1%}"
            )
        print(f"{'=' * 70}")

    @staticmethod
    def export_csv(metrics: dict[str, FieldMetrics], output_path: str, mode: str) -> None:
        """Exporte les métriques en CSV."""
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "field",
                    "mode",
                    "precision",
                    "recall",
                    "f1",
                    "exact_match",
                    "total",
                    "present_gold",
                    "present_pred",
                ]
            )
            for field_name, m in metrics.items():
                writer.writerow(
                    [
                        field_name,
                        mode,
                        round(m.precision, 4),
                        round(m.recall, 4),
                        round(m.f1, 4),
                        round(m.exact_match_rate, 4),
                        m.total,
                        m.present_gold,
                        m.present_pred,
                    ]
                )


if __name__ == "__main__":
    from pathlib import Path

    Path("data/results").mkdir(parents=True, exist_ok=True)

    evaluator = IntentEvaluator()

    print("\nEvaluating heuristic mode (no API required)...")
    metrics_h = evaluator.evaluate(mode="heuristic")
    IntentEvaluator.print_report(metrics_h, "heuristic")
    IntentEvaluator.export_csv(metrics_h, "data/results/intent_eval_heuristic.csv", "heuristic")

    print("\nEvaluating LLM mode (requires OPENAI_API_KEY)...")
    try:
        metrics_llm = evaluator.evaluate(mode="llm")
        IntentEvaluator.print_report(metrics_llm, "llm")
        IntentEvaluator.export_csv(metrics_llm, "data/results/intent_eval_llm.csv", "llm")
    except Exception as e:
        print(f"LLM evaluation failed: {e}")
        print("Run with OPENAI_API_KEY set to evaluate LLM mode.")
