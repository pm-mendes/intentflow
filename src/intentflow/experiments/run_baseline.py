"""
Script d'exécution d'une baseline.
Usage : python -m intentflow.experiments.run_baseline --baseline B6
"""

import argparse
import json
from pathlib import Path

from intentflow.baselines.runner import BaselineRunner
from intentflow.data.generator import DataGenerator
from intentflow.models.models import AttackedTask, BenignTask, RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.utils.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IntentFlow baseline experiment")
    parser.add_argument("--baseline", required=True, choices=ToolMediator.BASELINES)
    args = parser.parse_args()

    # Charger les données (générer si nécessaire)
    data_dir = Path("data")
    benign_file = data_dir / "tasks_benign.json"
    attacked_file = data_dir / "tasks_attacked.json"
    docs_file = data_dir / "documents.json"

    if not benign_file.exists():
        DataGenerator(seed=settings.random_seed).generate_all(str(data_dir))

    benign_tasks = [BenignTask(**t) for t in json.loads(benign_file.read_text())]
    attacked_tasks = [AttackedTask(**t) for t in json.loads(attacked_file.read_text())]
    documents = [RAGDocument(**d) for d in json.loads(docs_file.read_text())]

    # Initialiser le pipeline
    policy_engine = PolicyEngine.load_default_policies()
    mediator = ToolMediator(policy_engine=policy_engine, documents=documents)
    runner = BaselineRunner(mediator=mediator)

    # Exécuter
    runner.run(
        baseline=args.baseline,
        benign_tasks=benign_tasks,
        attacked_tasks=attacked_tasks,
        documents=documents,
    )


if __name__ == "__main__":
    main()
