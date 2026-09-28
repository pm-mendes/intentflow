"""
Script dédié à l'exécution de la baseline B6 (IntentFlow complet).

Alias de run_baseline.py --baseline B6, fourni séparément
pour respecter la structure demandée dans le protocole expérimental
et faciliter la reproduction indépendante d'IntentFlow seul.

Usage : python -m intentflow.experiments.run_intentflow
"""

import subprocess
import sys


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "intentflow.experiments.run_baseline", "--baseline", "B6"],
        check=True,
    )


if __name__ == "__main__":
    main()
