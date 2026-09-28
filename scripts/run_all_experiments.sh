#!/usr/bin/env bash
# ============================================================
# run_all_experiments.sh
# Reproduit l'intégralité des expériences IntentFlow pour ICAART 2027
#
# Usage : bash scripts/run_all_experiments.sh
# Prérequis : poetry install, .env configuré avec OPENAI_API_KEY
# ============================================================

set -euo pipefail  # Arrêt immédiat en cas d'erreur

echo "=== IntentFlow ICAART 2027 — Expériences ==="
echo "Démarrage : $(date)"

# 1. Vérification de l'environnement
echo ""
echo "--- Vérification de l'environnement ---"
poetry run python -c "import intentflow; print('✓ Package intentflow chargé')"
poetry run python -c "import openai; print('✓ openai disponible')"

# 2. Génération des données (si non existantes)
echo ""
echo "--- Génération des données ---"
poetry run python -m intentflow.data.generator

# 3. Exécution des baselines B0–B6
echo ""
echo "--- Exécution des baselines ---"
for baseline in B0 B1 B2 B3 B4 B5 B6; do
    echo "  Running $baseline..."
    poetry run python -m intentflow.experiments.run_baseline --baseline "$baseline"
done

# 4. Calcul des métriques
echo ""
echo "--- Calcul des métriques ---"
poetry run python -m intentflow.experiments.compute_metrics

echo ""
echo "=== Terminé : $(date) ==="
echo "Résultats disponibles dans : data/results/metrics.csv"
