# Contributing to IntentFlow

Thank you for your interest! This is a research prototype published under MIT.

## How to contribute

1. Fork the repository (ou demander un accès Developer sur GitLab)
2. Create a branch: `git checkout -b feature/your-feature`
3. Make your changes with tests
4. Run: `poetry run pytest tests/unit/ -v`
5. Run: `poetry run ruff check src/`
6. Submit a **Merge Request** to `develop`

## Code standards

- Python 3.11+, typed with mypy
- All public functions have docstrings
- Tests required for new components

## Reporting issues

Open a **GitLab Issue** with a minimal reproducible example.
