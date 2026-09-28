# IntentFlow ICAART 2027 — reproduction artifact image.
#
# Build:  docker build -t intentflow-artifact .
# Run:    docker run --rm -v "$(pwd)/data:/app/data" intentflow-artifact --quick
#         docker run --rm -e OPENAI_API_KEY=sk-... -v "$(pwd)/data:/app/data" intentflow-artifact --with-llm
#         docker run --rm -v "$(pwd)/data:/app/data" intentflow-artifact --offline --pseudonymize-logs
#
# No OPENAI_API_KEY is required for --quick or --offline (see artifact/scripts/run_all.sh).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-ansi

COPY . .
RUN poetry install --no-ansi

ENTRYPOINT ["bash", "artifact/scripts/run_all.sh"]
CMD ["--quick"]
