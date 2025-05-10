FROM python:3.12

WORKDIR /code

COPY pyproject.toml uv.lock alembic.ini start.sh start-worker.sh ruff.toml ./

RUN pip install --root-user-action ignore --upgrade pip uv &&  \
    uv sync --no-dev --compile-bytecode && \
    chmod +x start.sh &&  \
    chmod +x start-worker.sh


COPY src ./src
COPY data ./data
COPY alembic ./alembic
