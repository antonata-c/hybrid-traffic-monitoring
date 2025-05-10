#!/bin/bash
uv run alembic upgrade head
cd src
uv run uvicorn main:app --reload --host 0.0.0.0
