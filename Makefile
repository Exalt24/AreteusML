.PHONY: install dev lint test train serve demo clean docker-up docker-down

install:
	uv sync

dev:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -v --tb=short

test-cov:
	uv run pytest -v --cov=backend --cov=ml --cov-report=html --tb=short

train-baseline:
	uv run python -m ml.training.train_baseline

train:
	uv run python -m ml.training.train

train-fewshot:
	uv run python -m ml.training.train_fewshot

evaluate:
	uv run python -m ml.training.evaluate

export-onnx:
	uv run python -m ml.training.export_onnx

serve:
	uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

demo:
	uv run streamlit run dashboard/app.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage coverage.xml
