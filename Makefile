.PHONY: help setup vendor serve dev test lint eval eval-real baselines sample feed clean docker

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:           ## Install runtime + dev dependencies
	pip install -r requirements-dev.txt

vendor:          ## Fetch the dashboard's offline front-end assets (one time)
	python tools/fetch_vendor.py

serve:           ## Run the API + dashboard on :8000
	uvicorn server.main:app --host 127.0.0.1 --port 8000

dev:             ## Run with autoreload
	uvicorn server.main:app --reload --port 8000

test:            ## Run the test suite
	pytest -q

lint:            ## Lint
	ruff check netforecast server tools tests

sample:          ## Regenerate the bundled sample capture
	python -m netforecast.simulate_traffic --out data/sample_flows.csv \
		--n-benign 2600 --n-campaigns 5 --duration-hours 6 --seed 7

baselines:       ## Refit the LR + RF baselines (never touches world_model.pt)
	python -m netforecast.baseline --data data/synthetic_flows.csv --model-dir models --window-seconds 30

eval:            ## Full benchmark on the synthetic model
	python -m netforecast.evaluate --model-dir models --data data/synthetic_flows.csv \
		--report reports/benchmark.md --window-unit seconds

eval-real:       ## Full benchmark on the CIC-IDS2017 model
	python -m netforecast.evaluate --model-dir models_real --data data/cicids2017_processed.csv \
		--report reports/benchmark_cicids2017.md --window-unit flows

feed:            ## Replay flows into a running server to demo live ingest
	python tools/feeder.py --csv data/sample_flows.csv --rate 300

docker:          ## Build and run in Docker
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
