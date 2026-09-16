# mirror -- one command per thing. `make help` lists them.
.PHONY: help setup check lint types test test-all cov model clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

# The venv lives OFF the exFAT drive on purpose. Installing onto exFAT fails:
# macOS writes ._* inside the extracted wheel and uv's RECORD check rejects it
# ("could not find entry for: ruff-x.y.z.data/scripts/._ruff"). Same convention
# as phi's ~/venvs/so101-sim.
VENV ?= $(HOME)/venvs/mirror
PY   := $(VENV)/bin/python

setup:  ## Create the venv (off exFAT) and install everything
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -e ".[perception,sim,dev]"

check: lint types test  ## THE GATE. Everything that must be green before a commit.

lint:  ## ruff
	$(PY) -m ruff check src tests

types:  ## mypy (strict)
	$(PY) -m mypy src

test:  ## the pure-math suite -- no camera, no arm, no mujoco
	$(PY) -m pytest -m "not perception and not sim and not hardware"

test-all:  ## everything installable locally (still never hardware)
	$(PY) -m pytest -m "not hardware"

cov:  ## coverage for the pure-math suite
	$(PY) -m pytest -m "not perception and not sim and not hardware" --cov=mirror --cov-report=term-missing

model:  ## Fetch the SO-101 MuJoCo model from upstream (not vendored -- 16 MB of meshes)
	@echo "TODO(phase-5): curl the SO101 dir from TheRobotStudio/SO-ARM100 into model/"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache; find . -name __pycache__ -prune -exec rm -rf {} +
