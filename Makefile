# mirror -- one command per thing. `make help` lists them.
.PHONY: help setup check lint types test test-all cov assets fixtures doctor view record replay model sim teleop bench clean

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
	$(PY) -m ruff check src tests scripts

types:  ## mypy (strict)
	$(PY) -m mypy src

test:  ## the pure-math suite -- no camera, no arm, no mujoco
	$(PY) -m pytest -m "not perception and not sim and not hardware"

test-all:  ## everything installable locally (still never hardware)
	$(PY) -m pytest -m "not hardware"

cov:  ## coverage for the pure-math suite
	$(PY) -m pytest -m "not perception and not sim and not hardware" --cov=mirror --cov-report=term-missing

MP_MODEL := https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
MP_IMAGE := https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg

assets:  ## Fetch the hand-landmarker model + reference image (not committed)
	@mkdir -p assets fixtures/images
	@test -f assets/hand_landmarker.task || curl -sL -o assets/hand_landmarker.task $(MP_MODEL)
	@test -f fixtures/images/woman_hands.jpg || curl -sL -o fixtures/images/woman_hands.jpg $(MP_IMAGE)
	@shasum -a 256 assets/hand_landmarker.task fixtures/images/woman_hands.jpg

fixtures: assets  ## Regenerate the committed landmark goldens from the reference image
	$(PY) scripts/make_fixtures.py

doctor:  ## Check deps, model, camera permission, and the pipeline end to end
	@-$(PY) scripts/doctor.py

view: assets  ## Live hand tracking from the built-in camera (no recording)
	$(PY) scripts/record.py

record: assets  ## Same, but SPACE starts/stops capturing a clip to fixtures/clips/
	$(PY) scripts/record.py --name $(or $(NAME),clip)

replay: assets  ## Re-run a recorded clip through the identical pipeline: make replay CLIP=path.mp4
	@test -n "$(CLIP)" || { echo "usage: make replay CLIP=fixtures/clips/xxx.mp4"; exit 1; }
	$(PY) scripts/record.py --video $(CLIP)

model:  ## Fetch the SO-101 MuJoCo model from upstream (16.4 MB of meshes, hash-pinned)
	$(PY) scripts/fetch_model.py

sim: model  ## Drive the sim from a synthetic hand trajectory and report tracking error
	$(PY) scripts/run_sim.py $(ARGS)

teleop: assets model  ## LIVE: your hand on the left, the simulated arm on the right
	$(PY) scripts/teleop.py $(ARGS)

bench: model  ## Per-stage latency budget, no camera and no window
	$(PY) scripts/teleop.py --source synthetic --bench --frames 400 --engage

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache; find . -name __pycache__ -prune -exec rm -rf {} +
