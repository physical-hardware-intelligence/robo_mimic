# robo_mimic -- one command per thing. `make help` lists them.
.PHONY: help setup lock check lint types test test-all cov assets fixtures doctor view record replay model sim teleop bench clean scrub cameras

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

# The venv lives OFF the exFAT drive on purpose. Installing onto exFAT fails:
# macOS writes ._* inside the extracted wheel and uv's RECORD check rejects it
# ("could not find entry for: ruff-x.y.z.data/scripts/._ruff"). Same convention
# as phi's ~/venvs/so101-sim. uv's default is a project-local .venv, which is
# exactly the broken case, so `setup` overrides it with UV_PROJECT_ENVIRONMENT.
VENV ?= $(HOME)/venvs/robo_mimic
# Use the venv when it exists, otherwise whatever `python` is on PATH. CI
# installs into the runner's system Python and has no venv at this path, so a
# hardcoded $(VENV)/bin/python fails there with "No such file or directory" --
# which is exactly how the first public CI run broke.
PY   := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python)

# Which webcam. `?=` means an exported CAMERA in your shell wins, so you can set
# it once instead of passing it every run. Index order is AVFoundation's, and an
# external USB camera often enumerates BEFORE the built-in one -- `make cameras`
# prints the list. Every target that opens a camera honours this, doctor included.
CAMERA ?= 0

setup:  ## Create the venv (off exFAT) and install the EXACT locked versions
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync --locked --all-extras

# `--locked` fails loudly if pyproject and uv.lock disagree instead of silently
# installing something stale. CI deliberately does NOT use the lock: floating
# there is the canary that catches an upstream release breaking us.
lock:  ## Re-resolve uv.lock. Run this after editing dependencies in pyproject.
	uv lock

check: lint types test  ## THE GATE. Everything that must be green before a commit.

lint:  ## ruff
	$(PY) -m ruff check src tests scripts

# `src` ONLY for two months, and scripts/ is where the operator-facing code
# lives. A --mirror flag read `RetargetConfig.lateral_sign` off a slots=True
# dataclass, got the slot DESCRIPTOR instead of -1.0, and crashed teleop on
# the first frame. mypy catches `member_descriptor * float` instantly; it was
# simply never pointed at the file.
types:  ## mypy (strict), src AND scripts
	$(PY) -m mypy src scripts

test:  ## the pure-math suite -- no camera, no arm, no mujoco
	$(PY) -m pytest -m "not perception and not sim and not hardware"

test-all:  ## everything installable locally (still never hardware)
	$(PY) -m pytest -m "not hardware"

cov:  ## coverage for the pure-math suite
	$(PY) -m pytest -m "not perception and not sim and not hardware" --cov=robo_mimic --cov-report=term-missing

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
	@-$(PY) scripts/doctor.py --camera $(CAMERA)

view: assets  ## Live hand tracking, landmarks only -- NO simulator (that is `teleop`)
	$(PY) scripts/record.py --camera $(CAMERA) $(ARGS)

record: assets  ## Same, but SPACE starts/stops capturing a clip to fixtures/clips/
	$(PY) scripts/record.py --camera $(CAMERA) --name $(or $(NAME),clip) $(ARGS)

replay: assets  ## Re-run a recorded clip through the identical pipeline: make replay CLIP=path.mp4
	@test -n "$(CLIP)" || { echo "usage: make replay CLIP=fixtures/clips/xxx.mp4"; exit 1; }
	$(PY) scripts/record.py --video $(CLIP)

model:  ## Fetch the SO-101 MuJoCo model from upstream (16.4 MB of meshes, hash-pinned)
	$(PY) scripts/fetch_model.py

sim: model  ## Drive the sim from a synthetic hand trajectory and report tracking error
	$(PY) scripts/run_sim.py $(ARGS)

teleop: assets model  ## LIVE: your hand on the left, the simulated arm on the right
	$(PY) scripts/teleop.py --camera $(CAMERA) $(ARGS)

bench: model  ## Per-stage latency budget, no camera and no window
	$(PY) scripts/teleop.py --source synthetic --bench --frames 400 --engage

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache; find . -name __pycache__ -prune -exec rm -rf {} +
	$(MAKE) --no-print-directory scrub

# exFAT stores no extended attributes, so macOS externalises every xattr into a
# ._* sidecar. git writes new pack files with a com.apple.provenance xattr, so a
# `git gc` or `git clone` here leaves ._pack-*.idx next to the real index -- and
# git then tries to READ it as a pack index:
#   error: non-monotonic index .git/objects/pack/._pack-<sha>.idx
# Harmless but printed on every later git command. Deleting the sidecar deletes
# the xattr, which is the whole fix. Nothing prevents it recurring; run this
# after any operation that writes a pack.
# ffmpeg's enumeration IS the one OpenCV uses (both go through AVFoundation),
# which is why this is the authoritative answer and not a guess. Stop at the
# audio section: those indices are microphones and mean nothing to --camera.
cameras:  ## List the webcams macOS sees, with the CAMERA= value for each
	@ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 \
	  | awk '/video devices/{on=1;next} /audio devices/{on=0} on' \
	  | sed -n 's/.*\[\([0-9]*\)\] \(.*\)/  CAMERA=\1  \2/p' \
	  || system_profiler SPCameraDataType

scrub:  ## Delete the ._* AppleDouble files exFAT forces macOS to write
	find . -name '._*' -delete
