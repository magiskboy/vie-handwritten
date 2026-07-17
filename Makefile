# Vietnamese handwritten OCR - common tasks.
# Override variables on the command line, e.g.:
#   make evaluate CKPT=checkpoints/best.weights.h5 SPLIT=test DECODE=beam_lm

CONFIG     ?= configs/default.yaml
CKPT       ?= checkpoints/best.weights.h5
SPLIT      ?= test
TUNE_SPLIT ?= val
CLI        := uv run python -m vie_handwritten.cli
KENLM_DIR  := third_party/kenlm

.PHONY: help sync sync-gui build-kenlm build-data build-lm train evaluate infer tune-lm gui clean-lm

help:
	@echo "Targets (override vars like CKPT=, IMAGE=, SPLIT=, DECODE=, MAX=):"
	@echo "  sync         Install/refresh Python deps (uv sync)"
	@echo "  sync-gui     Install deps including GTK GUI extra (pygobject)"
	@echo "  build-kenlm  Build KenLM lmplz/build_binary from submodule (needs cmake + boost-devel)"
	@echo "  build-data   Build normalized JSONL manifests"
	@echo "  build-lm     Train KenLM syllable LM from train split (ORDER=, PRUNE=)"
	@echo "  train        Train CRNN, 2 phases (RESUME=)"
	@echo "  evaluate     CER/WER on a split (CKPT=, SPLIT=, DECODE=, MAX=)"
	@echo "  infer        OCR one image (IMAGE=, CKPT=, DECODE=)"
	@echo "  tune-lm      Grid-search KenLM alpha/beta on val (CKPT=, MAX=, ALPHAS=, BETAS=)"
	@echo "  gui          Launch GTK4 OCR viewer (vie-ocr-gui)"
	@echo "  clean-lm     Remove generated LM artifacts (lm/)"

sync:
	uv sync

sync-gui:
	uv sync --extra gui

gui:
	uv run vie-ocr-gui

build-kenlm:
	@command -v cmake >/dev/null 2>&1 || { echo "cmake not found. Install via package manager (e.g. 'sudo dnf install cmake')."; exit 1; }
	@test -f /usr/include/boost/version.hpp || { echo "boost-devel not found. Install via package manager (e.g. 'sudo dnf install boost-devel')."; exit 1; }
	@test -f $(KENLM_DIR)/CMakeLists.txt || { echo "KenLM submodule missing: git submodule update --init --recursive"; exit 1; }
	cmake -B $(KENLM_DIR)/build -S $(KENLM_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(KENLM_DIR)/build -j$$(nproc) --target lmplz build_binary
	@echo "Built: $(KENLM_DIR)/build/bin/{lmplz,build_binary}"

build-data:
	$(CLI) build-data --config $(CONFIG) $(if $(REBUILD),--rebuild,)

build-lm:
	$(CLI) build-lm --config $(CONFIG) $(if $(ORDER),--order $(ORDER),) $(if $(PRUNE),--prune $(PRUNE),)

train:
	$(CLI) train --config $(CONFIG) $(if $(RESUME),--resume $(RESUME),) $(if $(REBUILD),--rebuild-data,)

evaluate:
	$(CLI) evaluate --config $(CONFIG) --checkpoint $(CKPT) --split $(SPLIT) $(if $(DECODE),--decode $(DECODE),) $(if $(MAX),--max-samples $(MAX),)

infer:
	$(CLI) infer --config $(CONFIG) --checkpoint $(CKPT) --image $(IMAGE) $(if $(DECODE),--decode $(DECODE),)

tune-lm:
	$(CLI) tune-lm --config $(CONFIG) --checkpoint $(CKPT) --split $(TUNE_SPLIT) $(if $(MAX),--max-samples $(MAX),) $(if $(ALPHAS),--alphas $(ALPHAS),) $(if $(BETAS),--betas $(BETAS),)

clean-lm:
	rm -rf lm/
