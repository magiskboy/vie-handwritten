# Vietnamese handwritten OCR - common tasks.
# Override variables on the command line, e.g.:
#   make evaluate CKPT=checkpoints SPLIT=test DECODE=beam_lm

CONFIG     ?= configs/default.yaml
CKPT       ?= checkpoints
SPLIT      ?= test
TUNE_SPLIT ?= val
CLI        := uv run python -m vie_handwritten.cli
KENLM_DIR  := third_party/kenlm

.PHONY: help sync sync-gui build-kenlm build-data build-lm train train-word train-line curriculum evaluate infer tune-lm gui clean-lm

help:
	@echo "Targets (override vars like CKPT=, IMAGE=, SPLIT=, DECODE=, MAX=):"
	@echo "  sync         Install/refresh Python deps (uv sync)"
	@echo "  sync-gui     Install deps including GTK GUI extra (pygobject)"
	@echo "  build-kenlm  Build KenLM lmplz/build_binary from submodule (needs cmake + boost-devel)"
	@echo "  build-data   Build normalized JSONL manifests"
	@echo "  build-lm     Train KenLM syllable LM from train split (ORDER=, PRUNE=)"
	@echo "  train        Train CRNN, 2 phases (RESUME=checkpoint-dir)"
	@echo "  train-word   Curriculum stage 1: pretrain trên HWDB_word → checkpoints/word"
	@echo "  train-line   Curriculum stage 2: fine-tune HWDB_line, resume từ checkpoints/word"
	@echo "  curriculum   Chạy tuần tự train-word rồi train-line (word → line)"
	@echo "  evaluate     CER/WER on a split (CKPT=dir, SPLIT=, DECODE=, MAX=)"
	@echo "  infer        OCR one image (IMAGE=, CKPT=dir, DECODE=)"
	@echo "  tune-lm      Grid-search KenLM alpha/beta on val (CKPT=dir, MAX=, ALPHAS=, BETAS=)"
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

# Curriculum word → line. WORD_CKPT là thư mục checkpoint stage 1 mà stage 2 resume từ đó.
WORD_CFG   ?= configs/word.yaml
LINE_CFG   ?= configs/line.yaml
WORD_CKPT  ?= checkpoints/word

train-word:
	$(CLI) build-data --config $(WORD_CFG)
	$(CLI) train --config $(WORD_CFG)

train-line:
	$(CLI) build-data --config $(LINE_CFG)
	$(CLI) train --config $(LINE_CFG) --resume $(WORD_CKPT)

curriculum: train-word train-line

# evaluate / infer / tune-lm load CKPT dir: {model.weights.h5, config.yaml}
evaluate:
	$(CLI) evaluate --checkpoint $(CKPT) --split $(SPLIT) $(if $(DECODE),--decode $(DECODE),) $(if $(MAX),--max-samples $(MAX),)

infer:
	$(CLI) infer --checkpoint $(CKPT) --image $(IMAGE) $(if $(DECODE),--decode $(DECODE),)

tune-lm:
	$(CLI) tune-lm --checkpoint $(CKPT) --split $(TUNE_SPLIT) $(if $(MAX),--max-samples $(MAX),) $(if $(ALPHAS),--alphas $(ALPHAS),) $(if $(BETAS),--betas $(BETAS),)

clean-lm:
	rm -rf lm/
