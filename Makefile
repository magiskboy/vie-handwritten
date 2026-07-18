# Vietnamese handwritten OCR - common tasks.
# Override variables on the command line, e.g.:
#   make evaluate CKPT=checkpoints SPLIT=test DECODE=beam_lm

CONFIG     ?= configs/default.yaml
CKPT       ?= checkpoints
SPLIT      ?= test
TUNE_SPLIT ?= val
CLI        := uv run python -m vie_handwritten.cli
OVCLI      := uv run python -m converter.cli
KENLM_DIR  := third_party/kenlm
OVDIR      ?= $(CKPT)/openvino

.PHONY: help sync sync-gui sync-ov build-kenlm build-lm train train-word train-line curriculum evaluate infer tune-lm gui clean-lm convert-ov bench-ov-acc bench-ov-perf demo demo-ov demo-vendor sync-demo

help:
	@echo "Targets (override vars like CKPT=, IMAGE=, SPLIT=, DECODE=, MAX=):"
	@echo "  sync         Install/refresh Python deps (uv sync)"
	@echo "  sync-gui     Install deps including GTK GUI extra (pygobject)"
	@echo "  sync-demo    Install form-ocr-demo workspace package + openvino"
	@echo "  build-kenlm  Build KenLM lmplz/build_binary from vendored source (needs cmake + boost-devel)"
	@echo "  build-lm     Train KenLM syllable LM from train split (ORDER=, PRUNE=)"
	@echo "  train        Train CRNN, 2 phases (RESUME=checkpoint-dir)"
	@echo "  train-word   Curriculum stage 1: pretrain trên HWDB_word → checkpoints/word"
	@echo "  train-line   Curriculum stage 2: fine-tune HWDB_line, resume từ checkpoints/word"
	@echo "  curriculum   Chạy tuần tự train-word rồi train-line (word → line)"
	@echo "  evaluate     CER/WER on a split (CKPT=dir, SPLIT=, DECODE=, MAX=, FAILS=)"
	@echo "  infer        OCR one image (IMAGE=, CKPT=dir, DECODE=)"
	@echo "  tune-lm      Grid-search KenLM alpha/beta on val (CKPT=dir, MAX=, ALPHAS=, BETAS=)"
	@echo "  gui          Launch GTK4 OCR viewer (vie-ocr-gui)"
	@echo "  demo         Launch form-field extraction demo (form-ocr)"
	@echo "  demo-vendor  Copy $(CKPT)/openvino → demo/models/openvino"
	@echo "  demo-ov      Convert CKPT to OV (BATCHES=1) + vendor into demo/models"
	@echo "  convert-ov   Convert checkpoint -> OpenVINO FP16+INT8 IR (CKPT=dir)"
	@echo "  bench-ov-acc CER/WER: OV variants vs Keras (CKPT=dir, OVDIR=, SPLIT=)"
	@echo "  bench-ov-perf CPU latency/throughput per precision x batch (CKPT=dir, OVDIR=)"
	@echo "  clean-lm     Remove generated LM artifacts (lm/)"

sync:
	uv sync

sync-gui:
	uv sync --extra gui

sync-ov:
	uv sync --extra openvino

gui:
	uv run vie-ocr-gui

sync-demo:
	uv sync --all-packages --extra openvino

demo:
	uv run --package form-ocr-demo form-ocr

# Vendor a converted OpenVINO artifact into the demo app tree.
DEMO_OV ?= demo/models/openvino

demo-vendor:
	@test -d $(CKPT)/openvino || { echo "Missing $(CKPT)/openvino — run: make convert-ov CKPT=$(CKPT) BATCHES=1"; exit 1; }
	rm -rf $(DEMO_OV)
	mkdir -p $(dir $(DEMO_OV))
	cp -a $(CKPT)/openvino $(DEMO_OV)
	@echo "Vendored $(CKPT)/openvino → $(DEMO_OV)"

# Convert (batch 1 only) + vendor for the form-ocr demo.
demo-ov: sync-demo
	$(MAKE) convert-ov CKPT=$(CKPT) BATCHES=1
	$(MAKE) demo-vendor CKPT=$(CKPT)

build-kenlm:
	@command -v cmake >/dev/null 2>&1 || { echo "cmake not found. Install via package manager (e.g. 'sudo dnf install cmake')."; exit 1; }
	@test -f /usr/include/boost/version.hpp || { echo "boost-devel not found. Install via package manager (e.g. 'sudo dnf install boost-devel')."; exit 1; }
	@test -f $(KENLM_DIR)/CMakeLists.txt || { echo "KenLM source missing at $(KENLM_DIR) (vendored tree; see $(KENLM_DIR)/SOURCE.md)"; exit 1; }
	cmake -B $(KENLM_DIR)/build -S $(KENLM_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(KENLM_DIR)/build -j$$(nproc) --target lmplz build_binary
	@echo "Built: $(KENLM_DIR)/build/bin/{lmplz,build_binary}"

build-lm:
	$(CLI) build-lm --config $(CONFIG) $(if $(ORDER),--order $(ORDER),) $(if $(PRUNE),--prune $(PRUNE),)

train:
	$(CLI) train --config $(CONFIG) $(if $(RESUME),--resume $(RESUME),)

# Curriculum word → line. WORD_CKPT là thư mục checkpoint stage 1 mà stage 2 resume từ đó.
WORD_CFG   ?= configs/word.yaml
LINE_CFG   ?= configs/line.yaml
WORD_CKPT  ?= checkpoints/word

train-word:
	$(CLI) train --config $(WORD_CFG)

train-line:
	$(CLI) train --config $(LINE_CFG) --resume $(WORD_CKPT)

curriculum: train-word train-line

# evaluate / infer / tune-lm load CKPT dir: {model.weights.h5, config.yaml}
evaluate:
	$(CLI) evaluate --checkpoint $(CKPT) --split $(SPLIT) $(if $(DECODE),--decode $(DECODE),) $(if $(MAX),--max-samples $(MAX),) $(if $(FAILS),--failures-out $(FAILS),)

infer:
	$(CLI) infer --checkpoint $(CKPT) --image $(IMAGE) $(if $(DECODE),--decode $(DECODE),)

tune-lm:
	$(CLI) tune-lm --checkpoint $(CKPT) --split $(TUNE_SPLIT) $(if $(MAX),--max-samples $(MAX),) $(if $(ALPHAS),--alphas $(ALPHAS),) $(if $(BETAS),--betas $(BETAS),)

# OpenVINO convert + benchmarks. CKPT is the checkpoint dir; OVDIR defaults to
# $(CKPT)/openvino. Needs the openvino extra (make sync-ov).
convert-ov:
	$(OVCLI) convert --checkpoint $(CKPT) $(if $(OVCONFIG),--config $(OVCONFIG),) $(if $(BATCHES),--batches $(BATCHES),)

bench-ov-acc:
	$(OVCLI) bench-accuracy --ov-dir $(OVDIR) --checkpoint $(CKPT) $(if $(SPLIT),--split $(SPLIT),) $(if $(MAX),--max-samples $(MAX),) $(if $(PRECISIONS),--precisions $(PRECISIONS),) $(if $(JSON),--json $(JSON),)

bench-ov-perf:
	$(OVCLI) bench-perf --ov-dir $(OVDIR) --checkpoint $(CKPT) $(if $(PRECISIONS),--precisions $(PRECISIONS),) $(if $(BATCHES),--batches $(BATCHES),) $(if $(JSON),--json $(JSON),)

clean-lm:
	rm -rf lm/
