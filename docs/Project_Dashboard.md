---
project: Legacy Legal Records HTR
type: project-dashboard
status: in-progress
current_phase: 5th Sem Mini-Project
tags:
  - ai-ml
  - computer-vision
  - htr
  - college-project
  - arch-linux
hardware:
  gpu: NVIDIA GeForce GTX 1650 (4GB VRAM)
  compute: CUDA 12.4 / PyTorch 2.5
---

# Deep Learning Legal HTR & Searchable Repository

> [!abstract] Core Objective
> Build an end-to-end Document AI system to segment, transcribe, and index degraded historical and handwritten legal records into an auditable, searchable digital archive.

---

## 📌 Quick Reference & Connected Notes
- **Repository:** `legal-htr-engine`
- **Architecture & Runbook:** [[HTR_Pipeline_Architecture_and_Troubleshooting]]
- **Preprocessing Logic:** `src/preprocessing/binarize.py`
- **Line Slicing:** `src/preprocessing/line_segment.py`
- **Inference Engine:** `src/engine/model.py`
- **Container Spec:** `docker/Dockerfile.htr` & `docker-compose.yml`

---

## 🎯 Phase 1: 5th Semester Mini-Project (Proof of Concept)
**Target:** Single-page image cleanup, line extraction, and isolated handwritten transcription with confidence scoring.

### 1. Data Pipeline & Environment
- [x] Configure Docker container with PyTorch 2.5 + CUDA 12.4 runtime
- [x] Configure host NVIDIA Container Toolkit & CDI specification (`/etc/cdi/nvidia.yaml`) for GTX 1650
- [x] Fix non-root Docker execution (switch context from `desktop-linux` to `default`)
- [x] Initialize DVC repository structure (`data.dvc`, `models.dvc`) with local `.venv`
- [ ] Ingest IAM handwriting dataset benchmark split into `data/raw/iam/`
- [ ] Ingest historical cursive legal scans (Bentham / Transkribus public records)

### 2. Preprocessing Module (`src/preprocessing/`)
- [x] Implement MinAreaRect contour deskewing (`deskew_image`)
- [x] Implement Sauvola adaptive binarization ($k=0.2, w=25, R=128$) for degraded parchment
- [x] Implement median noise filtering for scan dust and bleed-through removal
- [x] Implement Horizontal Projection Profiling (HPP) for line-level slicing (`src/preprocessing/line_segment.py`)
- [ ] Add vertical seam-carving fallback for curved or tightly packed lines

### 3. Model Baseline & Evaluation (`src/engine/`)
- [x] Integrate `microsoft/trocr-base-handwritten` with Hugging Face Transformers
- [x] Optimize inference for 4GB VRAM using half-precision (`torch.float16`, peak VRAM < 1 GB)
- [x] Implement sequence-level confidence score calculation via softmax token probability averaging
- [ ] Build evaluation harness using `jiwer` for Character Error Rate (CER) and Word Error Rate (WER)
- [ ] Generate comparative confusion matrices on common legal character confusions ($c/e$, $f/s$, $1/l$)
- [ ] Prepare fine-tuning script (`src/engine/train.py`) with LoRA / PEFT for domain adaptation

### 4. Application & Presentation Deliverables
- [x] Scaffold FastAPI health checks and service structure (`src/api/main.py`)
- [ ] Wire end-to-end pipeline route (`POST /api/v1/transcribe`) accepting page scans
- [ ] Build Streamlit interactive UI (`app/streamlit_app.py`) with side-by-side scan & transcript viewer
- [ ] Export evaluation charts, CER benchmarks, and project deck for 5th-semester viva

---

## 🔮 Phase 2: Final Year Major Project (Full Archival Platform)
**Target:** Multi-region layout analysis, legal NER, hybrid vector search, and multi-tenant web repository.

### 1. Document Layout Analysis (DLA)
- [ ] Annotate legal pages for multi-element layout (court stamps, marginalia, signatures, typed headers)
- [ ] Train YOLOv8-Doc / LayoutLMv3 to segment complex structural regions

### 2. Information Extraction & Legal NLP
- [ ] Constrain TrOCR decoding using n-gram legal lexicon / court vocabulary
- [ ] Fine-tune Legal-BERT for Named Entity Recognition (Court, Petitioner, Respondent, Acts, Dates)

### 3. Search & Archival Infrastructure
- [ ] Spin up PostgreSQL with `pgvector` container
- [ ] Implement hybrid retrieval: BM25 lexical index + BGE-M3 dense embeddings
- [ ] Build PDF/A export pipeline with invisible searchable OCR text layers

---

## 📝 Sprint Log

- **2026-09-20:** Project scoped. Scaffolding created, Git and DVC initialized.
- **2026-09-21:**
  - Resolved Docker CDI discovery error (`failed to discover GPU vendor from CDI`) by switching Docker context from `desktop-linux` to `default` and generating `/etc/cdi/nvidia.yaml`.
  - Resolved NumPy 2.x ABI incompatibility by updating Docker base to `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` and pinning `numpy<2.0.0`.
  - Added `sentencepiece` and `protobuf` dependencies for Hugging Face tokenizers.
  - Implemented and verified `src/preprocessing/binarize.py` (Sauvola adaptive thresholding + deskewing).
  - Implemented `src/preprocessing/line_segment.py` using Horizontal Projection Profiling (HPP).
  - Implemented `src/engine/model.py` with `microsoft/trocr-base-handwritten` running in FP16 on the GTX 1650.
  - Created architecture guide and runbook in [[HTR_Pipeline_Architecture_and_Troubleshooting]].
