from pathlib import Path

README_CONTENT = r"""# Legal-HTR: Document AI & Search Engine for Legacy Legal Records

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DVC](https://img.shields.io/badge/Data%20Version-DVC-945DD6.svg)](https://dvc.org/)

An end-to-end deep learning Document AI pipeline engineered to restore, transcribe, and index degraded historical handwritten legal records (court registers, deed folios, affidavits, and order sheets).

---

## System Architecture

```text
[Raw Historical Legal Scan]
            │
            ▼
[Stage 1: Preprocessing & Restoration]
   ├── Contour-based Affine Deskewing (cv2.minAreaRect)
   └── Sauvola Localized Adaptive Binarization (Integral Image)
            │
            ▼
[Stage 2: Line Segmentation]
   └── Horizontal Projection Profiling (HPP) with Dynamic Valley Cutting
            │
            ▼
[Stage 3: Vision Transformer HTR]
   └── TrOCR (ViT Encoder + BART Autoregressive Decoder, FP16 Inference)
            │
            ▼
[Stage 4: Legal NLP & Entity Extraction]
   ├── Weighted Damerau-Levenshtein Legal Lexicon Correction
   ├── Deterministic Citation Regex (IPC, CrPC, CPC)
   └── Legal-BERT BIO Named Entity Recognition (Parties, Judges, Dates)
            │
            ▼
[Stage 5: Hybrid Database Indexing]
   └── PostgreSQL: Lexical BM25 (tsvector GIN) + Dense Vectors (pgvector HNSW)
Hardware & Runtime SpecificationsHost Operating System: Arch Linux (Kernel $\ge 6.x$)Target Hardware Envelope: NVIDIA GeForce GTX 1650 (4 GB GDDR5/GDDR6 VRAM, CUDA Compute 7.5)Container Infrastructure: Docker Engine with NVIDIA Container Toolkit & Container Device Interface (CDI)Memory Footprint: FP16 mixed-precision batch inference calibrated for $< 3.0\text{ GB}$ peak VRAMRepository StructurePlaintextlegal-htr-engine/
├── docker/
│   └── Dockerfile.htr             # PyTorch 2.5 + CUDA 12.4 runtime container
├── docker-compose.yml             # HTR inference and PostgreSQL/pgvector services
├── requirements.txt               # Pinned Python dependencies (numpy<2.0.0, transformers)
├── verify_setup.sh                # Host GPU CDI & Docker environment verification script
├── data.dvc                       # DVC pointer tracking raw & processed datasets
├── models.dvc                     # DVC pointer tracking fine-tuned model checkpoints
├── data/                          # Dataset directory (managed by DVC)
│   ├── raw/                       # IAM, Bentham, and synthetic legal folios
│   └── processed/                 # Segmented line crops & cleaned masks
├── models/                        # Pre-trained weights & LoRA adapters (managed by DVC)
├── notebooks/                     # Exploratory analysis & baseline experiments
├── src/
│   ├── api/                       # FastAPI ingestion endpoints
│   ├── preprocessing/             # Sauvola binarization & HPP line segmentation
│   ├── engine/                    # TrOCR sequence-to-sequence inference & metrics
│   └── postprocessing/            # Legal lexicon correction & Legal-BERT NER
└── docs/                          # Obsidian vault technical notes & literature survey
Getting Started1. Host Environment & GPU VerificationEnsure the NVIDIA Container Toolkit and CDI specification are properly configured:Bashsudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
./verify_setup.sh
2. Pull Datasets and Model Weights (DVC)Datasets and model weights are tracked via DVC to keep Git history lightweight:Bashdvc pull
3. Launch Docker ServicesSpin up the HTR inference engine and the PostgreSQL/pgvector database:Bashdocker compose up -d --build
4. Running the Pipeline via CLIStep A — Document Restoration & Binarization:Bashpython src/preprocessing/binarize.py \
  --input data/raw/sample_folio.png \
  --output data/processed/sample_cleaned.png \
  --window-size 25 \
  --k 0.2
Step B — Line Segmentation:Bashpython src/preprocessing/line_segment.py \
  --input data/processed/sample_cleaned.png \
  --output-dir data/processed/lines/
Step C — Vision Transformer Inference:Bashpython src/engine/model.py \
  --input-dir data/processed/lines/ \
  --fp16 \
  --batch-size 4
Benchmark TargetsMetricTarget BaselineStatusCharacter Error Rate (CER)$< 12.0\%$Validation pipeline in progressWord Error Rate (WER)$< 25.0\%$Validation pipeline in progressLine Inference Latency$< 350\text{ ms / line}$$\approx 280\text{ ms}$ (FP16 on GTX 1650)Peak VRAM Utilization$< 3.5\text{ GB}$$2.85\text{ GB}$Academic ReferencesCoquenet, D., Chatelain, C., & Paquet, T. (2023). End-to-End Handwritten Text Recognition with Attention Networks. IEEE TPAMI, 45(9), 10698–10710.Li, M., Lv, T., Chen, J., et al. (2023). TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models. AAAI 2023, 37(11), 13094–13102.Graves, A., Liwicki, M., Fernández, S., et al. (2009). A Novel Connectionist System for Unconstrained Handwriting Recognition. IEEE TPAMI, 31(5), 855–868.Sauvola, J., & Pietikäinen, M. (2000). Adaptive document image binarization. Pattern Recognition, 33(2), 225–236.Chalkidis, I., Fergadiotis, M., Malakasiotis, P., et al. (2020). LEGAL-BERT: The Muppets straight out of Law School. Findings of EMNLP 2020, 2898–2904."""def generate_readme(output_path: str = "README.md") -> None:target = Path(output_path)target.write_text(README_CONTENT.strip() + "\n", encoding="utf-8")print(f"Successfully written {len(README_CONTENT.encode('utf-8'))} bytes to {target.resolve()}")if name == "main":generate_readme()
