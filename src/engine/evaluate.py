import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
from PIL import Image
import torch

from src.engine.model import TrOCREngine


def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> int:
    """
    Computes exact Levenshtein distance using Wagner-Fischer 1D dynamic programming.
    Operates on characters (for CER) or word token sequences (for WER).
    """
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)

    if len(seq2) == 0:
        return len(seq1)

    previous_row = list(range(len(seq2) + 1))
    for i, elem1 in enumerate(seq1):
        current_row = [i + 1]
        for j, elem2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (1 if elem1 != elem2 else 0)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_metrics(reference: str, hypothesis: str) -> Dict[str, Any]:
    """Computes sample-level edit distances, CER, and WER."""
    ref_clean = reference.strip()
    hyp_clean = hypothesis.strip()

    # Character-level edit distance
    ref_chars = list(ref_clean)
    hyp_chars = list(hyp_clean)
    char_dist = levenshtein_distance(ref_chars, hyp_chars)
    char_len = max(len(ref_chars), 1)
    cer = min(1.0, char_dist / char_len)

    # Word-level edit distance
    ref_words = ref_clean.split()
    hyp_words = hyp_clean.split()
    word_dist = levenshtein_distance(ref_words, hyp_words)
    word_len = max(len(ref_words), 1)
    wer = min(1.0, word_dist / word_len)

    return {
        "char_dist": char_dist,
        "char_len": len(ref_chars),
        "word_dist": word_dist,
        "word_len": len(ref_words),
        "cer": cer,
        "wer": wer,
    }


def evaluate_benchmark(
    engine: TrOCREngine,
    tsv_path: Path,
    dataset_root: Path,
    batch_size: int = 4,
    max_samples: int = None
) -> Dict[str, Any]:
    """Evaluates TrOCR on a benchmark split TSV file."""
    if not tsv_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")
    if max_samples:
        df = df.head(max_samples)

    print(f"\n=======================================================")
    print(f"Benchmarking: {tsv_path.name} ({len(df)} samples)")
    print(f"=======================================================")

    records = []
    total_char_dist = 0
    total_char_len = 0
    total_word_dist = 0
    total_word_len = 0
    total_latency_sec = 0.0

    # Process in batches
    for i in range(0, len(df), batch_size):
        batch_slice = df.iloc[i : i + batch_size]
        batch_images = []
        valid_indices = []

        for idx, row in batch_slice.iterrows():
            img_rel_path = row["image_path"]
            img_abs_path = dataset_root / img_rel_path

            if not img_abs_path.exists():
                print(f"[WARN] Image file missing: {img_abs_path}")
                continue

            try:
                img = Image.open(img_abs_path).convert("RGB")
                batch_images.append(img)
                valid_indices.append((idx, row))
            except Exception as e:
                print(f"[WARN] Failed to open {img_abs_path}: {e}")

        if not batch_images:
            continue

        # Predict batch
        start_t = time.perf_counter()
        predictions = engine.predict_batch(batch_images)
        batch_duration = time.perf_counter() - start_t
        total_latency_sec += batch_duration

        per_line_ms = (batch_duration / len(batch_images)) * 1000.0

        for (_, row), hyp_text in zip(valid_indices, predictions):
            ref_text = str(row["text"])
            metrics = compute_metrics(ref_text, hyp_text)

            total_char_dist += metrics["char_dist"]
            total_char_len += metrics["char_len"]
            total_word_dist += metrics["word_dist"]
            total_word_len += metrics["word_len"]

            record = {
                "line_id": str(row["line_id"]),
                "reference": ref_text,
                "hypothesis": hyp_text,
                "cer": round(metrics["cer"] * 100.0, 2),
                "wer": round(metrics["wer"] * 100.0, 2),
                "latency_ms": round(per_line_ms, 2),
            }
            records.append(record)

    # Compute corpus-level macro metrics
    corpus_cer = (total_char_dist / max(total_char_len, 1)) * 100.0
    corpus_wer = (total_word_dist / max(total_word_len, 1)) * 100.0
    avg_latency = (total_latency_sec / max(len(records), 1)) * 100.0 * 10.0  # ms/sample

    summary = {
        "dataset": tsv_path.parent.name,
        "split": tsv_path.stem,
        "samples_evaluated": len(records),
        "corpus_cer_pct": round(corpus_cer, 2),
        "corpus_wer_pct": round(corpus_wer, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "samples": records,
    }

    print(f"\n--- Benchmark Results: {tsv_path.parent.name}/{tsv_path.stem} ---")
    print(f"Samples Evaluated : {len(records)}")
    print(f"Corpus CER        : {corpus_cer:.2f}% (Target: < 12.00%)")
    print(f"Corpus WER        : {corpus_wer:.2f}% (Target: < 25.00%)")
    print(f"Mean Line Latency : {avg_latency:.2f} ms")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrOCR on benchmark test splits.")
    parser.add_argument("--benchmark", choices=["iam", "bentham", "all"], default="all")
    parser.add_argument("--data-root", default="data/raw", help="Path containing dataset folders")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for FP16 inference")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap evaluation samples for quick checks")
    parser.add_argument("--output-dir", default="data/processed/benchmarks", help="Output directory for JSON reports")
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.benchmark in ["iam", "all"]:
        targets.append((data_root / "iam" / "test.tsv", data_root / "iam"))
    if args.benchmark in ["bentham", "all"]:
        targets.append((data_root / "bentham" / "test.tsv", data_root / "bentham"))

    # Initialize model once
    engine = TrOCREngine(model_name=args.model_name, fp16=True)

    all_summaries = {}
    for tsv_file, ds_root in targets:
        if not tsv_file.exists():
            print(f"[SKIP] {tsv_file} does not exist.")
            continue

        result = evaluate_benchmark(
            engine=engine,
            tsv_path=tsv_file,
            dataset_root=ds_root,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        all_summaries[result["dataset"]] = result

        # Export individual JSON
        out_file = out_dir / f"eval_{result['dataset']}_{result['split']}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved report to: {out_file}")

    # Export consolidated summary
    consolidated_file = out_dir / "summary_eval_report.json"
    with open(consolidated_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Consolidated benchmark report written to {consolidated_file}")


if __name__ == "__main__":
    main()
