import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Union
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TrOCREngine:
    """
    Inference engine for Vision Transformer Handwritten Text Recognition (TrOCR)
    optimized for FP16 execution within a 4 GB VRAM budget.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        device: str = None,
        fp16: bool = True
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.fp16 = fp16 and (self.device.type == "cuda")
        print(f"[INIT] Initializing TrOCR engine on device: {self.device} (FP16: {self.fp16})")

        print(f"[INIT] Loading processor: {model_name}...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)

        print(f"[INIT] Loading VisionEncoderDecoderModel weights...")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        if self.fp16:
            self.model.half()

        self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            print(f"[INIT] Model loaded. Initial VRAM allocated: {allocated:.2f} MB")

    @torch.inference_mode()
    def predict_batch(self, images: List[Image.Image]) -> List[str]:
        """Runs autoregressive greedy decoding on a batch of PIL line images."""
        if not images:
            return []

        # Convert images to RGB and preprocess into normalized pixel tensors
        rgb_images = [img.convert("RGB") for img in images]
        pixel_values = self.processor(rgb_images, return_tensors="pt").pixel_values

        if self.fp16:
            pixel_values = pixel_values.half()

        pixel_values = pixel_values.to(self.device)

        # Autoregressive generation
        generated_ids = self.model.generate(
            pixel_values,
            max_new_tokens=64,
            early_stopping=True,
            num_beams=1,  # Greedy search for maximum throughput
            no_repeat_ngram_size=3
        )

        transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [t.strip() for t in transcriptions]

    def predict_directory(
        self,
        input_dir: Union[str, Path],
        batch_size: int = 4
    ) -> List[Dict[str, Union[str, float]]]:
        """Transcribes all supported line images in a folder in batches."""
        target_dir = Path(input_dir)
        supported_exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        image_paths = sorted([p for p in target_dir.iterdir() if p.suffix.lower() in supported_exts])

        if not image_paths:
            print(f"[WARN] No image files found in {input_dir}")
            return []

        print(f"[RUN] Processing {len(image_paths)} images from {target_dir} (Batch size: {batch_size})...")
        results = []
        total_time = 0.0

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = [Image.open(p) for p in batch_paths]

            start_t = time.perf_counter()
            transcriptions = self.predict_batch(batch_images)
            batch_elapsed = time.perf_counter() - start_t
            total_time += batch_elapsed

            per_line_ms = (batch_elapsed / len(batch_paths)) * 1000.0

            for p, text in zip(batch_paths, transcriptions):
                results.append({
                    "image": p.name,
                    "text": text,
                    "latency_ms": round(per_line_ms, 2)
                })
                print(f"  [{p.name}] -> \"{text}\" ({per_line_ms:.1f} ms)")

        avg_latency = (total_time / len(image_paths)) * 1000.0
        print(f"\n[SUMMARY] Processed {len(image_paths)} lines in {total_time:.2f}s (Avg: {avg_latency:.1f} ms/line)")

        if self.device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            print(f"[SUMMARY] Peak VRAM consumed: {peak_vram:.2f} MB")

        return results


def parse_args():
    parser = argparse.ArgumentParser(description="TrOCR FP16 Line Transcription Engine.")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing line crops")
    parser.add_argument("-o", "--output", default="transcriptions.json", help="Path to save output JSON")
    parser.add_argument("--model-name", default="microsoft/trocr-base-handwritten", help="Hugging Face model checkpoint")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 mixed precision")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine = TrOCREngine(
        model_name=args.model_name,
        fp16=not args.no_fp16
    )
    records = engine.predict_directory(args.input_dir, batch_size=args.batch_size)

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Saved results to: {out_path}")
