from pathlib import Path
from typing import List, Dict, Union
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class HTREngine:
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        device: str = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing HTR Engine on device: {self.device}")
        
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        
        # Load in fp16 on CUDA to minimize VRAM footprint on 4GB GPUs
        if self.device == "cuda":
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)

        self.model.eval()

    def predict_line(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Union[str, float]]:
        """
        Transcribes a single text line crop and calculates an average token confidence score.
        """
        if isinstance(image_input, (str, Path)):
            image = Image.open(str(image_input)).convert("RGB")
        else:
            image = image_input.convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        if self.device == "cuda":
            pixel_values = pixel_values.to(self.device, dtype=torch.float16)
        else:
            pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=64
            )

        # Decode generated token IDs to plain text
        generated_ids = outputs.sequences
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Calculate approximate sequence confidence from token log-softmax scores
        if outputs.scores:
            logits = torch.stack(outputs.scores, dim=1)  # shape: (1, seq_len, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            token_ids = generated_ids[:, 1:]  # Exclude decoder start token
            
            # Gather probabilities of the chosen tokens
            token_probs = torch.gather(probs, 2, token_ids.unsqueeze(-1)).squeeze(-1)
            confidence = float(token_probs.mean().item())
        else:
            confidence = 1.0

        return {
            "text": transcription.strip(),
            "confidence": round(confidence, 4)
        }

    def predict_batch_directory(self, lines_dir: Union[str, Path]) -> List[Dict[str, Union[str, float]]]:
        """
        Transcribes all line crops inside a directory ordered by filename.
        """
        directory = Path(lines_dir)
        line_files = sorted(directory.glob("line_*.png"))
        results = []

        for line_path in line_files:
            res = self.predict_line(line_path)
            res["filename"] = line_path.name
            results.append(res)

        return results


if __name__ == "__main__":
    import sys
    import json

    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/lines/")
    
    engine = HTREngine()

    if input_path.is_dir():
        print(f"Processing lines in directory: {input_path}")
        outputs = engine.predict_batch_directory(input_path)
        print(json.dumps(outputs, indent=2))
    elif input_path.is_file():
        print(f"Processing single file: {input_path}")
        output = engine.predict_line(input_path)
        print(json.dumps(output, indent=2))
    else:
        print(f"Invalid path: {input_path}")
