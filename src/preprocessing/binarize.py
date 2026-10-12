import argparse
import sys
from pathlib import Path
import cv2
import numpy as np


def sauvola_threshold(image: np.ndarray, window_size: int = 25, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """Computes integral-image based Sauvola adaptive binarization."""
    if window_size % 2 == 0:
        window_size += 1

    half_w = window_size // 2
    img_float = image.astype(np.float64)

    # Pad image boundaries to prevent edge collapse
    padded = np.pad(img_float, half_w, mode='reflect')

    # Compute integral images for constant-time window sums
    integral = cv2.integral(padded)
    integral_sq = cv2.integral(padded ** 2)

    h, w = image.shape
    y1, y2 = 0, h
    x1, x2 = 0, w

    # Calculate local mean using integral area
    area = window_size * window_size
    mean = (
        integral[window_size : h + window_size, window_size : w + window_size]
        - integral[0:h, window_size : w + window_size]
        - integral[window_size : h + window_size, 0:w]
        + integral[0:h, 0:w]
    ) / area

    # Calculate local variance and standard deviation
    mean_sq = (
        integral_sq[window_size : h + window_size, window_size : w + window_size]
        - integral_sq[0:h, window_size : w + window_size]
        - integral_sq[window_size : h + window_size, 0:w]
        + integral_sq[0:h, 0:w]
    ) / area

    variance = np.maximum(mean_sq - mean ** 2, 0.0)
    std_dev = np.sqrt(variance)

    # Sauvola threshold surface formula: T = m * (1 + k * (s / R - 1))
    threshold = mean * (1.0 + k * ((std_dev / r) - 1.0))
    binary = np.where(image > threshold, 255, 0).astype(np.uint8)
    return binary


def preprocess_document(image_path: str, output_path: str, window_size: int = 25, k: float = 0.2) -> None:
    inp = Path(image_path)
    if not inp.exists():
        raise FileNotFoundError(f"Could not open image at {image_path}")

    # Load in grayscale
    img = cv2.imread(str(inp), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to decode image file at {image_path}")

    # Apply Sauvola thresholding
    binarized = sauvola_threshold(img, window_size=window_size, k=k)

    # Ensure output destination exists
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out), binarized)
    print(f"[SUCCESS] Saved binarized image to {out} (Shape: {binarized.shape})")


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive Sauvola binarization for degraded document records.")
    parser.add_argument("pos_input", nargs="?", default=None, help="Input image path (positional)")
    parser.add_argument("pos_output", nargs="?", default=None, help="Output image path (positional)")
    parser.add_argument("-i", "--input", dest="opt_input", help="Input image path")
    parser.add_argument("-o", "--output", dest="opt_output", help="Output image path")
    parser.add_argument("-w", "--window-size", type=int, default=25, help="Sliding local window size (odd integer)")
    parser.add_argument("-k", "--k", type=float, default=0.2, help="Sauvola parameter k (typically 0.1 to 0.5)")
    
    args = parser.parse_args()

    input_path = args.opt_input or args.pos_input
    output_path = args.opt_output or args.pos_output

    if not input_path or not output_path:
        parser.error("Both input and output paths are required (via flags or positional arguments).")

    return input_path, output_path, args.window_size, args.k


if __name__ == "__main__":
    inp, out, w_size, k_val = parse_args()
    preprocess_document(inp, out, window_size=w_size, k=k_val)
