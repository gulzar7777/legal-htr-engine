import argparse
from pathlib import Path
import cv2
import numpy as np


def horizontal_projection(binary_img: np.ndarray) -> np.ndarray:
    """
    Computes horizontal projection profile of foreground pixels.
    Assumes background is 255 (white) and ink is 0 (black).
    """
    # Invert so ink = 1, background = 0
    inv = (binary_img == 0).astype(np.uint8)
    return np.sum(inv, axis=1)


def smooth_profile(profile: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooths 1D projection array to eliminate noise from ascenders/descenders."""
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(profile, kernel, mode="same")


def extract_line_boundaries(
    profile: np.ndarray,
    threshold_ratio: float = 0.15,
    min_line_height: int = 18,
    padding: int = 4,
    img_height: int = 0
):
    """
    Finds upper and lower line cut-points based on profile valleys.
    """
    # Dynamic valley threshold based on mean foreground density
    threshold = np.mean(profile) * threshold_ratio
    in_line = False
    start_y = 0
    bounds = []

    for y, count in enumerate(profile):
        if count > threshold and not in_line:
            in_line = True
            start_y = y
        elif count <= threshold and in_line:
            in_line = False
            end_y = y
            if (end_y - start_y) >= min_line_height:
                # Add vertical safety margin without clipping past bounds
                y1 = max(0, start_y - padding)
                y2 = min(img_height, end_y + padding) if img_height > 0 else (end_y + padding)
                bounds.append((y1, y2))

    # Catch line extending to bottom edge
    if in_line:
        end_y = len(profile)
        if (end_y - start_y) >= min_line_height:
            y1 = max(0, start_y - padding)
            y2 = min(img_height, end_y + padding) if img_height > 0 else end_y
            bounds.append((y1, y2))

    return bounds


def segment_document_lines(
    image_path: str,
    output_dir: str,
    threshold_ratio: float = 0.15,
    min_height: int = 18,
    padding: int = 4
) -> list:
    inp = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(inp), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at: {image_path}")

    # Ensure binary input (0 and 255)
    if len(np.unique(img)) > 2:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    h, w = img.shape
    profile = horizontal_projection(img)
    smoothed = smooth_profile(profile, kernel_size=7)
    bounds = extract_line_boundaries(
        smoothed,
        threshold_ratio=threshold_ratio,
        min_line_height=min_height,
        padding=padding,
        img_height=h
    )

    saved_paths = []
    for idx, (y1, y2) in enumerate(bounds):
        line_crop = img[y1:y2, 0:w]
        crop_name = out_dir / f"line_{idx:03d}.png"
        cv2.imwrite(str(crop_name), line_crop)
        saved_paths.append(str(crop_name))

    print(f"[SUCCESS] Segmented {len(saved_paths)} lines into: {out_dir}")
    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Horizontal Projection Profiling (HPP) Line Segmenter.")
    parser.add_argument("pos_input", nargs="?", default=None, help="Input binary image path")
    parser.add_argument("pos_output_dir", nargs="?", default=None, help="Output directory for line crops")
    parser.add_argument("-i", "--input", dest="opt_input", help="Input binary image path")
    parser.add_argument("-o", "--output-dir", dest="opt_output", help="Output line crops directory")
    parser.add_argument("--threshold-ratio", type=float, default=0.15, help="Valley cut-point ratio relative to mean")
    parser.add_argument("--min-height", type=int, default=18, help="Minimum line height in pixels")
    parser.add_argument("--padding", type=int, default=4, help="Vertical boundary padding in pixels")
    args = parser.parse_args()

    inp = args.opt_input or args.pos_input
    out = args.opt_output or args.pos_output_dir

    if not inp or not out:
        parser.error("Both input image and output directory must be specified.")

    return inp, out, args.threshold_ratio, args.min_height, args.padding


if __name__ == "__main__":
    inp, out, thresh, m_h, pad = parse_args()
    segment_document_lines(inp, out, threshold_ratio=thresh, min_height=m_h, padding=pad)
