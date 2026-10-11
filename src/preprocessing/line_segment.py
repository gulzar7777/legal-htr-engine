from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np


def extract_line_crops(
    binary_image: np.ndarray,
    min_line_height: int = 20,
    padding: int = 5
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segments a binarized document page into individual text-line crops using
    Horizontal Projection Profiling (HPP).
    
    Returns a list of tuples: (cropped_line_image, (x, y, w, h))
    """
    # Foreground text is black (0), background is white (255)
    # Invert so text pixels have positive weight
    inverted = 255 - binary_image
    
    # Compute horizontal projection profile (sum across rows)
    proj = np.sum(inverted, axis=1) / 255.0

    # Threshold row density to detect text presence
    threshold = np.mean(proj) * 0.15
    in_line = False
    start_y = 0
    line_slices = []

    for y, count in enumerate(proj):
        if not in_line and count > threshold:
            in_line = True
            start_y = y
        elif in_line and count <= threshold:
            in_line = False
            if (y - start_y) >= min_line_height:
                line_slices.append((start_y, y))

    if in_line and (len(proj) - start_y) >= min_line_height:
        line_slices.append((start_y, len(proj)))

    h_img, w_img = binary_image.shape[:2]
    line_crops = []

    for y1, y2 in line_slices:
        y1_pad = max(0, y1 - padding)
        y2_pad = min(h_img, y2 + padding)
        crop = binary_image[y1_pad:y2_pad, 0:w_img]
        line_crops.append((crop, (0, y1_pad, w_img, y2_pad - y1_pad)))

    return line_crops


def process_and_save_lines(processed_image_path: Path, output_dir: Path) -> int:
    """
    Reads a processed binary document, segments lines, and saves them to output_dir.
    """
    image = cv2.imread(str(processed_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {processed_image_path}")

    crops = extract_line_crops(image)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (crop, bbox) in enumerate(crops):
        out_file = output_dir / f"line_{idx:03d}.png"
        cv2.imwrite(str(out_file), crop)

    return len(crops)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python line_segment.py <input_cleaned_image> <output_crops_dir>")
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    count = process_and_save_lines(inp, out)
    print(f"Extracted {count} text lines into: {out}")
