import argparse
from pathlib import Path
import cv2
import numpy as np


def horizontal_projection(binary_img: np.ndarray) -> np.ndarray:
    inv = (binary_img == 0).astype(np.uint8)
    return np.sum(inv, axis=1)


def smooth_profile(profile: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(profile, kernel, mode="same")


def compute_seam_energy(binary_img: np.ndarray) -> np.ndarray:
    """
    Computes energy map for seam carving.
    Ink pixels get maximum cost, while whitespace far from ink gets minimal cost.
    """
    inv = (binary_img == 0).astype(np.uint8)
    dist_transform = cv2.distanceTransform(1 - inv, cv2.DIST_L2, 5)
    max_dist = np.max(dist_transform) if np.max(dist_transform) > 0 else 1.0
    normalized_dist = dist_transform / max_dist

    # Energy is high on foreground ink and low inside valleys
    energy = (inv.astype(np.float32) * 255.0) + (1.0 - normalized_dist) * 50.0
    return energy


def find_horizontal_seam(energy_map: np.ndarray, y_start: int, y_end: int) -> np.ndarray:
    """
    Finds minimum energy path across image columns between y_start and y_end using DP.
    """
    h, w = energy_map.shape
    y1 = max(0, y_start)
    y2 = min(h, y_end)

    sub_energy = energy_map[y1:y2, :]
    sub_h, sub_w = sub_energy.shape

    if sub_h <= 1:
        return np.full(w, y1, dtype=np.int32)

    cost = np.copy(sub_energy)
    backtrack = np.zeros((sub_h, sub_w), dtype=np.int32)

    for x in range(1, sub_w):
        for y in range(sub_h):
            prev_y = [y]
            if y > 0:
                prev_y.append(y - 1)
            if y < sub_h - 1:
                prev_y.append(y + 1)

            best_idx = prev_y[np.argmin(cost[prev_y, x - 1])]
            backtrack[y, x] = best_idx
            cost[y, x] += cost[best_idx, x - 1]

    # Trace back seam path from minimum end-point
    seam = np.zeros(sub_w, dtype=np.int32)
    seam[-1] = np.argmin(cost[:, -1])

    for x in range(sub_w - 2, -1, -1):
        seam[x] = backtrack[seam[x + 1], x + 1]

    return seam + y1


def extract_line_boundaries_with_seams(
    binary_img: np.ndarray,
    threshold_ratio: float = 0.15,
    min_line_height: int = 18,
    overlap_threshold: float = 0.35,
    padding: int = 4
) -> list:
    """
    HPP baseline segmentation with dynamic programming seam carving fallback
    for touching ascenders, descenders, and curved baselines.
    """
    h, w = binary_img.shape
    profile = horizontal_projection(binary_img)
    smoothed = smooth_profile(profile, kernel_size=7)

    mean_val = np.mean(smoothed)
    valley_threshold = mean_val * threshold_ratio
    dense_overlap_limit = mean_val * overlap_threshold

    in_line = False
    start_y = 0
    raw_regions = []

    for y, count in enumerate(smoothed):
        if count > valley_threshold and not in_line:
            in_line = True
            start_y = y
        elif count <= valley_threshold and in_line:
            in_line = False
            if (y - start_y) >= min_line_height:
                raw_regions.append((start_y, y))

    if in_line and (h - start_y) >= min_line_height:
        raw_regions.append((start_y, h))

    if not raw_regions:
        return []

    energy_map = compute_seam_energy(binary_img)
    seams = []

    # Top boundary seam
    top_window = max(5, raw_regions[0][0])
    top_seam = find_horizontal_seam(energy_map, 0, top_window)
    seams.append(top_seam)

    # Inter-line seams
    for idx in range(len(raw_regions) - 1):
        r1_end = raw_regions[idx][1]
        r2_start = raw_regions[idx + 1][0]

        # Tightly packed or touching lines fallback condition
        if r2_start <= r1_end or (r2_start - r1_end) < 8:
            search_center = (r1_end + r2_start) // 2
            search_y1 = max(0, search_center - 16)
            search_y2 = min(h, search_center + 16)
            inter_seam = find_horizontal_seam(energy_map, search_y1, search_y2)
        else:
            inter_seam = find_horizontal_seam(energy_map, r1_end, r2_start)

        seams.append(inter_seam)

    # Bottom boundary seam
    bot_start = min(h - 5, raw_regions[-1][1])
    bot_seam = find_horizontal_seam(energy_map, bot_start, h)
    seams.append(bot_seam)

    # Extract masked line strips between consecutive seams
    line_crops = []
    for idx in range(len(seams) - 1):
        upper_seam = seams[idx]
        lower_seam = seams[idx + 1]

        y_min = max(0, int(np.min(upper_seam)) - padding)
        y_max = min(h, int(np.max(lower_seam)) + padding)

        if (y_max - y_min) < min_line_height:
            continue

        crop = np.ones((y_max - y_min, w), dtype=np.uint8) * 255
        raw_strip = binary_img[y_min:y_max, :]

        for col in range(w):
            u_bound = max(0, upper_seam[col] - y_min)
            l_bound = min(y_max - y_min, lower_seam[col] - y_min)
            if l_bound > u_bound:
                crop[u_bound:l_bound, col] = raw_strip[u_bound:l_bound, col]

        line_crops.append(crop)

    return line_crops


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

    if len(np.unique(img)) > 2:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    crops = extract_line_boundaries_with_seams(
        img,
        threshold_ratio=threshold_ratio,
        min_line_height=min_height,
        padding=padding
    )

    saved_paths = []
    for idx, crop in enumerate(crops):
        crop_name = out_dir / f"line_{idx:03d}.png"
        cv2.imwrite(str(crop_name), crop)
        saved_paths.append(str(crop_name))

    print(f"[SUCCESS] Segmented {len(saved_paths)} lines using seam-carving into: {out_dir}")
    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(description="HPP and Seam-Carving Fallback Line Segmenter.")
    parser.add_argument("pos_input", nargs="?", default=None, help="Input binary image path")
    parser.add_argument("pos_output_dir", nargs="?", default=None, help="Output directory")
    parser.add_argument("-i", "--input", dest="opt_input", help="Input binary image path")
    parser.add_argument("-o", "--output-dir", dest="opt_output", help="Output directory")
    parser.add_argument("--threshold-ratio", type=float, default=0.15, help="Valley cut-point ratio")
    parser.add_argument("--min-height", type=int, default=18, help="Minimum line height in pixels")
    parser.add_argument("--padding", type=int, default=4, help="Vertical padding")
    args = parser.parse_args()

    inp = args.opt_input or args.pos_input
    out = args.opt_output or args.pos_output_dir

    if not inp or not out:
        parser.error("Both input image and output directory must be specified.")

    return inp, out, args.threshold_ratio, args.min_height, args.padding


if __name__ == "__main__":
    inp, out, thresh, m_h, pad = parse_args()
    segment_document_lines(inp, out, threshold_ratio=thresh, min_height=m_h, padding=pad)
