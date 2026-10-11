from pathlib import Path
import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detects skew angle using minimum area bounding rectangle on foreground text
    and applies an affine rotation to straighten the document.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert and threshold to isolate text contours
    thresh = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(thresh > 0))
    
    if len(coords) < 50:
        return image  # Insufficient text points to reliably calculate angle

    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust OpenCV angle conventions (-90 to 0 degrees)
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate with border replication or white padding
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )
    return rotated


def sauvola_binarization(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0
) -> np.ndarray:
    """
    Applies Sauvola adaptive thresholding:
    T(x, y) = m(x, y) * (1 + k * ((s(x, y) / r) - 1))
    Optimized for degraded parchment, faint ink, and bleed-through.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Calculate local Sauvola threshold matrix
    thresh_sauvola = threshold_sauvola(gray, window_size=window_size, k=k, r=r)
    binary = (gray > thresh_sauvola).astype(np.uint8) * 255

    return binary


def preprocess_document(image_path: Path, output_path: Path = None) -> np.ndarray:
    """
    Full preprocessing pipeline: Load -> Deskew -> Sauvola Binarization -> Denoise
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not open image at {image_path}")

    # 1. Deskew
    deskewed = deskew_image(img)

    # 2. Binarize using Sauvola
    binary = sauvola_binarization(deskewed, window_size=25, k=0.2)

    # 3. Morphological median filter to remove salt-and-pepper scan noise
    denoised = cv2.medianBlur(binary, 3)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), denoised)

    return denoised


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python binarize.py <input_image_path> <output_image_path>")
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    preprocess_document(inp, out)
    print(f"Processed image written to: {out}")
