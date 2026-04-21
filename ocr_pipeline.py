"""
ocr_pipeline.py - Smart License Plate Detection
Phase B: OCR Integration

Provides functions to:
  - Pre-process a cropped license plate image
  - Run EasyOCR to extract the plate number text
  - Clean and return the plate string

Usage (standalone test):
    python ocr_pipeline.py --image path/to/plate_crop.jpg
"""

import argparse
import re
import cv2
import numpy as np

# Lazy-load EasyOCR to avoid slow startup when imported as module
_reader = None


def _get_reader():
    """Lazy-load the EasyOCR reader (English only)."""
    global _reader
    if _reader is None:
        import easyocr
        print("[OCR] Loading EasyOCR model (first load may take a moment) ...")
        _reader = easyocr.Reader(["en"], gpu=_gpu_available())
    return _reader


def _gpu_available():
    """Check whether a CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# --- Image Pre-processing ----------------------------------------------------

def _upscale(img_gray: np.ndarray, target_h: int = 300) -> np.ndarray:
    """
    Aggressively upscale a small grayscale image to a fixed height.
    Uses INTER_LANCZOS4 for best quality on tiny sources.
    Adds white padding (20px each side) so edge chars aren't clipped.
    """
    h, w = img_gray.shape[:2]
    scale = target_h / max(h, 1)
    new_w = max(int(w * scale), 1)
    upscaled = cv2.resize(img_gray, (new_w, target_h),
                          interpolation=cv2.INTER_LANCZOS4)
    # Add padding so border characters are not clipped by EasyOCR
    padded = cv2.copyMakeBorder(upscaled, 20, 20, 20, 20,
                                cv2.BORDER_CONSTANT, value=255)
    return padded


def _sharpen(img: np.ndarray) -> np.ndarray:
    """Apply an unsharp-mask to enhance character edges."""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharp = cv2.addWeighted(img, 1.8, blurred, -0.8, 0)
    return sharp


def _variant_clahe_thresh(gray: np.ndarray) -> np.ndarray:
    """CLAHE + adaptive threshold — good for uneven lighting."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 8
    )
    return thresh


def _variant_otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu global threshold — works well when contrast is high."""
    denoised = cv2.fastNlMeansDenoising(gray, h=15)
    _, thresh = cv2.threshold(denoised, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _variant_morph(gray: np.ndarray) -> np.ndarray:
    """Morphological top-hat + Otsu — recovers text on dark backgrounds."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, thresh = cv2.threshold(tophat, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def preprocess_plate(crop_bgr: np.ndarray) -> list:
    """
    Pre-process a BGR license plate crop for OCR.
    Returns a list of candidate preprocessed images (variants).
    Each variant targets a different lighting / contrast scenario.
    The caller will run OCR on all variants and pick the best result.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Empty image passed to preprocess_plate()")

    # Convert to grayscale
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Aggressively upscale (tiny plates need a lot of scaling)
    up = _upscale(gray, target_h=300)

    # Sharpen the upscaled image before thresholding
    sharpened = _sharpen(up)

    variants = [
        _variant_clahe_thresh(sharpened),   # CLAHE + adaptive
        _variant_otsu(sharpened),            # Otsu global
        _variant_morph(sharpened),           # morphological top-hat
        sharpened,                           # plain sharpened (no threshold)
    ]
    return variants


# --- Text Cleaning -----------------------------------------------------------

def clean_plate_text(raw_text: str) -> str:
    """
    Clean raw OCR output to extract only alphanumeric plate characters.
    Removes spaces, special characters, and obvious OCR noise.
    """
    # Common OCR misreads for license plates
    substitutions = {
        "O": "0",   # letter O -> zero  (context: all-digit section)
        # Keep both in place; let context decide
    }
    text = raw_text.upper()

    # Keep only alphanumeric characters and hyphens
    cleaned = re.sub(r"[^A-Z0-9\-]", "", text)

    # Remove sequences that are clearly too short to be a plate
    if len(cleaned) < 2:
        return ""
    return cleaned.strip()


# --- OCR on multiple pre-process variants ------------------------------------

def _run_ocr_on_variants(variants: list, reader, min_confidence: float,
                          allowlist: str) -> dict:
    """
    Run EasyOCR on each variant image, return the result with highest
    average confidence above min_confidence.
    """
    best = {"text": "", "raw_text": "", "confidence": 0.0, "all_results": []}

    for img in variants:
        try:
            results = reader.readtext(
                img,
                detail=1,
                paragraph=False,
                allowlist=allowlist,
                # Tuned for small license plates
                width_ths=0.7,
                contrast_ths=0.1,
                adjust_contrast=0.7,
                low_text=0.3,
                text_threshold=0.5,
                link_threshold=0.3,
            )
        except Exception as e:
            continue

        filtered = [(text, conf) for (_, text, conf) in results
                    if conf >= min_confidence]
        if not filtered:
            continue

        avg_conf = sum(c for _, c in filtered) / len(filtered)
        if avg_conf > best["confidence"]:
            raw_text = " ".join(t for t, _ in filtered)
            best = {
                "text":        clean_plate_text(raw_text),
                "raw_text":    raw_text,
                "confidence":  round(avg_conf, 4),
                "all_results": results,
            }

    return best


# --- Main OCR function -------------------------------------------------------

# Characters valid on a license plate (alphanumeric + hyphen)
_PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"


def extract_plate_text(crop_bgr: np.ndarray,
                       preprocess: bool = True,
                       min_confidence: float = 0.2) -> dict:
    """
    Extract license plate text from a BGR crop image.

    Args:
        crop_bgr:        BGR numpy array of the cropped plate region
        preprocess:      Whether to apply image preprocessing (recommended: True)
        min_confidence:  Minimum OCR confidence to accept a reading

    Returns:
        dict with keys:
            'text'        - cleaned plate string (e.g. "MH12AB1234")
            'raw_text'    - raw concatenated OCR output
            'confidence'  - average OCR confidence (0-1)
            'all_results' - list of (bbox, text, confidence) from EasyOCR
    """
    reader = _get_reader()

    if preprocess:
        variants = preprocess_plate(crop_bgr)
    else:
        # Even without full preprocessing, upscale for EasyOCR
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        variants = [_upscale(gray, target_h=300)]

    return _run_ocr_on_variants(variants, reader, min_confidence,
                                allowlist=_PLATE_ALLOWLIST)


# --- Standalone test ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test OCR pipeline on a single plate crop.")
    parser.add_argument("--image", required=True,
                        help="Path to a license plate crop image")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Disable image preprocessing")
    parser.add_argument("--min-conf", type=float, default=0.2,
                        help="Minimum OCR confidence threshold (default: 0.2)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Could not load image: {args.image}")
        return

    h, w = img.shape[:2]
    print(f"[INFO] Input size : {w}x{h}px")
    print(f"[INFO] Running OCR on: {args.image}")

    result = extract_plate_text(img,
                                preprocess=not args.no_preprocess,
                                min_confidence=args.min_conf)

    print("\n-- OCR Result ------------------------------------------")
    print(f"  Plate Text  : {result['text']}")
    print(f"  Raw Text    : {result['raw_text']}")
    print(f"  Confidence  : {result['confidence']:.2%}")
    if result["all_results"]:
        print("  All tokens  :")
        for (_, text, conf) in result["all_results"]:
            print(f"    '{text}'  ({conf:.2%})")
    print("--------------------------------------------------------")


if __name__ == "__main__":
    main()
