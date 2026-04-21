"""
ocr_pipeline.py — Smart License Plate Detection
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
        print("[OCR] Loading EasyOCR model (first load may take a moment) …")
        _reader = easyocr.Reader(["en"], gpu=_gpu_available())
    return _reader


def _gpu_available():
    """Check whether a CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─── Image Pre-processing ─────────────────────────────────────────────────────

def preprocess_plate(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Pre-process a BGR license plate crop for OCR.
    Steps:  resize → grayscale → denoise → CLAHE → threshold
    Returns a processed grayscale image.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Empty image passed to preprocess_plate()")

    # 1. Resize to a fixed height while keeping aspect ratio
    target_h = 80
    h, w = crop_bgr.shape[:2]
    scale  = target_h / h
    new_w  = max(int(w * scale), 1)
    resized = cv2.resize(crop_bgr, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 4. CLAHE (contrast enhancement)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(denoised)

    # 5. Adaptive threshold (handles uneven lighting)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )

    return thresh


# ─── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_plate_text(raw_text: str) -> str:
    """
    Clean raw OCR output to extract only alphanumeric plate characters.
    Removes spaces, special characters, and obvious OCR noise.
    """
    # Keep only alphanumeric characters and hyphens
    cleaned = re.sub(r"[^A-Z0-9\-]", "", raw_text.upper())
    # Remove sequences that are clearly too short to be a plate
    if len(cleaned) < 3:
        return ""
    return cleaned.strip()


# ─── Main OCR function ────────────────────────────────────────────────────────

def extract_plate_text(crop_bgr: np.ndarray,
                       preprocess: bool = True,
                       min_confidence: float = 0.3) -> dict:
    """
    Extract license plate text from a BGR crop image.

    Args:
        crop_bgr:        BGR numpy array of the cropped plate region
        preprocess:      Whether to apply image preprocessing (recommended: True)
        min_confidence:  Minimum OCR confidence to accept a reading

    Returns:
        dict with keys:
            'text'        — cleaned plate string (e.g. "MH12AB1234")
            'raw_text'    — raw concatenated OCR output
            'confidence'  — average OCR confidence (0-1)
            'all_results' — list of (text, confidence) tuples from EasyOCR
    """
    reader = _get_reader()

    if preprocess:
        processed = preprocess_plate(crop_bgr)
        # EasyOCR accepts grayscale or BGR; pass processed
        ocr_input = processed
    else:
        ocr_input = crop_bgr

    try:
        results = reader.readtext(ocr_input, detail=1, paragraph=False)
    except Exception as e:
        print(f"[OCR] EasyOCR error: {e}")
        return {"text": "", "raw_text": "", "confidence": 0.0, "all_results": []}

    # Filter by minimum confidence
    filtered = [(text, conf) for (_, text, conf) in results if conf >= min_confidence]

    if not filtered:
        return {"text": "", "raw_text": "", "confidence": 0.0, "all_results": results}

    raw_text      = " ".join(t for t, _ in filtered)
    avg_confidence = sum(c for _, c in filtered) / len(filtered)
    cleaned_text  = clean_plate_text(raw_text)

    return {
        "text":        cleaned_text,
        "raw_text":    raw_text,
        "confidence":  round(avg_confidence, 4),
        "all_results": results,
    }


# ─── Standalone test ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test OCR pipeline on a single plate crop.")
    parser.add_argument("--image", required=True, help="Path to a license plate crop image")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Disable image preprocessing")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Could not load image: {args.image}")
        return

    print(f"[INFO] Running OCR on: {args.image}")
    result = extract_plate_text(img, preprocess=not args.no_preprocess)

    print("\n── OCR Result ──────────────────────────────")
    print(f"  Plate Text  : {result['text']}")
    print(f"  Raw Text    : {result['raw_text']}")
    print(f"  Confidence  : {result['confidence']:.2%}")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()
