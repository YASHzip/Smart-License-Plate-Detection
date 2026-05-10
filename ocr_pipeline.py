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


# --- Text Cleaning / Indian Plate Validation ----------------------------------
#
# Supported formats (all spaces/hyphens stripped before matching):
#
#  TYPE              PATTERN (stripped)       EXAMPLE          LENGTH
#  ────────────────  ───────────────────────  ───────────────  ──────
#  Standard (9)      AA0A0000                 DL1A1234          9
#  Standard (10)     AA00AA0000               MH12AB1234        10
#  BH Series         00BH0000AA               22BH1234AB        10
#  Diplomatic        CD/CC/UN + 2-6 digits    CD1234             6-8
#  Military          00A000000A               22B123456A        10
#  IND-HSRP (12)     IND + 9-char standard    INDDL1A1234       12
#  IND-HSRP (13)     IND + 10-char standard   INDMH12AB1234     13
#  IND-BH (13)       IND + BH 10-char         IND22BH1234AB     13

# 1. Standard civilian – 2-letter state + 1-2 digit district
#    + 1-2 letter series + 4 digits  (9 or 10 chars)
_RE_STANDARD = re.compile(
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
)

# 2. BH (Bharat) series – YY BH NNNN LL  (10 chars)
#    Year digits (20-99) + BH + 4 digits + 2 letters (excluding I and O)
_RE_BH = re.compile(
    r'^[2-9][0-9]BH[0-9]{4}[A-HJ-NP-Z]{2}$'
)

# 3. Diplomatic – CD / CC / UN followed by 2-6 digits  (4-8 chars)
_RE_DIPLOMATIC = re.compile(
    r'^(CD|CC|UN)[0-9]{2,6}$'
)

# 4. Military – YY + 1 class letter + 6 serial digits + 1 check letter (10 chars)
_RE_MILITARY = re.compile(
    r'^[0-9]{2}[A-Z][0-9]{6}[A-Z]$'
)

# Plate types for logging / DB tagging
_PLATE_PATTERNS = [
    ("STANDARD",   _RE_STANDARD),
    ("BH",         _RE_BH),
    ("DIPLOMATIC", _RE_DIPLOMATIC),
    ("MILITARY",   _RE_MILITARY),
]


def validate_indian_plate(text: str) -> str:
    """
    Validate cleaned OCR text against ALL official Indian plate formats.

    Accepted formats
    ----------------
    Standard    (9-10 chars) : SS D(D) L(L) NNNN   e.g. MH12AB1234 / DL1A5678
    BH Series      (10 chars): YY BH NNNN LL       e.g. 22BH1234AB
    Diplomatic     (4-8 chars): CD/CC/UN + digits   e.g. CD12345
    Military       (10 chars): YY C NNNNNN X       e.g. 22B123456A
    IND-HSRP prefix (+3 chars): IND + any above     e.g. INDMH12AB1234

    The IND prefix is ALWAYS stripped from the result — only the core
    plate number is returned (e.g. INDMH12AB1234 -> MH12AB1234).

    Returns the validated core plate string (uppercase, no spaces/hyphens),
    or "" if it matches no known Indian plate format.
    """
    # Normalise: uppercase, strip spaces/hyphens
    text = text.upper().replace("-", "").replace(" ", "")

    # Strip optional IND prefix (HSRP / international plates)
    core = text[3:] if text.startswith("IND") else text

    for _name, pattern in _PLATE_PATTERNS:
        if pattern.fullmatch(core):
            return core   # always return without IND prefix

    return ""   # matches no known Indian plate format


# --- Context-aware OCR correction --------------------------------------------
#
# All 36 Indian state / UT registration codes (MoRTH official list)
_STATE_CODES = frozenset({
    # States
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP",
    "JH", "JK", "KA", "KL", "MP", "MH", "MN", "ML", "MZ",
    "NL", "OD", "PB", "RJ", "SK", "TN", "TS", "TR", "UP",
    "UK", "WB",
    # Union Territories
    "AN", "CH", "DD", "DL", "DN", "LA", "LD", "PY",
})

# Characters that look like digits but OCR reads as letters (use in digit slots)
_LETTER_TO_DIGIT = {"O": "0", "I": "1", "S": "5", "B": "8",
                     "G": "6", "Z": "2", "T": "7", "A": "4"}
# Characters that look like letters but OCR reads as digits (use in letter slots)
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B",
                     "6": "G", "2": "Z", "7": "T", "4": "A"}


def _fix_digit(c: str) -> str:
    return _LETTER_TO_DIGIT.get(c, c)


def _fix_letter(c: str) -> str:
    return _DIGIT_TO_LETTER.get(c, c)


def _try_fix_state(s: str) -> str:
    """Try every single-char substitution to produce a valid state code."""
    if s in _STATE_CODES:
        return s
    for i in range(len(s)):
        fixed = list(s)
        fixed[i] = _DIGIT_TO_LETTER.get(fixed[i], fixed[i])
        candidate = "".join(fixed)
        if candidate in _STATE_CODES:
            return candidate
    return s  # best effort


def correct_ocr_errors(text: str) -> str:
    """
    Context-aware correction for Indian plate OCR errors.

    Applies character-position rules:
      Standard (10):  SS  DD  LL  NNNN
                      ^^  ^^  ^^  ^^^^
                      letter digit letter digit

      Standard (9):   SS  D  L  NNNN
      BH series (10): NN BH NNNN LL
      Diplomatic:     CC NNNNNN  (no positional fix — left as-is)
      Military (10):  NN L NNNNNN L

    Returns a corrected string (may still fail final validation).
    """
    n = len(text)

    # --- Standard 10-char: SS DD LL NNNN ---
    if n == 10:
        state    = _try_fix_state(text[0:2])
        district = "".join(_fix_digit(c)  for c in text[2:4])
        series   = "".join(_fix_letter(c) for c in text[4:6])
        number   = "".join(_fix_digit(c)  for c in text[6:10])
        return state + district + series + number

    # --- Standard 9-char: SS D LL NNNN ---
    if n == 9:
        state    = _try_fix_state(text[0:2])
        district = _fix_digit(text[2])
        series   = "".join(_fix_letter(c) for c in text[3:5])
        number   = "".join(_fix_digit(c)  for c in text[5:9])
        return state + district + series + number

    # --- BH 10-char: YY BH NNNN LL ---
    if n == 10 and text[2:4] == "BH":
        year   = "".join(_fix_digit(c) for c in text[0:2])
        number = "".join(_fix_digit(c) for c in text[4:8])
        series = "".join(_fix_letter(c) for c in text[8:10])
        return year + "BH" + number + series

    # --- Military 10-char: YY C NNNNNN X ---
    if n == 10:
        year    = "".join(_fix_digit(c)  for c in text[0:2])
        cls     = _fix_letter(text[2])
        serial  = "".join(_fix_digit(c)  for c in text[3:9])
        check   = _fix_letter(text[9])
        return year + cls + serial + check

    # Diplomatic / other lengths — no positional correction
    return text


def clean_plate_text(raw_text: str) -> str:
    """
    Clean raw OCR output and attempt to validate as a known Indian plate.

    Pipeline:
        1. Strip non-alphanumeric characters
        2. Apply context-aware OCR error correction
        3. Validate against all known Indian plate formats
        4. If validation fails, return the best-effort cleaned text anyway
           (never silently discard a reading the OCR was confident about)

    Returns the validated plate string if it matches a known format,
    otherwise returns the cleaned raw text as-is (never empty if input
    had at least 2 alphanumeric characters).
    """
    text = raw_text.upper()

    # Strip all non-alphanumeric chars (punctuation, OCR noise)
    cleaned = re.sub(r"[^A-Z0-9]", "", text)

    if len(cleaned) < 2:
        return ""

    # Strip IND prefix (HSRP plates) before any matching
    if cleaned.startswith("IND"):
        cleaned = cleaned[3:]

    # Step 1: try as-is (exact match)
    result = validate_indian_plate(cleaned)
    if result:
        return result

    # Step 2: apply position-aware OCR correction and try again
    corrected = correct_ocr_errors(cleaned)
    result = validate_indian_plate(corrected)
    if result:
        return result

    # Step 3: validation failed — return the cleaned text anyway
    # (OCR detected something real; IND prefix already stripped above)
    return cleaned


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
