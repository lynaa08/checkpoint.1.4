"""
ocr_level2.py  —  Smart Classroom Assistant
OCR module: extract text from lecture slide images using Tesseract + OpenCV preprocessing.
"""

import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from difflib import SequenceMatcher


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "images")
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "..", "outputs", "ocr_output.txt")

# Optional ground truth for accuracy metrics 
GROUND_TRUTH = {
    "image-1.png": "BE STRONGER THAN THE CHALLENGE",
    "image-2.jpg": "IT'S OK TO TAKE A BREAK",
    "image-3.jpg": (
        "LUKE 19:11 Jesus predicts his death a third time. "
        "We are going up to Jerusalem and everything written by the prophets about the Son of Man will be fulfilled."
    ),
    "my-image.jpg": "BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS",
}
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Apply the full preprocessing pipeline to a BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if w < 1000:
        scale = 1000 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.fastNlMeansDenoising(gray, h=10)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    binary = deskew(binary)

    return binary


def deskew(img: np.ndarray) -> np.ndarray:
    """Straighten a slightly rotated image using moment-based deskewing."""
    coords = np.column_stack(np.where(img < 127))   # dark pixels
    if coords.size == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    # Only correct small skews (< 10 degrees) to avoid overcorrection
    if abs(angle) > 10:
        return img
    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def word_accuracy(reference: str, hypothesis: str) -> float:
    ref_words = reference.upper().split()
    hyp_words = hypothesis.upper().split()
    if not ref_words:
        return 0.0
    matches = sum(1 for w in ref_words if w in hyp_words)
    return (matches / len(ref_words)) * 100


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.upper(), b.upper()).ratio() * 100


# ─────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────
def run_ocr(image: np.ndarray) -> str:
    """Run Tesseract on a preprocessed (numpy) image and return clean text."""
    pil_img = Image.fromarray(image)
    raw = pytesseract.image_to_string(pil_img, config="--psm 6 --oem 3")
    return " ".join(raw.split())   # collapse whitespace


def extract_text_from_file(path: str) -> dict:
    """
    Full pipeline for one image file.
    Returns a dict with 'file', 'raw_text', 'processed_text', 'metrics'.
    """
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return {"file": os.path.basename(path), "error": "Could not read image"}

    filename = os.path.basename(path)

    # Baseline (no preprocessing)
    raw_text = run_ocr(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    # Preprocessed
    processed = preprocess(img_bgr)
    processed_text = run_ocr(processed)

    result = {
        "file": filename,
        "raw_text": raw_text,
        "processed_text": processed_text,
        "metrics": {},
    }

    # Compute metrics if ground truth exists
    truth_key = None
    for k in GROUND_TRUTH:
        if os.path.splitext(k)[0] == os.path.splitext(filename)[0]:
            truth_key = k
            break

    if truth_key:
        gt = GROUND_TRUTH[truth_key]
        result["metrics"] = {
            "baseline_word_accuracy":    round(word_accuracy(gt, raw_text), 1),
            "processed_word_accuracy":   round(word_accuracy(gt, processed_text), 1),
            "baseline_char_similarity":  round(char_similarity(gt, raw_text), 1),
            "processed_char_similarity": round(char_similarity(gt, processed_text), 1),
        }

    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_all(image_folder: str = IMAGE_FOLDER) -> list[dict]:
    """Process every image in the folder and return results."""
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    files = [
        f for f in sorted(os.listdir(image_folder))
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not files:
        print(f"[OCR] No images found in {image_folder}")
        return []

    results = []
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("=== OCR OUTPUT — Smart Classroom Assistant ===\n\n")

        for fname in files:
            path = os.path.join(image_folder, fname)
            print(f"[OCR] Processing: {fname}")
            r = extract_text_from_file(path)
            results.append(r)

            # Console
            print(f"  Raw      : {r.get('raw_text', '')[:80]}")
            print(f"  Processed: {r.get('processed_text', '')[:80]}")
            if r.get("metrics"):
                m = r["metrics"]
                print(f"  Baseline  accuracy={m['baseline_word_accuracy']}%  similarity={m['baseline_char_similarity']}%")
                print(f"  Processed accuracy={m['processed_word_accuracy']}%  similarity={m['processed_char_similarity']}%")
            print()

            # File
            out.write(f"--- {fname} ---\n")
            out.write(f"Raw text      : {r.get('raw_text', '')}\n")
            out.write(f"Processed text: {r.get('processed_text', '')}\n")
            if r.get("metrics"):
                out.write(f"Metrics       : {r['metrics']}\n")
            out.write("\n")

    print(f"[OCR] Saved → {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    run_all()
