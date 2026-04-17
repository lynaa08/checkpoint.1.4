import pytesseract
import cv2
from PIL import Image
import os
from difflib import SequenceMatcher

# ---------------- PATH ----------------
IMAGE_FOLDER = r"C:\Users\tassili\Documents\checkpoint.1.4\data\images"

# ---------------- GROUND TRUTH ----------------
GROUND_TRUTH = {
    "image-1.jpg": "BE STRONGER THAN THE CHALLENGE",
    "image-2.jpg": "IT'S OK TO TAKE A BREAK",
    "image-3.jpg": (
        "LUKE 19:11 Jesus predicts his death a third time. "
        "We are going up to Jerusalem and everything written by the prophets about the Son of Man will be fulfilled. "
        "He will be handed over to the Gentiles. They will mock him insult him and spit on him; they will flog him and kill him. "
        "On the third day he will rise again. "
        "A blind beggar receives his sight. "
        "Jesus Son of David have mercy on me. Lord I want to see. "
        "Receive your sight your faith has healed you."
    ),
    "my-image.jpg": "BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS"
}

# ---------------- METRICS ----------------
def word_accuracy(reference, hypothesis):
    ref_words = reference.upper().split()
    hyp_words = hypothesis.upper().split()

    if len(ref_words) == 0:
        return 0

    matches = sum(1 for w in ref_words if w in hyp_words)
    return (matches / len(ref_words)) * 100


def similarity(a, b):
    return SequenceMatcher(None, a.upper(), b.upper()).ratio() * 100


# ---------------- OCR ----------------
def ocr(img):
    text = pytesseract.image_to_string(
        img,
        config="--psm 6 --oem 3"
    )
    return " ".join(text.split())


# ---------------- MAIN ----------------
os.makedirs("outputs", exist_ok=True)

for file in os.listdir(IMAGE_FOLDER):
    path = os.path.join(IMAGE_FOLDER, file)

    img = cv2.imread(path)
    if img is None:
        continue

    print("\n==============================")
    print("IMAGE:", file)

    # -------- Ground truth fix (handles jpg/png mismatch) --------
    base = os.path.splitext(file)[0]

    truth = None
    for k in GROUND_TRUTH:
        if os.path.splitext(k)[0] == base:
            truth = GROUND_TRUTH[k]
            break

    if truth is None:
        print("No ground truth available\n")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------- BASELINE ----------------
    baseline = pytesseract.image_to_string(Image.open(path)).strip()
    baseline = " ".join(baseline.split())

    print(f"Baseline: {baseline}")
    print(f"Accuracy: {word_accuracy(truth, baseline):.1f}%")
    print(f"Similarity: {similarity(truth, baseline):.1f}%\n")

    # ---------------- GRAYSCALE ----------------
    gray_text = ocr(gray)
    print(f"Grayscale: {gray_text}")
    print(f"Accuracy: {word_accuracy(truth, gray_text):.1f}%")
    print(f"Similarity: {similarity(truth, gray_text):.1f}%\n")

    # ---------------- THRESHOLD ----------------
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_text = ocr(thresh)
    print(f"Threshold: {thresh_text}")
    print(f"Accuracy: {word_accuracy(truth, thresh_text):.1f}%")
    print(f"Similarity: {similarity(truth, thresh_text):.1f}%\n")

    # ---------------- RESIZE ----------------
    resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    resized_text = ocr(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))

    print(f"Resize x2: {resized_text}")
    print(f"Accuracy: {word_accuracy(truth, resized_text):.1f}%")
    print(f"Similarity: {similarity(truth, resized_text):.1f}%\n")

    # ---------------- DENOISE ----------------
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    denoised_text = ocr(denoised)

    print(f"Denoised: {denoised_text}")
    print(f"Accuracy: {word_accuracy(truth, denoised_text):.1f}%")
    print(f"Similarity: {similarity(truth, denoised_text):.1f}%\n")

    # ---------------- SAVE ----------------
    with open("outputs/ocr_level1_output.txt", "a", encoding="utf-8") as f:
        f.write("\n==============================\n")
        f.write(f"IMAGE: {file}\n")
        f.write(f"Baseline: {baseline}\n")
        f.write(f"Grayscale: {gray_text}\n")
        f.write(f"Threshold: {thresh_text}\n")
        f.write(f"Resize x2: {resized_text}\n")
        f.write(f"Denoised: {denoised_text}\n")

print("\nSaved to outputs/ocr_level1_output.txt")