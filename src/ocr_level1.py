import pytesseract
import cv2
from PIL import Image
import os
from difflib import SequenceMatcher

IMAGE_PATH = r"C:\Users\tassili\Documents\checkpoint.1.4\data\my-image.jpg"
GROUND_TRUTH = "BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS"

#  Improved word accuracy (not position-based)
def word_accuracy(reference, hypothesis):
    ref = reference.upper().split()
    hyp = hypothesis.upper().split()

    correct = 0
    for word in ref:
        if word in hyp:
            correct += 1

    return correct / len(ref) * 100

#  Similarity (more flexible metric)
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio() * 100

#  OCR with clean formatting (remove line breaks/noise)
def ocr(img):
    text = pytesseract.image_to_string(Image.fromarray(img))
    return " ".join(text.split())

# load image
img = cv2.imread(IMAGE_PATH)

# --- baseline (no preprocessing) ---
baseline = pytesseract.image_to_string(Image.open(IMAGE_PATH)).strip()
baseline = " ".join(baseline.split())

print(f"Baseline: {baseline}")
print(f"Accuracy: {word_accuracy(GROUND_TRUTH, baseline):.1f}%")
print(f"Similarity: {similarity(GROUND_TRUTH, baseline):.1f}%\n")

# --- grayscale ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_text = ocr(gray)

print(f"Grayscale: {gray_text}")
print(f"Accuracy: {word_accuracy(GROUND_TRUTH, gray_text):.1f}%")
print(f"Similarity: {similarity(GROUND_TRUTH, gray_text):.1f}%\n")

# --- threshold (binarization) ---
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_text = ocr(thresh)

print(f"Threshold: {thresh_text}")
print(f"Accuracy: {word_accuracy(GROUND_TRUTH, thresh_text):.1f}%")
print(f"Similarity: {similarity(GROUND_TRUTH, thresh_text):.1f}%\n")

# --- resize x2 ---
resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
resized_text = ocr(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))

print(f"Resize x2: {resized_text}")
print(f"Accuracy: {word_accuracy(GROUND_TRUTH, resized_text):.1f}%")
print(f"Similarity: {similarity(GROUND_TRUTH, resized_text):.1f}%\n")

# --- denoise ---
denoised = cv2.fastNlMeansDenoising(gray, h=10)
denoised_text = ocr(denoised)

print(f"Denoised: {denoised_text}")
print(f"Accuracy: {word_accuracy(GROUND_TRUTH, denoised_text):.1f}%")
print(f"Similarity: {similarity(GROUND_TRUTH, denoised_text):.1f}%\n")

#  create outputs folder automatically
os.makedirs("outputs", exist_ok=True)

#  save results
with open("outputs/ocr_level1_output.txt", "w") as f:
    f.write(f"Baseline:  {baseline}\n")
    f.write(f"Grayscale: {gray_text}\n")
    f.write(f"Threshold: {thresh_text}\n")
    f.write(f"Resize x2: {resized_text}\n")
    f.write(f"Denoised:  {denoised_text}\n")

print("saved to outputs/ocr_level1_output.txt")