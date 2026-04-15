import pytesseract
import easyocr
from PIL import Image

IMAGE_PATH = r"C:\Users\tassili\Documents\checkpoint.1.4\data\my-image.jpg"
GROUND_TRUTH = "BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS"

def word_accuracy(reference, hypothesis):
    ref = reference.upper().split()
    hyp = hypothesis.upper().split()
    matches = sum(w1 == w2 for w1, w2 in zip(ref, hyp))
    return matches / len(ref) * 100

# Tesseract
image = Image.open(IMAGE_PATH)
tesseract_text = pytesseract.image_to_string(image).strip()
print(f"Tesseract: {tesseract_text}")
print(f"Tesseract Word Accuracy: {word_accuracy(GROUND_TRUTH, tesseract_text):.1f}%\n")

# EasyOCR (alternative library - step 4)
reader = easyocr.Reader(['en'], gpu=False)
easyocr_text = " ".join(reader.readtext(IMAGE_PATH, detail=0))
print(f"EasyOCR: {easyocr_text}")
print(f"EasyOCR Word Accuracy: {word_accuracy(GROUND_TRUTH, easyocr_text):.1f}%")
