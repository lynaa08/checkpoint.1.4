import pytesseract
from PIL import Image

img = Image.open(r"C:\Users\tassili\Documents\checkpoint.1.4\data\image.png")
text = pytesseract.image_to_string(img)

print(text)