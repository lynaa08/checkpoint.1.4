import pyttsx3

# reads the combined output and speaks it out loud
with open("outputs/ocr_level1_output.txt", "r") as f:
    text = f.read()

engine = pyttsx3.init()
engine.setProperty("rate", 150)   # speed (words per minute)
engine.setProperty("volume", 1.0)

print("Speaking the OCR output...\n")
print(text)

engine.say(text)
engine.save_to_file(text, "outputs/tts_output.mp3")
engine.runAndWait()

print("\nsaved to outputs/tts_output.mp3")