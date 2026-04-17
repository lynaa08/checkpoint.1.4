from gtts import gTTS

with open("outputs/ocr_level1_output.txt", "r") as f:
    text = f.read()

tts = gTTS(text)
tts.save("outputs/tts_output.mp3")

print("saved to outputs/tts_output.mp3")