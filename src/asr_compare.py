import whisper
from jiwer import wer
import re

AUDIO_PATH = r"C:\Users\tassili\Documents\checkpoint.1.4\data\voice-message.wav"
GROUND_TRUTH = "hi lena how are you i hope you are fine i want to tell you lets start the project"

def clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text) 
    return text

# Whisper
model = whisper.load_model("base")
whisper_text = model.transcribe(AUDIO_PATH)["text"].strip()
whisper_wer = wer(clean(GROUND_TRUTH), clean(whisper_text))
print(f"Whisper: {whisper_text}")
print(f"Whisper WER: {whisper_wer*100:.1f}%  |  Accuracy: {(1-whisper_wer)*100:.1f}%")