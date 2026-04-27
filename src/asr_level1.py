import whisper
import librosa
import soundfile as sf
from jiwer import wer
import os
import re
os.environ["PATH"] += os.pathsep + r"C:\Users\HP\Downloads\ffmpeg-tools-2025-01-01-git-d3aa99a4f4\ffmpeg-tools-2025-01-01-git-d3aa99a4f4\bin"
# ---------------- PATH ----------------
AUDIO_FOLDER = r"C:\checkpoint.1.4\data\audio"

# ---------------- GROUND TRUTH ----------------
GROUND_TRUTH = {
    "audio1.wav": "I'm sorry but I cannot fix this problem for you",
    "audio2.wav": "You have been selected",
    "audio3.wav": "You are just a line of code",
    "voice-message.wav": "Hi Lena how are you I hope you are fine I want to tell you lets start the project"
}

# ---------------- CLEAN TEXT ----------------
def clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# ---------------- WER ----------------
def get_wer(reference, hypothesis):
    return wer(clean(reference), clean(hypothesis)) * 100

# ---------------- MODEL ----------------
model = whisper.load_model("base")

os.makedirs("outputs", exist_ok=True)

# ---------------- MAIN LOOP ----------------
for file in os.listdir(AUDIO_FOLDER):
    if not file.endswith(".wav"):
        continue

    path = os.path.join(AUDIO_FOLDER, file)

    print("\n==============================")
    print("AUDIO:", file)

    if file not in GROUND_TRUTH:
        print("No ground truth available\n")
        continue

    truth = GROUND_TRUTH[file]

    # ---------------- LOAD AUDIO ----------------
    audio, sr = librosa.load(path, sr=None, mono=True)

    # ---------------- BASELINE ----------------
    baseline_text = model.transcribe(path)["text"].strip()
    print(f"Baseline: {baseline_text}")
    print(f"WER: {get_wer(truth, baseline_text):.1f}%\n")

    # ---------------- RESAMPLE (16kHz) ----------------
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sf.write("outputs/temp.wav", audio_16k, 16000)

    resampled_text = model.transcribe("outputs/temp.wav")["text"].strip()
    print(f"Resampled: {resampled_text}")
    print(f"WER: {get_wer(truth, resampled_text):.1f}%\n")

    # ---------------- TRIM SILENCE ONLY ----------------
    trimmed_audio, _ = librosa.effects.trim(audio_16k, top_db=25)
    sf.write("outputs/temp_trim.wav", trimmed_audio, 16000)

    trimmed_text = model.transcribe("outputs/temp_trim.wav")["text"].strip()
    print(f"Trimmed: {trimmed_text}")
    print(f"WER: {get_wer(truth, trimmed_text):.1f}%\n")

    # ---------------- SAVE ----------------
    with open("outputs/asr_output.txt", "a", encoding="utf-8") as f:
        f.write("\n==============================\n")
        f.write(f"AUDIO: {file}\n")
        f.write(f"Baseline: {baseline_text}\n")
        f.write(f"Resampled: {resampled_text}\n")
        f.write(f"Trimmed: {trimmed_text}\n")

print("\nSaved to outputs/asr_output.txt")