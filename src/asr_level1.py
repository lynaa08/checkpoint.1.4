import whisper
import librosa
import soundfile as sf
import noisereduce as nr
from jiwer import wer
import re

AUDIO_PATH = r"C:\Users\tassili\Documents\checkpoint.1.4\data\voice-message.wav"
GROUND_TRUTH = "hi lena how are you i hope you are fine i want to tell you lets start the project"

def clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def get_wer(reference, hypothesis):
    return wer(clean(reference), clean(hypothesis)) * 100

model = whisper.load_model("base")

# load audio
audio, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)

# --- baseline ---
baseline_text = model.transcribe(AUDIO_PATH)["text"].strip()
print(f"Baseline: {baseline_text}")
print(f"WER: {get_wer(GROUND_TRUTH, baseline_text):.1f}%\n")

# --- resample to 16kHz (whisper's preferred rate) ---
audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
sf.write("outputs/temp_resampled.wav", audio_16k, 16000)
resampled_text = model.transcribe("outputs/temp_resampled.wav")["text"].strip()
print(f"Resampled 16kHz: {resampled_text}")
print(f"WER: {get_wer(GROUND_TRUTH, resampled_text):.1f}%\n")

# --- trim silence ---
audio_trimmed, _ = librosa.effects.trim(audio_16k, top_db=20)
sf.write("outputs/temp_trimmed.wav", audio_trimmed, 16000)
trimmed_text = model.transcribe("outputs/temp_trimmed.wav")["text"].strip()
print(f"Trimmed silence: {trimmed_text}")
print(f"WER: {get_wer(GROUND_TRUTH, trimmed_text):.1f}%\n")

# --- noise reduction ---
noise_sample = audio_16k[:int(16000 * 0.5)]  # first 0.5s as noise profile
audio_denoised = nr.reduce_noise(y=audio_16k, sr=16000, y_noise=noise_sample)
sf.write("outputs/temp_denoised.wav", audio_denoised, 16000)
denoised_text = model.transcribe("outputs/temp_denoised.wav")["text"].strip()
print(f"Noise reduced: {denoised_text}")
print(f"WER: {get_wer(GROUND_TRUTH, denoised_text):.1f}%\n")

# save results
with open("outputs/asr_level1_output.txt", "w") as f:
    f.write(f"Ground truth:    {GROUND_TRUTH}\n\n")
    f.write(f"Baseline:        {baseline_text}  (WER: {get_wer(GROUND_TRUTH, baseline_text):.1f}%)\n")
    f.write(f"Resampled 16kHz: {resampled_text}  (WER: {get_wer(GROUND_TRUTH, resampled_text):.1f}%)\n")
    f.write(f"Trimmed silence: {trimmed_text}  (WER: {get_wer(GROUND_TRUTH, trimmed_text):.1f}%)\n")
    f.write(f"Noise reduced:   {denoised_text}  (WER: {get_wer(GROUND_TRUTH, denoised_text):.1f}%)\n")

print("saved to outputs/asr_level1_output.txt")