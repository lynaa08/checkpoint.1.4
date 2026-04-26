"""
asr_level2.py  —  Smart Classroom Assistant
ASR module: transcribe lecture audio using OpenAI Whisper + audio preprocessing.
"""

import os
import re
import librosa
import soundfile as sf
import whisper
import numpy as np

try:
    from jiwer import wer as jiwer_wer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
AUDIO_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "audio")
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "..", "outputs", "asr_output.txt")
TEMP_DIR     = os.path.join(os.path.dirname(__file__), "..", "outputs", "temp_audio")

GROUND_TRUTH = {
    "audio1.wav":        "I'm sorry but I cannot fix this problem for you",
    "audio2.wav":        "You have been selected",
    "audio3.wav":        "You are just a line of code",
    "voice-message.wav": "Hi Lena how are you I hope you are fine I want to tell you lets start the project",
}


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_audio(path: str, out_path: str, target_sr: int = 16000) -> str:
    """
    Full audio preprocessing pipeline:
      1. Load and convert to mono
      2. Resample to 16 kHz (Whisper's native rate)
      3. Trim leading/trailing silence
      4. Simple noise gate (zero-out very quiet frames)
    Returns the path of the processed file.
    """
    audio, sr = librosa.load(path, sr=None, mono=True)

    # Resample
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    sr = target_sr

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # Noise gate: zero samples below 2% of max amplitude
    threshold = 0.02 * np.max(np.abs(audio))
    audio = np.where(np.abs(audio) < threshold, 0.0, audio)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, audio, sr)
    return out_path


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    if HAS_JIWER:
        return jiwer_wer(clean(reference), clean(hypothesis)) * 100
    # Fallback simple WER
    ref = clean(reference).split()
    hyp = clean(hypothesis).split()
    if not ref:
        return 0.0
    # Levenshtein distance (word-level)
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hyp) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return (d[len(ref)][len(hyp)] / len(ref)) * 100


# ─────────────────────────────────────────────
# TRANSCRIBE ONE FILE
# ─────────────────────────────────────────────
def transcribe_file(model, path: str) -> dict:
    filename = os.path.basename(path)
    print(f"[ASR] Processing: {filename}")

    # Baseline
    baseline_text = model.transcribe(path)["text"].strip()

    # Preprocessed
    temp_path = os.path.join(TEMP_DIR, "processed_" + filename)
    preprocess_audio(path, temp_path)
    processed_text = model.transcribe(temp_path)["text"].strip()

    result = {
        "file": filename,
        "baseline_text": baseline_text,
        "processed_text": processed_text,
        "metrics": {},
    }

    if filename in GROUND_TRUTH:
        gt = GROUND_TRUTH[filename]
        result["metrics"] = {
            "baseline_wer":   round(compute_wer(gt, baseline_text), 1),
            "processed_wer":  round(compute_wer(gt, processed_text), 1),
        }

    print(f"  Baseline : {baseline_text}")
    print(f"  Processed: {processed_text}")
    if result["metrics"]:
        m = result["metrics"]
        print(f"  WER  baseline={m['baseline_wer']}%  processed={m['processed_wer']}%")
    print()

    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_all(audio_folder: str = AUDIO_FOLDER, model_size: str = "base") -> list[dict]:
    supported = {".wav", ".mp3", ".m4a", ".flac"}
    files = [
        f for f in sorted(os.listdir(audio_folder))
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not files:
        print(f"[ASR] No audio files found in {audio_folder}")
        return []

    print("[ASR] Loading Whisper model …")
    model = whisper.load_model(model_size)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    results = []
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("=== ASR OUTPUT — Smart Classroom Assistant ===\n\n")

        for fname in files:
            path = os.path.join(audio_folder, fname)
            r = transcribe_file(model, path)
            results.append(r)

            out.write(f"--- {fname} ---\n")
            out.write(f"Baseline : {r['baseline_text']}\n")
            out.write(f"Processed: {r['processed_text']}\n")
            if r["metrics"]:
                out.write(f"Metrics  : {r['metrics']}\n")
            out.write("\n")

    print(f"[ASR] Saved → {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    run_all()
