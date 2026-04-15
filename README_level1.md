# Level 1 — Improvement & Analysis

Building on Level 0, we now apply **image preprocessing** for OCR and **audio preprocessing** for ASR to improve accuracy.

---

## What's New in Level 1

| Task | What we added |
|------|--------------|
| OCR  | Image preprocessing with OpenCV |
| ASR  | Audio preprocessing with librosa + noisereduce |
| Bonus | Text-to-Speech with pyttsx3 |

---

## New Files

```
src/
├── ocr_level1.py     # OCR with image preprocessing
├── asr_level1.py     # ASR with audio preprocessing
└── tts_bonus.py      # Bonus: read the output aloud
```

---

## Install Dependencies

```bash
pip install opencv-python librosa soundfile noisereduce jiwer pyttsx3
```

---

## OCR Preprocessing (`ocr_level1.py`)

We tested 4 preprocessing steps and compared accuracy against the baseline.

| Step | What it does | Why it helps |
|------|-------------|--------------|
| Grayscale | Remove color info | Simpler image = easier for Tesseract |
| Threshold (Otsu) | Turn image black & white | Separates text from background clearly |
| Resize ×2 | Make image bigger | Small text is hard to read |
| Denoise | Remove noise/grain | Clean image = fewer OCR mistakes |

**How to run:**
```bash
python src/ocr_level1.py
```

**Sample output:**
```
Baseline:  8ELlEVlNG lN Y0URSELF lS THE SECRET TO SUCCESS   Accuracy: 40.0%
Grayscale: BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS   Accuracy: 100.0%
Threshold: BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS   Accuracy: 100.0%
Resize x2: BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS   Accuracy: 100.0%
Denoised:  BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS   Accuracy: 100.0%
```

**Observation:** Even simple grayscale conversion dramatically improved the result. The baseline failed because the colored background confused Tesseract.

---

## ASR Preprocessing (`asr_level1.py`)

We tested 3 preprocessing steps on the audio before feeding it to Whisper.

| Step | What it does | Why it helps |
|------|-------------|--------------|
| Resample to 16kHz | Change sample rate | Whisper is trained on 16kHz audio |
| Trim silence | Remove empty parts at start/end | Shorter input = fewer hallucinations |
| Noise reduction | Filter background noise | Cleaner signal = better transcription |

**How to run:**
```bash
python src/asr_level1.py
```

**Sample output:**
```
Baseline:        Hi Lena, how are you? ...   WER: 5.0%
Resampled 16kHz: Hi Lena, how are you? ...   WER: 5.0%
Trimmed silence: Hi Lena, how are you? ...   WER: 0.0%
Noise reduced:   Hi Lena, how are you? ...   WER: 0.0%
```

**Observation:** Whisper was already good on clean audio. Trimming silence gave the biggest improvement by removing parts where Whisper sometimes hallucinates text.

---

## Bonus — TTS (`tts_bonus.py`)

Reads the OCR output file aloud and saves it as an MP3.

```bash
python src/tts_bonus.py
```

Output saved to: `outputs/tts_output.mp3`

---

## Key Takeaway

> Preprocessing matters more for OCR than ASR. Tesseract is very sensitive to image quality, while Whisper handles most audio well even without preprocessing.