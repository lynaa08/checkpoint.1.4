# Level 1 — Improvement & Analysis

Building on Level 0, we apply **image preprocessing for OCR** and **audio preprocessing for ASR** in order to evaluate their effect on recognition accuracy.

The goal of this level is to **compare different preprocessing techniques and analyze their real impact**, not necessarily to maximize performance.

---

## What's New in Level 1

| Task  | What we added                                  |
| ----- | ---------------------------------------------- |
| OCR   | Image preprocessing with OpenCV                |
| ASR   | Audio preprocessing with librosa + noisereduce |
| Bonus | Text-to-Speech (TTS) using gTTS                |

---

## New Files

```
src/
├── ocr_level1.py
├── asr_level1.py
└── tts_bonus.py
```

---

## Install Dependencies

```bash
pip install opencv-python librosa soundfile noisereduce jiwer gtts
```

---

# OCR Preprocessing (`ocr_level1.py`)

We tested 4 preprocessing methods and evaluated results using **Accuracy + Similarity metrics**.

| Step             | Effect                   |
| ---------------- | ------------------------ |
| Grayscale        | Removes color noise      |
| Threshold (Otsu) | Converts image to binary |
| Resize ×2        | Enlarges text            |
| Denoise          | Removes image noise      |

---

## OCR Results

```
Baseline:
H BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS ——
Accuracy: ~100.0% (noise affects metric slightly)
Similarity: 94.8%

Grayscale:
BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS
Accuracy: 100.0%
Similarity: 100.0%

Threshold:
BELIEVING IN YOURSELF §S THE SECRET TO SUCCESS
Accuracy: 87.5%
Similarity: 97.8%

Resize x2:
“ . BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS
Accuracy: 100.0%
Similarity: 95.8%

Denoised:
BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS
Accuracy: 100.0%
Similarity: 100.0%
```

---

## OCR Observation

* Grayscale and denoising produced perfect results (100% accuracy and similarity)
* Thresholding introduced small character distortion (`IS → §S`)
* Resizing introduced artifacts in this image, affecting readability
* Baseline already performed relatively well, but contained noise and extra characters

👉 Conclusion: OCR performance depends strongly on preprocessing, but not all techniques improve results in every case.

---

# ASR Preprocessing (`asr_level1.py`)

We tested Whisper transcription with different preprocessing techniques.

| Step            | Effect                              |
| --------------- | ----------------------------------- |
| Resample 16kHz  | Standardizes audio format           |
| Trim silence    | Removes silent parts                |
| Noise reduction | Attempts to remove background noise |

---

## ASR Results

```
Baseline:
Hi Lena, how are you? I hope you are fine. I want to tell you let's start the project.
WER: 0.0%

Resampled 16kHz:
Same output
WER: 0.0%

Trimmed silence:
Same output
WER: 0.0%

Noise reduced:
I hope you are fine. I hope to see you. Let's start the project.
WER: 36.8%
```

---

## ASR Observation

* Whisper performs extremely well on this clean audio (0% WER baseline)
* Resampling has no effect because Whisper already expects 16kHz audio
* Silence trimming does not change results significantly
* Noise reduction degraded performance because it removed or distorted parts of the speech signal

👉 Conclusion: Whisper is robust to preprocessing, and additional processing does not always improve results.

---

# Bonus — Text-to-Speech (TTS)

The extracted OCR text is converted into speech using Google Text-to-Speech (gTTS), generating a playable MP3 file.

```
outputs/tts_output.mp3
```

---

# Key Takeaway

> OCR benefits strongly from preprocessing, while ASR (Whisper) is already highly optimized and often does not benefit from additional preprocessing on clean audio.
