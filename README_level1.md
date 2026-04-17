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
├── tts_bonus.py
├── images/
│   ├── image-1.png
│   ├── image-2.jpg
│   ├── image-3.jpg
│   └── my-image.jpg
└── audio/
    ├── audio1.wav
    ├── audio2.wav
    ├── audio3.wav
    └── voice-message.wav
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

## image-1.png

Baseline: BE STRONGER THAN THE CHALLENGE  
Accuracy: 100.0%  
Similarity: 100.0%

Grayscale: BE STRONGER THAN THE CHALLENGE  
Accuracy: 100.0%  
Similarity: 100.0%

Threshold: BE STRONGER THAN THE CHALLENGE  
Accuracy: 100.0%  
Similarity: 100.0%

Resize x2: BE STRONGER THAN THE CHALLENGE  
Accuracy: 100.0%  
Similarity: 100.0%

Denoised: BE STRONGER THAN THE CHALLENGE  
Accuracy: 100.0%  
Similarity: 100.0%

---

## image-2.jpg

Baseline:  
Accuracy: 0.0%  
Similarity: 0.0%

Grayscale:  
Accuracy: 0.0%  
Similarity: 0.0%

Threshold:  
Accuracy: 0.0%  
Similarity: 0.0%

Resize x2: T0 ae A  
Accuracy: 16.7%  
Similarity: 40.0%

Denoised:  
Accuracy: 0.0%  
Similarity: 0.0%

---

## image-3.jpg

Baseline: noisy text detected  
Accuracy: 50.6%  
Similarity: 3.8%

Grayscale:  
Accuracy: 50.6%  
Similarity: 4.1%

Threshold:  
Accuracy: 49.4%  
Similarity: 4.4%

Resize x2:  
Accuracy: 49.4%  
Similarity: 1.6%

Denoised:  
Accuracy: 57.3%  
Similarity: 3.5%

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

## audio1.wav

Baseline: I'm sorry, but I cannot fix this problem for you.  
WER: 0.0%

Resampled: same output  
WER: 0.0%

Trimmed: same output  
WER: 0.0%

---

## audio2.wav

Baseline: You have been selected.  
WER: 0.0%

Resampled: same output  
WER: 0.0%

Trimmed: same output  
WER: 0.0%

---

## audio3.wav

Baseline: You are just a line of code.  
WER: 0.0%

Resampled: same output  
WER: 0.0%

Trimmed: same output  
WER: 0.0%

---

## voice-message.wav

Baseline: Hi Lena, how are you? I hope you are fine. I want to tell you let's start the project.  
WER: 0.0%

Resampled: same output  
WER: 0.0%

Trimmed: same output  
WER: 0.0%

---

# ASR Observation

- Whisper is very stable on all audio files
- Preprocessing does not change results for clean audio
- Only noisy audio would show improvement

---

# Bonus — Text-to-Speech (TTS)

outputs/tts_output.mp3

---

# Key Takeaway

> OCR benefits strongly from preprocessing, while ASR (Whisper) is already highly optimized and often does not benefit from additional preprocessing on clean audio.
