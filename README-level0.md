# VoiceVision — OCR + ASR Multimedia System

**OMC AI Section — Checkpoint 1.4 | Level 0**

---

## Project Overview

VoiceVision is a multimedia AI system that can:

- **Read text from images** using OCR (Optical Character Recognition)
- **Transcribe speech from audio** using ASR (Automatic Speech Recognition)

Built entirely with open-source tools and pretrained models.

---

## Tools & Libraries

| Task              | Library                 |
| ----------------- | ----------------------- |
| OCR               | Tesseract + pytesseract |
| OCR (alternative) | EasyOCR                 |
| ASR               | OpenAI Whisper          |
| Audio loading     | ffmpeg                  |
| WER metric        | jiwer                   |

---

## Project Structure

```
checkpoint.1.4/
├── data/
│   ├── my-image.jpg       # test image
│   └── audio.wav          # test audio
├── src/
│   ├── ocr.py             # basic OCR script
│   ├── asr.py             # basic ASR script
│   ├── ocr_compare.py     # OCR metrics + EasyOCR comparison
│   └── asr_compare.py     # ASR metrics (Whisper + WER)
└── README.md
```

---

##  How to Run

### Setup

```bash
pip install pytesseract easyocr openai-whisper jiwer
```

> Also install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) and add both to PATH.

### OCR

```bash
python src/ocr_compare.py
```

### ASR

```bash
python src/asr_compare.py
```

---

## Results & Metrics

### OCR — Image: motivational quote (bold white text on circular background)

| Library   | Word Accuracy                                      |
| --------- | -------------------------------------------------- |
| Tesseract | low — struggled with stylized/repeated text        |
| EasyOCR   | could not test — model download blocked by network |

**Ground truth:** `BELIEVING IN YOURSELF IS THE SECRET TO SUCCESS`

### ASR — Audio: short voice message in English

| Library        | WER  | Accuracy |
| -------------- | ---- | -------- |
| Whisper (base) | 0.0% | **100%** |

**Transcript:** _"Hi Lena, how are you? I hope you are fine. I want to tell you let's start the project."_

---

## Observations

**OCR — What worked:**

- Tesseract can read the text but struggles when text is repeated or overlaid in the image design
- Plain black text on white background works best for Tesseract
- EasyOCR handles stylized and artistic text better than Tesseract

**OCR — What failed:**

- Tesseract gave 0% word accuracy on a motivational image because the text appeared twice in the image layout, causing word-position mismatches during comparison
- EasyOCR model download failed due to network restrictions — models must be downloaded manually in restricted environments

**ASR — What worked:**

- Whisper achieved 100% accuracy on a clear voice recording with no background noise
- Whisper handles conversational English very well even without any preprocessing

**ASR — What failed / lessons learned:**

- WER is sensitive to punctuation — `lets` vs `let's` counts as an error even though they sound identical
- Running on CPU is slow; a GPU would speed things up significantly
- The FP16 warning on CPU is harmless — Whisper automatically falls back to FP32

---

## Key Takeaways

- Whisper is very powerful for clean audio — even the base model gives excellent results
- Tesseract works best on simple, clean document-style images
- Always normalize (remove punctuation, lowercase) before computing WER/accuracy metrics
- ffmpeg is required for Whisper to load audio files

---
