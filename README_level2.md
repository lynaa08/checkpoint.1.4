#  Smart Classroom Assistant

**Multimedia Systems Mini Project — Level 2**
*A fully multimodal note-taking system that reads lecture slides and transcribes spoken audio into structured, summarised notes.*

---
##  What It Does

| Step | Module | Description |
|------|--------|-------------|
| 1 | `ocr_level2.py` | Reads text from lecture slide images (Tesseract + OpenCV) |
| 2 | `asr_level2.py` | Transcribes lecture audio recordings (OpenAI Whisper) |
| 3 | `combine.py` | Merges OCR + ASR outputs into one structured notes document |
| 4 | `summarizer.py` | Generates a concise bullet-point summary (Claude LLM) |
| 5 | `tts_level2.py` | Converts the summary to an audio recap (gTTS / pyttsx3) |

---

## ️ Project Structure

```
smart-classroom-assistant/
├── data/
│   ├── images/          ← place your lecture slide images here
│   └── audio/           ← place your lecture audio files here
├── outputs/             ← all generated files appear here
│   ├── ocr_output.txt
│   ├── asr_output.txt
│   ├── combined_notes.txt
│   ├── summary.txt
│   └── summary_audio.mp3
├── src/
│   ├── ocr_level2.py    ← OCR pipeline
│   ├── asr_level2.py    ← ASR pipeline
│   ├── summarizer.py    ← LLM summariser
│   ├── tts_level2.py    ← Text-to-Speech
│   └── combine.py       ← Main pipeline runner
├── requirements.txt
└── README.md
```

---

##  Setup

### 1. Install system dependencies

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr ffmpeg

# macOS
brew install tesseract ffmpeg
```

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

### 3. Add your data

- Drop slide images (`.png`, `.jpg`) into `data/images/`
- Drop audio files (`.wav`, `.mp3`) into `data/audio/`

---

## Run

**Full pipeline (recommended):**
```bash
python src/combine.py
```

**Custom paths:**
```bash
python src/combine.py --images path/to/slides --audio path/to/recordings
```

**Skip optional steps:**
```bash
python src/combine.py --skip-summary   # no LLM call
python src/combine.py --skip-tts       # no audio output
```

**Use a smaller/faster Whisper model:**
```bash
python src/combine.py --whisper-size tiny
```

**Run individual modules:**
```bash
python src/ocr_level2.py    # OCR only
python src/asr_level2.py    # ASR only
python src/summarizer.py    # Summariser test
python src/tts_level2.py    # TTS test
```

---

##  Metrics

### OCR
| Metric | Description |
|--------|-------------|
| Word Accuracy | % of ground-truth words found in extracted text |
| Char Similarity | SequenceMatcher character-level similarity (%) |

### ASR
| Metric | Description |
|--------|-------------|
| WER | Word Error Rate — lower is better |

---

##  OCR Preprocessing Pipeline

1. **Grayscale** conversion  
2. **Upscaling** (if image width < 1000 px)  
3. **Denoising** (`cv2.fastNlMeansDenoising`)  
4. **Adaptive thresholding** (handles uneven lighting)  
5. **Deskewing** (corrects small rotation angles)

##  ASR Preprocessing Pipeline

1. **Mono conversion** (merge stereo channels)  
2. **Resampling** to 16 kHz (Whisper's native rate)  
3. **Silence trimming** (`librosa.effects.trim`)  
4. **Noise gate** (zero-out sub-threshold frames)

---

##  Tools Used

| Task | Tool |
|------|------|
| OCR | Tesseract 5, pytesseract, OpenCV, Pillow |
| ASR | OpenAI Whisper (base model) |
| Image Processing | OpenCV, NumPy |
| Audio Processing | librosa, soundfile |
| LLM Summarisation | Claude (Anthropic API) |
| TTS | gTTS / pyttsx3 |
| ASR Metrics | jiwer (WER) |
| OCR Metrics | SequenceMatcher |

---

##  Known Failure Cases

**OCR:**
- Decorative / handwritten fonts are poorly recognised
- Very dark or low-contrast backgrounds confuse thresholding
- Overlapping text on busy slide backgrounds

**ASR:**
- Heavy background noise (music, crowd) degrades accuracy
- Strong accents may increase WER
- Very fast speech or overlapping speakers

---

##  Author

OMC AI Section — Checkpoint 1.4 — Level 2
