"""
combine.py  —  Smart Classroom Assistant  
Level 2 Multimodal Note-Taking System

Pipeline:
  1. OCR  — extract text from lecture slide images
  2. ASR  — transcribe lecture audio recordings
  3. Combine — merge both into a single notes document
  4. Summarize — generate a concise summary (via Claude LLM)
  5. TTS  — read the summary aloud (audio file)

Usage:
  python combine.py
  python combine.py --images path/to/images --audio path/to/audio
  python combine.py --skip-tts        # skip audio generation
  python combine.py --skip-summary    # skip LLM summary step
"""

import os
import sys
import argparse
import datetime

# ── allow running from the project root ───────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import ocr_level2   as ocr_mod
import asr_level2   as asr_mod
import summarizer   as summarizer_mod
import tts_level2   as tts_mod


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT   = os.path.join(SRC_DIR, "..")
DEFAULT_IMAGES = os.path.join(PROJECT_ROOT, "data", "images")
DEFAULT_AUDIO  = os.path.join(PROJECT_ROOT, "data", "audio")
COMBINED_FILE  = os.path.join(PROJECT_ROOT, "outputs", "combined_notes.txt")


# ─────────────────────────────────────────────
# COMBINE
# ─────────────────────────────────────────────
def build_combined_doc(ocr_results: list, asr_results: list) -> str:
    """Merge OCR and ASR results into one structured notes document."""
    lines = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append("=" * 60)
    lines.append(f"  SMART CLASSROOM ASSISTANT — Combined Lecture Notes")
    lines.append(f"  Generated: {ts}")
    lines.append("=" * 60)
    lines.append("")

    # ── OCR section ────────────────────────────
    lines.append("## SLIDE CONTENT (OCR)")
    lines.append("-" * 40)
    if ocr_results:
        for r in ocr_results:
            if "error" in r:
                lines.append(f"[{r['file']}] Error: {r['error']}")
                continue
            text = r.get("processed_text") or r.get("raw_text", "")
            if text.strip():
                lines.append(f"[{r['file']}]")
                lines.append(text.strip())
                if r.get("metrics"):
                    m = r["metrics"]
                    lines.append(
                        f"  ↳ Accuracy: {m.get('processed_word_accuracy', '?')}%  "
                        f"Similarity: {m.get('processed_char_similarity', '?')}%"
                    )
                lines.append("")
    else:
        lines.append("(No images processed)")
        lines.append("")

    # ── ASR section ────────────────────────────
    lines.append("## AUDIO TRANSCRIPTS (ASR)")
    lines.append("-" * 40)
    if asr_results:
        for r in asr_results:
            text = r.get("processed_text") or r.get("baseline_text", "")
            if text.strip():
                lines.append(f"[{r['file']}]")
                lines.append(text.strip())
                if r.get("metrics"):
                    m = r["metrics"]
                    lines.append(f"  ↳ WER: {m.get('processed_wer', '?')}%")
                lines.append("")
    else:
        lines.append("(No audio files processed)")
        lines.append("")

    return "\n".join(lines)


def extract_plain_text(combined_doc: str) -> str:
    """Strip headers/metadata — keep only the actual content for the LLM."""
    import re
    # Remove section headers and separator lines
    lines = []
    for line in combined_doc.splitlines():
        stripped = line.strip()
        if stripped.startswith("=") or stripped.startswith("-"):
            continue
        if stripped.startswith("##") or stripped.startswith("↳"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        if "Smart Classroom" in stripped or "Generated:" in stripped:
            continue
        if stripped:
            lines.append(stripped)
    return " ".join(lines)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Smart Classroom Assistant — Level 2")
    parser.add_argument("--images",       default=DEFAULT_IMAGES, help="Folder with slide images")
    parser.add_argument("--audio",        default=DEFAULT_AUDIO,  help="Folder with audio files")
    parser.add_argument("--skip-summary", action="store_true",    help="Skip LLM summarisation")
    parser.add_argument("--skip-tts",     action="store_true",    help="Skip TTS audio output")
    parser.add_argument("--whisper-size", default="base",         help="Whisper model size (tiny/base/small)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  SMART CLASSROOM ASSISTANT — Level 2 Pipeline")
    print("="*60 + "\n")

    # ── Step 1: OCR ────────────────────────────────────────────
    print("─── STEP 1: OCR ──────────────────────────────────────────")
    ocr_results = []
    if os.path.isdir(args.images):
        ocr_results = ocr_mod.run_all(args.images)
    else:
        print(f"[OCR] Images folder not found: {args.images}")

    # ── Step 2: ASR ────────────────────────────────────────────
    print("─── STEP 2: ASR ──────────────────────────────────────────")
    asr_results = []
    if os.path.isdir(args.audio):
        asr_results = asr_mod.run_all(args.audio, model_size=args.whisper_size)
    else:
        print(f"[ASR] Audio folder not found: {args.audio}")

    # ── Step 3: Combine ────────────────────────────────────────
    print("─── STEP 3: COMBINE ──────────────────────────────────────")
    combined_doc = build_combined_doc(ocr_results, asr_results)

    os.makedirs(os.path.dirname(COMBINED_FILE), exist_ok=True)
    with open(COMBINED_FILE, "w", encoding="utf-8") as f:
        f.write(combined_doc)
    print(f"[Combine] Combined notes saved → {COMBINED_FILE}\n")

    # ── Step 4: Summarize ──────────────────────────────────────
    summary = ""
    if not args.skip_summary:
        print("─── STEP 4: SUMMARIZE ────────────────────────────────────")
        plain_text = extract_plain_text(combined_doc)
        if plain_text.strip():
            summary = summarizer_mod.run(plain_text)
            print("\nSUMMARY PREVIEW:")
            print(summary[:500])
        else:
            print("[Summarizer] No text to summarize.")
        print()
    else:
        print("─── STEP 4: SUMMARIZE (skipped) ──────────────────────────\n")

    # ── Step 5: TTS ────────────────────────────────────────────
    if not args.skip_tts:
        print("─── STEP 5: TTS ──────────────────────────────────────────")
        tts_text = summary if summary.strip() else extract_plain_text(combined_doc)[:600]
        if tts_text.strip():
            tts_mod.synthesize(tts_text)
        else:
            print("[TTS] Nothing to read aloud.")
        print()
    else:
        print("─── STEP 5: TTS (skipped) ────────────────────────────────\n")

    # ── Done ───────────────────────────────────────────────────
    print("="*60)
    print("  Pipeline complete! Output files:")
    print(f"    OCR notes   → {ocr_mod.OUTPUT_FILE}")
    print(f"    ASR notes   → {asr_mod.OUTPUT_FILE}")
    print(f"    Combined    → {COMBINED_FILE}")
    if not args.skip_summary:
        print(f"    Summary     → {summarizer_mod.SUMMARY_FILE}")
    if not args.skip_tts:
        print(f"    Audio recap → {tts_mod.TTS_OUTPUT}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
