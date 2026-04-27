"""
tts_level2.py  —  Smart Classroom Assistant
Text-to-Speech module: converts the lecture summary to an audio file.
Tries pyttsx3 first (offline), then falls back to gTTS (online).
"""

import os
import sys

TTS_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "summary_audio.mp3")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _try_gtts(text: str, out_path: str) -> bool:
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(out_path)
        print(f"[TTS] Audio saved via gTTS → {out_path}")
        return True
    except Exception as e:
        print(f"[TTS] gTTS failed: {e}")
        return False


def _try_pyttsx3(text: str, out_path: str) -> bool:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)   # words per minute
        engine.setProperty("volume", 1.0)
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        print(f"[TTS] Audio saved via pyttsx3 → {out_path}")
        return True
    except Exception as e:
        print(f"[TTS] pyttsx3 failed: {e}")
        return False


def _write_txt_fallback(text: str, out_path: str) -> str:
    txt_path = out_path.replace(".mp3", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("[TTS FALLBACK — no audio engine available]\n\n")
        f.write(text)
    print(f"[TTS] No audio engine found. Summary saved as text → {txt_path}")
    return txt_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def synthesize(text: str, out_path: str = TTS_OUTPUT) -> str:
    """
    Convert text to speech and save as MP3.
    Returns the path of the output file.
    """
    if not text.strip():
        print("[TTS] Empty text — nothing to synthesize.")
        return ""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if _try_gtts(text, out_path):
        return out_path
    if _try_pyttsx3(text, out_path):
        return out_path
    return _write_txt_fallback(text, out_path)


if __name__ == "__main__":
    sample = (
        "Welcome to the Smart Classroom Assistant. "
        "Today's lecture covered machine learning basics, "
        "including supervised learning, neural networks, and gradient descent."
    )
    synthesize(sample)
