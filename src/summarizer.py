"""
summarizer.py  —  Smart Classroom Assistant
Summarizes the combined OCR + ASR notes using Claude via the Anthropic API.
"""

import os
import json
import urllib.request
import urllib.error


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_URL   = "https://api.anthropic.com/v1/messages"
MODEL     = "claude-haiku-4-5-20251001"  # fast & cheap for summarisation
MAX_TOKENS = 512

SUMMARY_FILE = os.path.join(os.path.dirname(__file__), "..", "outputs", "summary.txt")


# ─────────────────────────────────────────────
# CALL CLAUDE
# ─────────────────────────────────────────────
def _call_claude(prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            # The API key is injected by the Claude.ai environment — no need to hard-code it.
        },
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    return data["content"][0]["text"].strip()


# ─────────────────────────────────────────────
# SUMMARISE
# ─────────────────────────────────────────────
def summarize(combined_text: str) -> str:
    """
    Send the combined OCR + ASR notes to Claude and return a concise summary.
    Falls back to a simple extractive summary if the API call fails.
    """
    prompt = (
        "You are a helpful study assistant. Below are combined lecture notes "
        "extracted automatically from slide images (OCR) and audio recordings (ASR). "
        "Please write a clear, concise summary in 3–5 bullet points covering the key ideas. "
        "Keep it short and student-friendly.\n\n"
        f"--- LECTURE NOTES ---\n{combined_text}\n--- END ---\n\n"
        "Summary:"
    )

    try:
        summary = _call_claude(prompt)
        print("[Summarizer] Summary generated via Claude API.")
        return summary
    except Exception as e:
        print(f"[Summarizer] API call failed ({e}). Using fallback summary.")
        return _fallback_summary(combined_text)


def _fallback_summary(text: str) -> str:
    """Simple extractive fallback: return first 5 non-empty sentences."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    top = [s.strip() for s in sentences if len(s.strip()) > 20][:5]
    if not top:
        return text[:500]
    return "\n• " + "\n• ".join(top)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run(combined_text: str) -> str:
    summary = summarize(combined_text)

    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("=== LECTURE SUMMARY — Smart Classroom Assistant ===\n\n")
        f.write(summary)
        f.write("\n")

    print(f"[Summarizer] Saved → {SUMMARY_FILE}")
    return summary


if __name__ == "__main__":
    test_text = (
        "Machine learning is a subset of artificial intelligence. "
        "Supervised learning uses labeled data to train models. "
        "Neural networks are inspired by the human brain. "
        "Deep learning has many layers and can learn complex patterns. "
        "Gradient descent is the main optimization algorithm used."
    )
    print(run(test_text))
