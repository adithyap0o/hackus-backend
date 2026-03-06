"""
backend.py
──────────
Complete SubtitleAI backend pipeline.
Combines audio extraction (moviepy) + transcription + translation (Groq).

Usage:
    python3 backend.py video.mp4
    python3 backend.py video.mp4 Malayalam

Import into Streamlit:
    from backend import process_video
"""

import os
import sys
from pathlib import Path
from openai import OpenAI

# ─── PASTE YOUR GROQ API KEY HERE ────────────────────────────────────────────
GROQ_API_KEY = "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"   # ← replace with your gsk_... key
# ─────────────────────────────────────────────────────────────────────────────

# Groq uses the OpenAI SDK — just with a different base URL
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


# ─── STEP 1: EXTRACT AUDIO ───────────────────────────────────────────────────

def extract_audio(video_path: str, audio_output_path: str = "temp_audio.mp3") -> str:
    """
    Extract audio track from a video file and save as MP3.

    Args:
        video_path:         Path to the input video file (.mp4, .mov, etc.)
        audio_output_path:  Where to save the extracted audio. Default: temp_audio.mp3

    Returns:
        Path to the saved audio file.
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise RuntimeError(
            "moviepy is not installed.\n"
            "Fix: pip install moviepy"
        )

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"  → Extracting audio from: {video_path.name}")

    with VideoFileClip(str(video_path)) as video:
        if video.audio is None:
            raise RuntimeError("This video has no audio track.")
        video.audio.write_audiofile(audio_output_path, logger=None)

    print(f"  → Audio saved to: {audio_output_path}")
    return audio_output_path


# ─── STEP 2: TRANSCRIBE ──────────────────────────────────────────────────────

def transcribe(audio_path: str) -> str:
    """
    Send audio to Groq Whisper and get back a formatted .srt string.

    Args:
        audio_path: Path to the audio file (mp3, wav, m4a, etc.)

    Returns:
        SRT-formatted subtitle string with timestamps.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "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":
        raise ValueError(
            "No Groq API key set.\n"
            "Fix: open backend.py and paste your gsk_... key into GROQ_API_KEY at the top."
        )

    print("  → Sending audio to Groq Whisper...")

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-large-v3",   # Groq's Whisper model (same quality as OpenAI's)
            file=audio_file,
            response_format="srt",      # returns ready-made SRT with timestamps
        )

    print("  → Transcription complete.")
    return response   # response IS the srt string


# ─── STEP 3: TRANSLATE ───────────────────────────────────────────────────────

def translate(srt_content: str, target_language: str) -> str:
    """
    Translate the dialogue lines of an SRT string while keeping timestamps intact.

    Args:
        srt_content:      The raw SRT string from transcribe().
        target_language:  Language name e.g. "Malayalam", "Hindi", "Spanish".

    Returns:
        Translated SRT string.
    """
    print(f"  → Translating to {target_language} via Groq Llama...")

    system_prompt = (
        f"Translate the following SRT subtitle text into {target_language}. "
        "You must keep all timestamp lines (e.g., 00:00:01,000 --> 00:00:04,000) "
        "exactly the same. Only translate the spoken dialogue lines. "
        "Do not add any conversational filler or markdown code blocks to your response."
    )

    response = client.chat.completions.create(
        model="llama3-8b-8192",     # Groq's free fast model — great for translation
        temperature=0.2,            # low = more literal and consistent
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": srt_content},
        ],
    )

    translated = response.choices[0].message.content.strip()
    print(f"  → Translation to {target_language} complete.")
    return translated


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    target_language: str | None = None,
    audio_output_path: str = "temp_audio.mp3",
) -> tuple[str, str]:
    """
    Full pipeline: video → audio → transcription → (optional) translation → SRT.

    This is the function to import into your Streamlit app.

    Args:
        video_path:         Path to the uploaded video file.
        target_language:    Optional language to translate into (e.g. "Malayalam").
                            Pass None to skip translation.
        audio_output_path:  Where to store the temporary audio file.

    Returns:
        A tuple of (srt_string, status_message) ready to display in the UI.
    """
    try:
        # Step 1 — Extract audio
        audio_path = extract_audio(video_path, audio_output_path)

        # Step 2 — Transcribe
        srt_content = transcribe(audio_path)

        # Step 3 — Translate (optional)
        if target_language:
            srt_content = translate(srt_content, target_language)
            status = f"Done · Transcribed + Translated to {target_language}"
        else:
            status = "Done · Transcription complete (original language)"

        # Clean up temp audio file
        if Path(audio_output_path).exists():
            os.remove(audio_output_path)

        return srt_content, status

    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))
    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {e}")


# ─── SMOKE TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    video  = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
    lang   = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\n── SubtitleAI Backend ───────────────────────")
    print(f"   Video  : {video}")
    print(f"   Translate to: {lang or 'none (original language)'}")
    print(f"────────────────────────────────────────────\n")

    srt, status = process_video(video, target_language=lang)

    print(f"\n── Status ───────────────────────────────────")
    print(f"   {status}")
    print(f"\n── SRT Output (first 500 chars) ─────────────")
    print(srt[:500], "..." if len(srt) > 500 else "")
