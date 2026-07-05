"""Generate TTS audio for each segment using edge-tts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts

from . import theme
from .parser import ScriptData


async def _generate_one(text: str, outpath: Path, voice: str) -> None:
    """Generate a single MP3 from text."""
    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        rate=theme.TTS_RATE,
        pitch=theme.TTS_PITCH,
    )
    await communicate.save(str(outpath))


def _get_duration(mp3_path: Path) -> float:
    """Get MP3 duration in seconds using moviepy."""
    from moviepy import AudioFileClip
    with AudioFileClip(str(mp3_path)) as clip:
        return clip.duration


def generate_audio(
    script: ScriptData,
    output_dir: Path,
    voice: str | None = None,
) -> list[tuple[Path, float]]:
    """Generate MP3 files for all segments.

    Returns list of (mp3_path, duration_seconds).
    """
    voice = voice or theme.TTS_VOICE
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[Path, float]] = []

    for seg in script.segments:
        narration = seg.narration.strip()
        if not narration:
            narration = seg.visual_hint or seg.section_title or "..."

        mp3_path = output_dir / f"seg_{seg.segment_index:03d}.mp3"
        print(f"  TTS [{seg.segment_index:03d}] {seg.section_title[:40]}...", end=" ", flush=True)

        asyncio.run(_generate_one(narration, mp3_path, voice))

        duration = _get_duration(mp3_path)
        duration = max(duration, theme.MIN_SLIDE_DURATION)
        results.append((mp3_path, duration))
        print(f"{duration:.1f}s")

    return results
