"""Assemble slides + audio into final MP4 video."""

from __future__ import annotations

from pathlib import Path

from moviepy import AudioFileClip, ImageClip, concatenate_videoclips

from . import theme


def assemble_video(
    slide_paths: list[Path],
    audio_results: list[tuple[Path, float]],
    output_path: Path,
) -> Path:
    """Combine slides and audio into a video.

    slide_paths has title + N content slides + closing = N+2 items.
    audio_results has N items (one per content segment).

    Title slide gets duration of first audio segment.
    Closing slide gets MIN_SLIDE_DURATION with silence.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clips = []

    # Title slide — use first segment's audio
    if audio_results:
        title_duration = audio_results[0][1]
        title_audio_path = audio_results[0][0]
        title_clip = ImageClip(str(slide_paths[0]), duration=title_duration)
        title_audio = AudioFileClip(str(title_audio_path))
        title_clip = title_clip.with_audio(title_audio)
        clips.append(title_clip)

    # Content slides — each paired with its audio
    for i, (audio_path, duration) in enumerate(audio_results):
        slide_idx = i + 1  # offset by title slide
        if slide_idx >= len(slide_paths) - 1:  # don't exceed content slides
            break
        clip = ImageClip(str(slide_paths[slide_idx]), duration=duration)
        audio = AudioFileClip(str(audio_path))
        clip = clip.with_audio(audio)
        clips.append(clip)

    # Closing slide — silent
    if len(slide_paths) > 1:
        closing_duration = max(theme.MIN_SLIDE_DURATION, 5.0)
        closing_clip = ImageClip(str(slide_paths[-1]), duration=closing_duration)
        clips.append(closing_clip)

    if not clips:
        raise ValueError("No clips to assemble")

    # Concatenate with crossfade
    padding = -theme.CROSSFADE_DURATION if len(clips) > 1 else 0
    final = concatenate_videoclips(clips, method="compose", padding=padding)

    print(f"  Rendering {final.duration:.1f}s video...")
    final.write_videofile(
        str(output_path),
        fps=theme.FPS,
        codec=theme.VIDEO_CODEC,
        audio_codec=theme.AUDIO_CODEC,
        preset=theme.VIDEO_PRESET,
        threads=theme.VIDEO_THREADS,
        logger="bar",
    )

    # Cleanup
    for c in clips:
        c.close()
    final.close()

    return output_path
