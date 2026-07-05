#!/usr/bin/env python3
"""Bloom pipeline CLI — render markdown scripts to video.

Usage:
    python bloom/pipeline/render.py bloom/drafts/2026-02-11_security_and_bloom.md
    python bloom/pipeline/render.py script.md --parse-only
    python bloom/pipeline/render.py script.md --audio-only
    python bloom/pipeline/render.py script.md --slides-only
    python bloom/pipeline/render.py script.md --voice en-US-GuyNeural
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure bloom package is importable when run directly
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bloom.pipeline.parser import parse_script
from bloom.pipeline.tts import generate_audio
from bloom.pipeline.slides import generate_slides
from bloom.pipeline.assembler import assemble_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Bloom: render markdown scripts to video")
    parser.add_argument("script", type=Path, help="Path to markdown script")
    parser.add_argument("--parse-only", action="store_true", help="Parse and print segments, no output")
    parser.add_argument("--audio-only", action="store_true", help="Generate TTS audio only")
    parser.add_argument("--slides-only", action="store_true", help="Generate slide images only")
    parser.add_argument("--voice", type=str, default=None, help="TTS voice (e.g. en-US-GuyNeural)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    args = parser.parse_args()

    if not args.script.exists():
        print(f"Error: {args.script} not found")
        sys.exit(1)

    t0 = time.time()

    # Output directory
    script_name = args.script.stem
    out_base = args.output_dir or (Path("bloom/output") / script_name)
    segments_dir = out_base / "segments"

    # Step 1: Parse
    print(f"[1/4] Parsing {args.script.name}...")
    script = parse_script(args.script)
    print(f"  Title: {script.title}")
    print(f"  Entry: {script.entry_number}")
    print(f"  Segments: {len(script.segments)}")

    if args.parse_only:
        print()
        for seg in script.segments:
            ts = f"[{seg.timestamps}] " if seg.timestamps else ""
            code_flag = " [CODE]" if seg.code_block else ""
            narr_preview = seg.narration[:80].replace("\n", " ") + ("..." if len(seg.narration) > 80 else "")
            print(f"  {seg.segment_index:03d} {ts}{seg.section_title}{code_flag}")
            print(f"      Visual: {seg.visual_hint[:60]}" if seg.visual_hint else "      Visual: (none)")
            print(f"      Narration: {narr_preview}")
            print()
        print(f"Total: {len(script.segments)} segments")
        return

    # Step 2: TTS
    if not args.slides_only:
        print(f"\n[2/4] Generating audio ({len(script.segments)} segments)...")
        audio_dir = segments_dir / "audio"
        audio_results = generate_audio(script, audio_dir, voice=args.voice)
        total_audio = sum(d for _, d in audio_results)
        print(f"  Total audio: {total_audio:.1f}s ({total_audio / 60:.1f} min)")
    else:
        audio_results = []

    if args.audio_only:
        print(f"\nAudio files in: {segments_dir / 'audio'}")
        print(f"Done in {time.time() - t0:.1f}s")
        return

    # Step 3: Slides
    if not args.audio_only:
        print(f"\n[3/4] Generating slides ({len(script.segments) + 2} images)...")
        slides_dir = segments_dir / "slides"
        slide_paths = generate_slides(script, slides_dir)
    else:
        slide_paths = []

    if args.slides_only:
        print(f"\nSlide images in: {segments_dir / 'slides'}")
        print(f"Done in {time.time() - t0:.1f}s")
        return

    # Step 4: Assemble
    print(f"\n[4/4] Assembling video...")
    video_path = out_base / f"{script_name}.mp4"
    assemble_video(slide_paths, audio_results, video_path)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
