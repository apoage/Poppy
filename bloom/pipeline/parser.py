"""Parse Bloom markdown scripts into structured segments."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Segment:
    """One visual+narration unit — becomes one slide + one audio clip."""
    section_title: str
    timestamps: str  # e.g. "0:00-0:30", empty if none
    visual_hint: str
    narration: str
    code_block: str | None = None
    segment_index: int = 0


@dataclass
class ScriptData:
    """Full parsed script."""
    title: str
    entry_number: str  # e.g. "011" or "001"
    date: str
    segments: list[Segment] = field(default_factory=list)


# Patterns
RE_SECTION = re.compile(
    r"^##\s+"
    r"(?:\[(\d+:\d+-\d+:\d+)\]\s*)?"  # optional [0:00-0:30]
    r"(.+)$"
)
RE_VISUAL = re.compile(r"^\*Visual:\s*(.+?)\*?\s*$")
RE_NARRATION_MARKER = re.compile(r"^\*\*NARRATION:\*\*\s*$")
RE_CODE_FENCE = re.compile(r"^```")
RE_TITLE_H1 = re.compile(r"^#\s+(?:Bloom Script:\s*)?(?:Entry\s+\d+\s*[-–—]\s*)?(.+)$")
RE_DATE = re.compile(r"^\*?\*?Date\*?\*?:\s*(.+)$")
RE_ENTRY_NUM = re.compile(r"Entry\s+(\d+)", re.IGNORECASE)
RE_QUOTE = re.compile(r"^>\s*(.+)$")


def parse_script(path: Path) -> ScriptData:
    """Parse a Bloom markdown script into ScriptData."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    sd = ScriptData(title="", entry_number="", date="")
    current_section = ""
    current_timestamps = ""
    current_visual = ""
    current_narration_lines: list[str] = []
    current_code_lines: list[str] | None = None
    in_code = False
    in_narration = False
    stop_parsing = False

    def _flush_segment():
        """Emit current segment if there's content."""
        narration = "\n".join(current_narration_lines).strip()
        code = "\n".join(current_code_lines).strip() if current_code_lines else None
        if narration or code:
            seg = Segment(
                section_title=current_section,
                timestamps=current_timestamps,
                visual_hint=current_visual,
                narration=narration,
                code_block=code if code else None,
                segment_index=len(sd.segments),
            )
            sd.segments.append(seg)

    for line in lines:
        if stop_parsing:
            break

        # Code fence toggle
        if RE_CODE_FENCE.match(line):
            if in_code:
                # End code block
                in_code = False
                continue
            else:
                # Start code block
                in_code = True
                if current_code_lines is None:
                    current_code_lines = []
                continue

        if in_code:
            if current_code_lines is None:
                current_code_lines = []
            current_code_lines.append(line)
            continue

        # H1 title
        m = RE_TITLE_H1.match(line)
        if m and not sd.title:
            sd.title = m.group(1).strip()
            # Try to extract entry number from full line
            em = RE_ENTRY_NUM.search(line)
            if em:
                sd.entry_number = em.group(1).zfill(3)
            continue

        # Date line
        m = RE_DATE.match(line)
        if m and not sd.date:
            sd.date = m.group(1).strip()
            continue

        # Section header
        m = RE_SECTION.match(line)
        if m:
            section_name = m.group(2).strip()
            # Stop at PRODUCTION NOTES
            if "PRODUCTION NOTES" in section_name.upper():
                _flush_segment()
                stop_parsing = True
                continue
            # New section — flush previous segment
            _flush_segment()
            current_section = section_name
            current_timestamps = m.group(1) or ""
            current_visual = ""
            current_narration_lines = []
            current_code_lines = None
            in_narration = False
            continue

        # Visual hint — starts a new segment within a section
        m = RE_VISUAL.match(line)
        if m:
            # Flush previous segment if it has content
            if current_narration_lines or current_code_lines:
                _flush_segment()
                current_narration_lines = []
                current_code_lines = None
                in_narration = False
            current_visual = m.group(1).rstrip("*").strip()
            continue

        # NARRATION marker
        if RE_NARRATION_MARKER.match(line):
            in_narration = True
            continue

        # Quote lines — treated as narration
        m = RE_QUOTE.match(line)
        if m:
            current_narration_lines.append(m.group(1))
            continue

        # Horizontal rule — ignore
        if line.strip() == "---":
            continue

        # Blank line
        if not line.strip():
            # Preserve paragraph breaks in narration
            if in_narration and current_narration_lines:
                current_narration_lines.append("")
            continue

        # Regular text — collect as narration if we're in a section
        if current_section:
            in_narration = True  # implicit narration after section header
            current_narration_lines.append(line.strip())

    # Flush final segment
    if not stop_parsing:
        _flush_segment()

    # Infer entry number from filename if not found
    if not sd.entry_number:
        fname = path.stem
        m = re.search(r"(\d+)", fname)
        if m:
            sd.entry_number = m.group(1).zfill(3)

    # Infer title from filename if not found
    if not sd.title:
        sd.title = path.stem.replace("_", " ").title()

    return sd
