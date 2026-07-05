"""Generate slide images from ScriptData using Pillow."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import theme
from .parser import ScriptData, Segment


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines or [""]


def _vignette_overlay(width: int, height: int, strength: float = 0.4) -> Image.Image:
    """Create a radial vignette overlay (darker edges)."""
    vignette = Image.new("L", (width, height), 255)
    cx, cy = width / 2, height / 2
    max_dist = math.sqrt(cx ** 2 + cy ** 2)

    # Low-res then upscale for speed
    small_w, small_h = width // 8, height // 8
    small = Image.new("L", (small_w, small_h), 255)
    scx, scy = small_w / 2, small_h / 2
    s_max = math.sqrt(scx ** 2 + scy ** 2)

    for y in range(small_h):
        for x in range(small_w):
            dist = math.sqrt((x - scx) ** 2 + (y - scy) ** 2) / s_max
            val = int(255 * (1.0 - strength * dist * dist))
            small.putpixel((x, y), max(0, val))

    return small.resize((width, height), Image.BILINEAR)


# Cache vignette
_VIGNETTE: Image.Image | None = None


def _get_vignette() -> Image.Image:
    global _VIGNETTE
    if _VIGNETTE is None:
        _VIGNETTE = _vignette_overlay(theme.WIDTH, theme.HEIGHT)
    return _VIGNETTE


def _new_slide() -> Image.Image:
    """Create base slide with background and vignette."""
    img = Image.new("RGB", (theme.WIDTH, theme.HEIGHT), theme.BG)
    # Apply vignette
    vig = _get_vignette()
    # Multiply blend: darken edges
    r, g, b = img.split()
    r = Image.composite(r, Image.new("L", img.size, 0), vig)
    g = Image.composite(g, Image.new("L", img.size, 0), vig)
    b = Image.composite(b, Image.new("L", img.size, 0), vig)
    return Image.merge("RGB", (r, g, b))


def _draw_accent_line(draw: ImageDraw.ImageDraw, y: int) -> int:
    """Draw a horizontal accent line. Returns new Y position."""
    x0 = theme.MARGIN
    x1 = theme.WIDTH - theme.MARGIN
    draw.rectangle(
        [(x0, y), (x1, y + theme.ACCENT_LINE_HEIGHT)],
        fill=theme.ACCENT,
    )
    return y + theme.ACCENT_LINE_HEIGHT + 16


def _render_title_slide(script: ScriptData) -> Image.Image:
    """Title card: entry number + title."""
    img = _new_slide()
    draw = ImageDraw.Draw(img)

    entry_font = _load_font(theme.FONT_SERIF, theme.ENTRY_NUM_SIZE)
    title_font = _load_font(theme.FONT_SERIF_BOLD, theme.TITLE_SIZE)

    entry_text = f"Entry {script.entry_number}"
    if script.date:
        entry_text += f"  —  {script.date}"

    # Center entry number
    bbox = draw.textbbox((0, 0), entry_text, font=entry_font)
    ew = bbox[2] - bbox[0]
    ex = (theme.WIDTH - ew) // 2
    ey = theme.HEIGHT // 2 - 80
    draw.text((ex, ey), entry_text, fill=theme.TEXT_DIM, font=entry_font)

    # Center title
    title_lines = _wrap_text(draw, script.title, title_font, theme.WIDTH - 2 * theme.MARGIN)
    line_h = int(theme.TITLE_SIZE * theme.LINE_SPACING)
    total_h = len(title_lines) * line_h
    ty = ey + 50
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        lw = bbox[2] - bbox[0]
        lx = (theme.WIDTH - lw) // 2
        draw.text((lx, ty), line, fill=theme.TEXT, font=title_font)
        ty += line_h

    return img


def _render_closing_slide(text: str = "Cultivation continues.") -> Image.Image:
    """Closing card."""
    img = _new_slide()
    draw = ImageDraw.Draw(img)
    font = _load_font(theme.FONT_SERIF_ITALIC, theme.CLOSING_SIZE)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (theme.WIDTH - tw) // 2
    y = (theme.HEIGHT - th) // 2
    draw.text((x, y), text, fill=theme.CLOSING_TEXT, font=font)
    return img


def _render_segment_slide(seg: Segment) -> Image.Image:
    """Render a standard narration/code segment slide."""
    img = _new_slide()
    draw = ImageDraw.Draw(img)
    max_w = theme.WIDTH - 2 * theme.MARGIN
    y = theme.CONTENT_TOP

    # Section title (dim, top)
    if seg.section_title:
        sec_font = _load_font(theme.FONT_SERIF, theme.SECTION_TITLE_SIZE)
        draw.text((theme.MARGIN, y), seg.section_title.upper(), fill=theme.TEXT_DIM, font=sec_font)
        y += int(theme.SECTION_TITLE_SIZE * theme.LINE_SPACING) + 8

    # Visual hint (italic)
    if seg.visual_hint:
        hint_font = _load_font(theme.FONT_SERIF_ITALIC, theme.VISUAL_HINT_SIZE)
        hint_lines = _wrap_text(draw, seg.visual_hint, hint_font, max_w)
        line_h = int(theme.VISUAL_HINT_SIZE * theme.LINE_SPACING)
        for line in hint_lines:
            draw.text((theme.MARGIN, y), line, fill=theme.TEXT_ITALIC, font=hint_font)
            y += line_h
        y += 12

    # Accent line
    y = _draw_accent_line(draw, y)

    # Code block (if present)
    if seg.code_block:
        y = _render_code_block(draw, img, seg.code_block, y, max_w)
        y += 16

    # Narration text
    if seg.narration:
        narr_font = _load_font(theme.FONT_SERIF, theme.NARRATION_SIZE)
        # Budget remaining vertical space
        available_h = theme.HEIGHT - y - theme.MARGIN
        narr_lines = _wrap_text(draw, seg.narration.replace("\n\n", " ¶ ").replace("\n", " "), narr_font, max_w)
        line_h = int(theme.NARRATION_SIZE * theme.LINE_SPACING)

        # If text overflows, shrink font
        total_h = len(narr_lines) * line_h
        if total_h > available_h and available_h > 0:
            shrunk_size = max(20, int(theme.NARRATION_SIZE * available_h / total_h))
            narr_font = _load_font(theme.FONT_SERIF, shrunk_size)
            narr_lines = _wrap_text(draw, seg.narration.replace("\n\n", " ¶ ").replace("\n", " "), narr_font, max_w)
            line_h = int(shrunk_size * theme.LINE_SPACING)

        for line in narr_lines:
            if y + line_h > theme.HEIGHT - theme.MARGIN // 2:
                break
            # Handle paragraph marker
            if "¶" in line:
                parts = line.split("¶")
                line = parts[0].strip()
                if line:
                    draw.text((theme.MARGIN, y), line, fill=theme.TEXT, font=narr_font)
                y += line_h
                line = parts[-1].strip() if len(parts) > 1 else ""
                if not line:
                    continue
            draw.text((theme.MARGIN, y), line, fill=theme.TEXT, font=narr_font)
            y += line_h

    return img


def _render_code_block(draw: ImageDraw.ImageDraw, img: Image.Image, code: str, y: int, max_w: int) -> int:
    """Render a code block with background rectangle. Returns new Y."""
    # Auto-shrink code font if lines are too wide
    code_size = theme.CODE_SIZE
    code_lines = code.split("\n")
    pad = theme.CODE_PADDING

    while code_size >= theme.CODE_SIZE_MIN:
        code_font = _load_font(theme.FONT_MONO, code_size)
        max_line_w = 0
        for cl in code_lines:
            bbox = draw.textbbox((0, 0), cl or " ", font=code_font)
            max_line_w = max(max_line_w, bbox[2] - bbox[0])
        if max_line_w <= theme.CODE_MAX_WIDTH:
            break
        code_size -= 1
    else:
        code_font = _load_font(theme.FONT_MONO, theme.CODE_SIZE_MIN)

    line_h = int(code_size * 1.35)
    block_h = len(code_lines) * line_h + 2 * pad
    # Cap block height
    max_block_h = theme.HEIGHT // 2
    block_h = min(block_h, max_block_h)

    x0 = theme.MARGIN
    x1 = theme.WIDTH - theme.MARGIN

    # Background rectangle
    overlay = Image.new("RGB", img.size, theme.CODE_BG)
    mask = Image.new("L", img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([(x0, y), (x1, y + block_h)], radius=8, fill=200)
    img.paste(Image.composite(overlay, img, mask), (0, 0))

    # Border
    draw.rounded_rectangle(
        [(x0, y), (x1, y + block_h)],
        radius=8,
        outline=theme.CODE_BORDER,
        width=1,
    )

    # Code text
    cy = y + pad
    for cl in code_lines:
        if cy + line_h > y + block_h:
            break
        draw.text((x0 + pad, cy), cl, fill=theme.CODE_FG, font=code_font)
        cy += line_h

    return y + block_h


def generate_slides(
    script: ScriptData,
    output_dir: Path,
) -> list[Path]:
    """Generate PNG slides for all segments plus title and closing.

    Returns list of PNG paths in order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Title slide
    title_path = output_dir / "seg_000_title.png"
    _render_title_slide(script).save(str(title_path))
    paths.append(title_path)
    print(f"  Slide [title] saved")

    # Content slides
    for seg in script.segments:
        p = output_dir / f"seg_{seg.segment_index:03d}.png"
        _render_segment_slide(seg).save(str(p))
        paths.append(p)
        print(f"  Slide [{seg.segment_index:03d}] {seg.section_title[:40]}...")

    # Closing slide
    closing_path = output_dir / f"seg_{len(script.segments):03d}_closing.png"
    _render_closing_slide().save(str(closing_path))
    paths.append(closing_path)
    print(f"  Slide [closing] saved")

    return paths
