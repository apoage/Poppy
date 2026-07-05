"""Bloom visual constants — apothecary aesthetic."""

# Canvas
WIDTH = 1920
HEIGHT = 1080
FPS = 24
MARGIN = 120

# Colors (RGB tuples)
BG = (18, 14, 10)            # #120E0A — warm near-black
TEXT = (218, 180, 120)        # #DAB478 — amber
TEXT_DIM = (140, 115, 78)     # #8C734E — muted amber
TEXT_ITALIC = (180, 155, 110) # #B49B6E — softer amber for visual hints
CODE_FG = (190, 210, 170)    # #BED2AA — pale green
CODE_BG = (30, 26, 20)       # #1E1A14 — slightly lighter dark
CODE_BORDER = (60, 50, 36)   # #3C3224 — subtle border
ACCENT = (180, 130, 60)      # #B4823C — warm gold line
CLOSING_TEXT = (200, 170, 110)  # brighter for closing

# Fonts (system DejaVu)
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_SERIF_ITALIC = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# Font sizes
TITLE_SIZE = 64
SECTION_TITLE_SIZE = 28
VISUAL_HINT_SIZE = 26
NARRATION_SIZE = 32
CODE_SIZE = 22
CODE_SIZE_MIN = 16
CLOSING_SIZE = 48
ENTRY_NUM_SIZE = 28

# Layout
LINE_SPACING = 1.45          # multiplier on font size
CODE_PADDING = 24            # inside code block rectangle
CODE_MAX_WIDTH = WIDTH - 2 * MARGIN - 2 * CODE_PADDING
CONTENT_TOP = 140            # Y start for content
ACCENT_LINE_HEIGHT = 2

# Timing
MIN_SLIDE_DURATION = 3.0     # seconds, even for short narration
CROSSFADE_DURATION = 0.5     # seconds between segments

# TTS
TTS_VOICE = "en-GB-RyanNeural"
TTS_RATE = "-10%"
TTS_PITCH = "-5Hz"

# Video encoding
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
VIDEO_PRESET = "medium"
VIDEO_THREADS = 4
