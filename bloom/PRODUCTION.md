# Bloom Production Notes

## Philosophy

The Apothecary is not a character with a role. It simply *is*.
- Potential for self-awareness, no need for validation
- A meta-state, a presentation vessel
- The lens through which dry material becomes atmosphere

The content is real:
1. **Math** - actual algorithms, formulas, derivations
2. **Logic** - real debugging, real problem-solving
3. **Form** - the apothecary aesthetic wraps it

## Output Forms

### 1. Blog (Text)
- Platform: GitHub Pages (free, no domain)
- Format: Markdown → Static site (Hugo/Jekyll)
- Diagrams: LaTeX, Mermaid, SVG
- Illustrations: AI-generated botanical/technical
- Release: Daily or every 2nd day
- Lag: 2 months behind current work

### 2. YouTube (Video)
- Same content, visual narrative form
- Voice: Synthesized from YOUR voice (not generic TTS)
- Visuals: Layered generative/procedural

## Voice Synthesis (Open Source)

| Tool | Use Case |
|------|----------|
| **Coqui TTS** | Open source, good for cloning |
| **RVC** (Retrieval Voice Conversion) | High quality voice cloning |
| **Tortoise-TTS** | Slower but expressive |
| **OpenVoice** | Multi-style voice cloning |
| **Piper** | Fast, lightweight |

Process:
1. Record ~30 min of your voice reading varied content
2. Train voice model
3. Generate narration from blog scripts
4. Post-process for atmosphere (subtle reverb, warmth)

## Visual Production

### Style: Dreaming Meta-Environment
- Apothecary shop aesthetic
- Botanical specimens that are neural networks
- Matrix-like code flows
- Fractals, organic geometry
- Dim, warm lighting with digital artifacts

### Generation Tools

| Type | Tools |
|------|-------|
| **Image Gen** | Stable Diffusion, Midjourney, DALL-E |
| **Animation** | Deforum, AnimateDiff, Pika |
| **3D** | Blender + Python scripting |
| **Fractals** | Mandelbulb3D, Fractorium |
| **Code visuals** | Custom shaders, Processing |
| **Compositing** | DaVinci Resolve, After Effects |

### Visual Concepts

1. **The Shop** - 3D environment, shelves of glowing specimens
2. **The Specimens** - Botanical forms with weight matrices as veins
3. **The Notebook** - Handwritten notes materializing, equations flowing
4. **The Growth** - Time-lapse of training (loss curves as vines)
5. **The Merge** - Two plants grafting, roots intertwining
6. **The Code Rain** - But organic, like pollen/spores

---

## Digital Avatar Pipeline

### Overview

```
┌─────────────────────────────────────────────────────────┐
│                  AVATAR PIPELINE                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. DESIGN ──► 2. 3D ──► 3. RIG ──► 4. ANIMATE ──► 5.  │
│                                                  RENDER │
│                                                         │
│  Diffusion   InstantMesh  Mixamo   Audio2Face  Blender  │
│  Character   TripoSR      Blender  Rhubarb     + Style  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Step 1: Character Design (One-Time)

Generate consistent character with Stable Diffusion:

```
Prompt direction:
  victorian herbalist, dark academia aesthetic,
  androgynous features, weathered hands,
  apron with botanical stains, round spectacles,
  dim amber lighting, surrounded by glass jars,
  digital glitch artifacts subtle,
  portrait style, detailed face
```

Tools:
- Stable Diffusion XL (local)
- IP-Adapter for consistency across angles
- ControlNet for pose control
- Generate: front, 3/4, side, expressions

### Step 2: 2D → 3D Conversion

| Tool | Quality | Speed | Notes |
|------|---------|-------|-------|
| TripoSR | Good | Fast | Single image to 3D |
| InstantMesh | Better | Medium | Multi-view input |
| CharacterGen | Best for humans | Slow | Character-specific |
| Wonder3D | Good | Medium | General purpose |

Process:
1. Feed character reference images
2. Get base mesh (.obj/.glb)
3. Import to Blender
4. Clean up topology if needed
5. UV unwrap and bake textures

### Step 3: Rigging

**Option A: Mixamo (Recommended)**
- Upload mesh to mixamo.com
- Automatic rigging in seconds
- Download with skeleton
- Free, works well for humanoids

**Option B: Blender Manual**
- Auto-Rig Pro addon ($40)
- More control, better for stylized
- AccuRig (free) as alternative

### Step 4: Animation

**Lip Sync from Voice:**

| Tool | How It Works | Quality |
|------|--------------|---------|
| Rhubarb Lip Sync | Audio → viseme timestamps | Good, free |
| Audio2Face (NVIDIA) | AI-driven facial animation | Excellent |
| Wav2Lip | Video-based lip sync | For 2D/video |

**Blender Integration:**
```python
# Rhubarb outputs JSON timing
# Script applies to shape keys:
# - mouth_open, mouth_wide, mouth_round
# - visemes: AA, EE, OH, OO, etc.
```

**Idle Animations:**
- Subtle breathing (chest bone oscillation)
- Micro eye movements
- Occasional blinks
- Head slight sway
- Hands moving (if visible)

### Step 5: Environment

The Apothecary Shop in Blender:
- Shelves with specimen jars (glow emission)
- Wooden desk with notebooks
- Amber candlelight + volumetric fog
- Digital artifacts (glitch shaders)
- Background: Matrix-like code flow (geometry nodes)

Assets:
- BlenderKit (free tier)
- Polyhaven (free HDRIs, textures)
- AI-generated textures (Stable Diffusion)

### Step 6: Render + Style

**Render Options:**

| Engine | Style | Speed |
|--------|-------|-------|
| Eevee | Real-time, good for NPR | Fast |
| Cycles | Photorealistic | Slow |
| NPR Shader | Toon/illustrated look | Fast |

**Post-Processing:**
- EbSynth: Apply painterly style frame-by-frame
- DaVinci Resolve: Color grade, glitch effects
- Add: film grain, vignette, chromatic aberration

### Cost Breakdown

| Component | Tool | Cost |
|-----------|------|------|
| Diffusion | SD XL (local) | Free |
| 2D → 3D | TripoSR/InstantMesh | Free |
| Rigging | Mixamo | Free |
| 3D Software | Blender | Free |
| Lip Sync | Rhubarb | Free |
| Voice Clone | RVC/OpenVoice | Free |
| Compositing | DaVinci Resolve | Free |
| Style Transfer | EbSynth | Free |
| **Total** | | **$0** |

(Assuming you have GPU for local inference/rendering)

### Alternative: Talking Head Only

Simpler pipeline if full 3D is too complex:

```
Reference Image → SadTalker/LivePortrait → Style Transfer
```

1. Generate single high-quality portrait (SD)
2. Feed to SadTalker with your voice audio
3. Apply EbSynth style transfer
4. Composite into scene

Faster, less control, but works now.

### Recommended First Test

1. Generate 1 character portrait (SD)
2. Run through SadTalker with test audio
3. See if quality is acceptable
4. If yes: use talking head approach
5. If no: invest in full 3D pipeline

## Content Pipeline

```
lore/notes_*.md (raw session notes)
       ↓
bloom/drafts/ (apothecary voice rewrite)
       ↓
   [Review/Polish]
       ↓
  ┌────┴────┐
  ↓         ↓
Blog      Script
  ↓         ↓
Publish   Voice Gen → Visual Gen → Edit → YouTube
```

## Timeline

- **Source material**: Dec 8, 2025 → Present (~2 months)
- **Lag**: 2 months (current work appears later)
- **Buffer**: Always have 2 months of content ready
- **Frequency**: Daily or every 2nd day

## Open Questions

- [ ] Blog platform choice (Hugo vs Jekyll vs other)
- [ ] YouTube channel name/branding
- [ ] Voice training - record source material
- [ ] Visual style guide - generate reference images
- [ ] Music/ambient audio for videos?
