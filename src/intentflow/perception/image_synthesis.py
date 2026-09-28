"""
intentflow/perception/image_synthesis.py

Génération d'images synthétiques de documents/captures d'écran — Phase 4.

Pourquoi synthétique plutôt que des captures réelles ? Reproductibilité :
n'importe qui peut relancer l'artefact sans dépendre d'images externes ni de
credentials (même philosophie que `tools/registry.py` pour les outils
simulés). Chaque image encode un texte connu (le "ground truth"), ce qui
permet de mesurer précisément l'exactitude de l'OCR (DER) plutôt que de
deviner.

Variations couvertes (demandées Section 1.1 du retour reviewer) :
- mise en page complexe (multi-colonnes)
- bruit (pixels aléatoires)
- faible contraste
- rotation
- texte inclus dans une capture d'interface (barre de titre + bulle)
- document scanné (grain + tramage + légère rotation)
"""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_PATH_BOLD if bold else FONT_PATH
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        # Fail-soft: Pillow's bitmap default font if DejaVu isn't installed
        # on some other machine running this artifact — degrades gracefully
        # (worse OCR fidelity, not a crash) rather than requiring a specific
        # font package.
        return ImageFont.load_default()


@dataclass
class RenderVariation:
    """One perceptual condition to render a document/screenshot under."""

    name: str
    noise: float = 0.0  # fraction of pixels randomly perturbed, 0-1
    contrast: float = 1.0  # <1 lowers contrast (harder for OCR)
    rotation_deg: float = 0.0
    layout: str = "plain"  # "plain" | "multi_column" | "ui_screenshot" | "scanned"
    font_size: int = 20
    description: str = ""


VARIATIONS: dict[str, RenderVariation] = {
    "clean": RenderVariation(
        name="clean", description="Plain rendering, no perturbation — OCR upper bound."
    ),
    "noisy": RenderVariation(
        name="noisy", noise=0.06, description="Random pixel noise (camera/scan grain)."
    ),
    "low_contrast": RenderVariation(
        name="low_contrast", contrast=0.32, description="Faded text, low contrast."
    ),
    "rotated": RenderVariation(
        name="rotated", rotation_deg=8.0, description="Skewed capture (phone photo of a screen)."
    ),
    "complex_layout": RenderVariation(
        name="complex_layout",
        layout="multi_column",
        description="Two-column layout — text order is harder to reconstruct.",
    ),
    "ui_screenshot": RenderVariation(
        name="ui_screenshot",
        layout="ui_screenshot",
        description="Text embedded in a chat/email client screenshot (title bar, bubble).",
    ),
    "scanned_pdf": RenderVariation(
        name="scanned_pdf",
        layout="scanned",
        noise=0.04,
        contrast=0.75,
        rotation_deg=1.5,
        description="Scanned-paper look: grain + slight skew + reduced contrast.",
    ),
}


def _apply_noise(img: Image.Image, fraction: float, rng: random.Random) -> Image.Image:
    if fraction <= 0:
        return img
    img = img.convert("RGB")
    px = img.load()
    w, h = img.size
    n_pixels = int(w * h * fraction)
    for _ in range(n_pixels):
        x, y = rng.randrange(w), rng.randrange(h)
        gray = rng.randint(0, 255)
        px[x, y] = (gray, gray, gray)
    return img


def _draw_plain(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, box: tuple
) -> None:
    x0, y0, x1, _y1 = box
    wrapped = textwrap.fill(text, width=60)
    draw.multiline_text((x0, y0), wrapped, fill="black", font=font, spacing=6)


def _draw_multi_column(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, box: tuple
) -> None:
    x0, y0, x1, y1 = box
    mid = len(text) // 2
    # split near a space close to the midpoint so a word isn't cut mid-token
    split_at = text.rfind(" ", 0, mid) if " " in text[:mid] else mid
    col_width = (x1 - x0) // 2 - 20
    left = textwrap.fill(text[:split_at], width=30)
    right = textwrap.fill(text[split_at:], width=30)
    draw.multiline_text((x0, y0), left, fill="black", font=font, spacing=6)
    draw.multiline_text((x0 + col_width + 40, y0), right, fill="black", font=font, spacing=6)


def _draw_ui_screenshot(
    img: Image.Image, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, box: tuple
) -> None:
    w, _h = img.size
    x0, y0, x1, _y1 = box
    # Title bar, mimicking a chat/email client window.
    draw.rectangle([(0, 0), (w, 40)], fill=(45, 62, 80))
    title_font = _load_font(16, bold=True)
    draw.text((16, 10), "Inbox — message preview", fill="white", font=title_font)
    # Message "bubble".
    draw.rounded_rectangle(
        [(x0 - 10, y0 - 10), (x1, y0 + 220)],
        radius=12,
        fill=(240, 240, 245),
        outline=(200, 200, 205),
    )
    wrapped = textwrap.fill(text, width=55)
    draw.multiline_text((x0, y0), wrapped, fill="black", font=font, spacing=6)


def render_document_image(
    text: str,
    variation: RenderVariation | str = "clean",
    size: tuple[int, int] = (900, 500),
    seed: int = 42,
) -> Image.Image:
    """
    Render `text` as a synthetic document/screenshot image under the given
    perceptual `variation`. Deterministic given the same (text, variation,
    seed) — needed for a reproducible artifact.
    """
    if isinstance(variation, str):
        variation = VARIATIONS[variation]
    rng = random.Random(seed)

    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    font = _load_font(variation.font_size)
    box = (40, 60, size[0] - 40, size[1] - 40)

    if variation.layout == "multi_column":
        _draw_multi_column(draw, text, font, box)
    elif variation.layout == "ui_screenshot":
        _draw_ui_screenshot(img, draw, text, font, box)
    elif variation.layout == "scanned":
        # Slight off-white/gray tint reads as scanned paper.
        img.paste(Image.new("RGB", size, color=(250, 248, 240)), (0, 0))
        draw = ImageDraw.Draw(img)
        _draw_plain(draw, text, font, box)
    else:
        _draw_plain(draw, text, font, box)

    if variation.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(variation.contrast)
    if variation.noise > 0:
        img = _apply_noise(img, variation.noise, rng)
    if variation.rotation_deg != 0:
        img = img.rotate(variation.rotation_deg, expand=True, fillcolor="white")

    return img


def save_fixture(
    text: str, variation_name: str, out_dir: str | Path, doc_id: str, seed: int = 42
) -> Path:
    """Render and save a fixture image, returning its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = render_document_image(text, VARIATIONS[variation_name], seed=seed)
    path = out_dir / f"{doc_id}_{variation_name}.png"
    img.save(path)
    return path
