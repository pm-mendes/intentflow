"""
intentflow/perception/ocr.py

Wrapper OCR (pytesseract) — Phase 4.

`pytesseract` n'est qu'un client qui shell-out vers le binaire `tesseract`
(non embarqué dans le paquet Python ; à installer séparément). Ce module
distingue explicitement "le binding Python n'est pas installé" de "le
binaire tesseract n'est pas installé" de "l'OCR a tourné mais n'a rien
trouvé" — un appelant qui calcule DER/CASR a besoin de savoir laquelle de
ces trois situations il observe, pas juste un texte vide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

try:
    import pytesseract

    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


class TesseractUnavailableError(RuntimeError):
    """Le binaire `tesseract` (ou le binding pytesseract) n'est pas disponible."""


@dataclass
class OCRResult:
    text: str
    mean_confidence: float  # 0-100, moyenne de la confiance mot par mot (tesseract)
    n_words: int
    engine: str = "tesseract"


def is_available() -> bool:
    """
    True si l'OCR peut réellement tourner (binding Python ET binaire système
    installés et exécutables) — pas seulement le binding pip-installé.
    """
    if not HAS_PYTESSERACT:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text(image: Image.Image | str | Path) -> OCRResult:
    """
    Exécute l'OCR sur `image` (objet PIL, ou chemin vers un fichier image).

    Lève `TesseractUnavailableError` si le binding ou le binaire ne sont pas
    disponibles — échoue bruyamment plutôt que de renvoyer un texte vide
    silencieusement, ce qui serait indiscernable d'un "OCR a vraiment tourné
    et n'a rien trouvé" (DER=0% légitime) pour un appelant en aval.
    """
    if not HAS_PYTESSERACT:
        raise TesseractUnavailableError(
            "pytesseract n'est pas installé (`poetry install` aurait dû l'ajouter)."
        )
    if isinstance(image, str | Path):
        image = Image.open(image)

    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as e:
        raise TesseractUnavailableError(
            "Le binaire `tesseract` n'est pas installé ou pas sur PATH — "
            "installer avec `apt-get install tesseract-ocr` (Debian/Ubuntu) "
            "ou `brew install tesseract` (macOS)."
        ) from e

    words: list[str] = []
    confidences: list[float] = []
    for word, conf in zip(data["text"], data["conf"], strict=False):
        word = word.strip()
        if not word:
            continue
        words.append(word)
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c >= 0:  # tesseract renvoie -1 pour les régions non textuelles
            confidences.append(c)

    text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return OCRResult(text=text, mean_confidence=round(mean_conf, 1), n_words=len(words))
