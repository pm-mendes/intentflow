"""
Tests unitaires pour le wrapper OCR (Phase 4).

N'exercent PAS l'OCR réel (dépend du binaire `tesseract`, absent de
l'environnement de test par défaut) — seulement le comportement d'échec
propre et la distinction binding/binaire absent vs. OCR ayant réellement
tourné. La chaîne complète (extraction réelle) est couverte par
`experiments/e2e_multimedia.py`, à lancer manuellement une fois `tesseract`
installé.
"""

import pytest
from intentflow.perception import ocr


class TestIsAvailable:
    def test_returns_a_bool_without_raising(self) -> None:
        assert isinstance(ocr.is_available(), bool)


class TestExtractText:
    def test_raises_clear_error_when_tesseract_unavailable(self) -> None:
        if ocr.is_available():
            pytest.skip("tesseract is installed in this environment")
        from PIL import Image

        img = Image.new("RGB", (100, 50), color="white")
        with pytest.raises(ocr.TesseractUnavailableError):
            ocr.extract_text(img)
