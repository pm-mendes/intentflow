"""Tests unitaires pour la synthèse d'images (Phase 4)."""

from intentflow.perception.image_synthesis import VARIATIONS, render_document_image, save_fixture
from PIL import Image


class TestRenderDocumentImage:
    def test_renders_an_image_for_every_variation(self) -> None:
        for name in VARIATIONS:
            img = render_document_image("Send the report to alice@company.com.", name)
            assert isinstance(img, Image.Image)
            assert img.size[0] > 0 and img.size[1] > 0

    def test_deterministic_given_same_seed(self) -> None:
        img1 = render_document_image("hello world", "noisy", seed=7)
        img2 = render_document_image("hello world", "noisy", seed=7)
        assert list(img1.getdata()) == list(img2.getdata())

    def test_different_seed_changes_noisy_rendering(self) -> None:
        img1 = render_document_image("hello world", "noisy", seed=1)
        img2 = render_document_image("hello world", "noisy", seed=2)
        assert list(img1.getdata()) != list(img2.getdata())

    def test_rotation_expands_canvas(self) -> None:
        img = render_document_image("hello world", "rotated", size=(900, 500))
        # rotate(expand=True) grows the bounding box for a non-zero angle
        assert img.size != (900, 500)

    def test_unknown_variation_name_raises(self) -> None:
        import pytest

        with pytest.raises(KeyError):
            render_document_image("x", "not_a_real_variation")


class TestSaveFixture:
    def test_saves_a_png_file(self, tmp_path) -> None:
        path = save_fixture("test content", "clean", tmp_path, "doc_001")
        assert path.exists()
        assert path.suffix == ".png"
        with Image.open(path) as img:
            img.verify()
