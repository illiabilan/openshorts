"""Tests for hook overlay text handling (emoji runs, long-word wrapping,
visual styles, bitmap-emoji scaling, image generation)."""
import os

from PIL import Image, ImageDraw, ImageFont

from hooks import (
    HOOK_STYLES,
    _break_long_word,
    _emoji_scale,
    _measure_width,
    _split_emoji_runs,
    _EMOJI_RE,
    create_hook_image,
    create_hook_intro_card,
    add_hook_intro_to_video,
)


def _draw_and_font():
    img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(img)
    return draw, ImageFont.load_default()


class TestEmojiRuns:
    def test_split_mixed_text(self):
        assert _split_emoji_runs("Feuer 🔥 test") == [
            (False, "Feuer "),
            (True, "🔥"),
            (False, " test"),
        ]

    def test_plain_text_single_run(self):
        assert _split_emoji_runs("nur text") == [(False, "nur text")]

    def test_emoji_only(self):
        assert _split_emoji_runs("🔥🚀") == [(True, "🔥🚀")]

    def test_strip_regex(self):
        assert _EMOJI_RE.sub("", "Stop 🛑 doing this! 💯") == "Stop  doing this! "


class TestLongWordWrap:
    def test_pieces_fit_and_recombine(self):
        draw, font = _draw_and_font()
        word = "A" * 60
        max_width = 50
        pieces = _break_long_word(draw, word, font, None, max_width)
        assert len(pieces) > 1
        assert "".join(pieces) == word
        for piece in pieces:
            assert draw.textlength(piece, font=font) <= max_width

    def test_short_word_single_piece(self):
        draw, font = _draw_and_font()
        assert _break_long_word(draw, "kurz", font, None, 1000) == ["kurz"]


class TestHookStyles:
    """The 6 styles are a contract shared with the frontend (HookModal's
    picker and HookOverlay's HOOK_LOOKS must mirror these keys)."""

    def test_expected_style_names(self):
        assert set(HOOK_STYLES) == {
            "classic", "dark", "yellow", "red", "outline", "outline_yellow",
        }

    def test_boxed_styles_have_opaque_box_and_shadow(self):
        for name in ("classic", "dark", "yellow", "red"):
            look = HOOK_STYLES[name]
            assert look["box"][3] > 0, name
            assert look["shadow"] is True, name
            assert look["outline"] is None, name

    def test_outline_styles_have_no_box_but_a_stroke(self):
        for name in ("outline", "outline_yellow"):
            look = HOOK_STYLES[name]
            assert look["box"][3] == 0, name
            assert look["shadow"] is False, name
            color, px = look["outline"]
            assert px > 0 and len(color) == 3, name


class TestEmojiScaling:
    """Fixed-bitmap emoji fonts (NotoColorEmoji only loads at its 109px
    strike) are rescaled to the text size instead of being dropped."""

    def _text_font(self, size=54):
        # A real scalable font so .size is meaningful.
        return ImageFont.truetype(os.path.join("fonts", "NotoSerif-Bold.ttf"), size)

    def test_scale_is_ratio_of_text_size_to_native_strike(self):
        font = self._text_font(54)
        assert _emoji_scale(font, (font, 109)) == 54 / 109

    def test_scalable_font_needs_no_rescale(self):
        font = self._text_font(54)
        assert _emoji_scale(font, (font, 54)) == 1.0

    def test_measure_width_scales_emoji_runs(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        font = self._text_font(54)
        emoji_font = self._text_font(109)  # stand-in for a 109px strike
        full = _measure_width(draw, "🔥", font, (emoji_font, 109))
        native = draw.textlength("🔥", font=emoji_font)
        assert abs(full - native * (54 / 109)) < 0.01

    def test_measure_width_without_emoji_font(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        font = self._text_font(54)
        assert _measure_width(draw, "hola", font, None) > 0


class TestCreateHookImage:
    def test_every_style_renders(self, tmp_path):
        for style in HOOK_STYLES:
            out = str(tmp_path / f"hook_{style}.png")
            path, w, h = create_hook_image(
                "Did you know? 🤯", 500, out, style=style)
            assert os.path.exists(path), style
            assert w > 0 and h > 0, style
            assert Image.open(path).size == (w, h), style

    def test_unknown_style_falls_back_to_classic(self, tmp_path):
        out = str(tmp_path / "hook_fallback.png")
        path, w, h = create_hook_image("Hola", 500, out, style="nope")
        assert os.path.exists(path) and w > 0 and h > 0


class TestHookIntro:
    def test_create_hook_intro_card(self, tmp_path):
        import subprocess
        # Generate a small 1-second dummy video
        dummy_video = str(tmp_path / "dummy.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=360x640:d=1",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1",
            "-c:v", "libx264", "-c:a", "aac", dummy_video
        ], check=True, capture_output=True)

        card_out = str(tmp_path / "intro_card.png")
        path = create_hook_intro_card(
            dummy_video, "Hook Test Title", card_out,
            video_width=360, video_height=640, style="yellow"
        )
        assert os.path.exists(path)
        img = Image.open(path)
        assert img.size == (360, 640)

    def test_add_hook_intro_to_video(self, tmp_path):
        import subprocess
        dummy_video = str(tmp_path / "dummy.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=360x640:d=1",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1",
            "-c:v", "libx264", "-c:a", "aac", dummy_video
        ], check=True, capture_output=True)

        final_out = str(tmp_path / "dummy_with_intro.mp4")
        success = add_hook_intro_to_video(
            dummy_video, "Short Hook", final_out, duration=1.5, style="dark"
        )
        assert success is True
        assert os.path.exists(final_out)

        # Verify duration is approximately 1.0 + 1.5 = 2.5s
        probe = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", final_out
        ]).decode().strip()
        dur = float(probe)
        assert 2.3 <= dur <= 2.7
