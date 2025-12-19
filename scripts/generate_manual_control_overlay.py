"""Generate the manual-control overlay illustration without storing binaries.

Run from the repository root:

    python scripts/generate_manual_control_overlay.py

The output is saved to ``images/manual_control_overlay.png``. Pillow is the
only dependency and is intentionally kept as a script-only requirement.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "images" / "manual_control_overlay.png"


def _load_fonts():
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        title_font = label_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    return title_font, label_font, body_font


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    # Pillow >= 10: use textbbox; works across modern versions.
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _font_text_size(font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
    # For places where you were using font.getsize(...)
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def _draw_button(draw: ImageDraw.ImageDraw, box, label, font, accent=None):
    x0, y0, x1, y1 = box
    fill = accent if accent is not None else (48, 87, 146)
    outline = (210, 224, 239)
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=2)

    text_w, text_h = _text_size(draw, label, font)
    draw.text(
        (x0 + (x1 - x0 - text_w) / 2, y0 + (y1 - y0 - text_h) / 2),
        label,
        font=font,
        fill=(240, 248, 255),
    )


def _draw_section_title(draw, text, origin, font):
    x, y = origin
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    text_w, text_h = _font_text_size(font, text)
    underline_y = y + text_h + 2
    draw.line((x, underline_y, x + text_w, underline_y), fill=(120, 170, 255), width=2)
    return underline_y + 10


def _draw_movement_pad(draw, origin, font):
    x, y = origin
    pad_size = 230
    pad_box = (x, y, x + pad_size, y + pad_size)
    draw.rounded_rectangle(pad_box, radius=18, fill=(25, 36, 52), outline=(88, 110, 140), width=2)

    spacing = 10
    btn_w = btn_h = 66
    cx = x + pad_size // 2 - btn_w // 2
    cy = y + pad_size // 2 - btn_h // 2

    _draw_button(draw, (cx, y + spacing, cx + btn_w, y + spacing + btn_h), "↑", font)
    _draw_button(draw, (cx, cy, cx + btn_w, cy + btn_h), "•", font, accent=(70, 100, 140))
    _draw_button(draw, (cx, y + pad_size - spacing - btn_h, cx + btn_w, y + pad_size - spacing), "↓", font)
    _draw_button(draw, (x + spacing, cy, x + spacing + btn_w, cy + btn_h), "←", font)
    _draw_button(draw, (x + pad_size - spacing - btn_w, cy, x + pad_size - spacing, cy + btn_h), "→", font)

    caption = "WASD / Arrow keys"
    text_w, text_h = _text_size(draw, caption, font)
    draw.text(
        (x + (pad_size - text_w) / 2, y + pad_size - text_h - 8),
        caption,
        font=font,
        fill=(205, 216, 230),
    )


def _draw_look_pad(draw, origin, font):
    x, y = origin
    pad_w, pad_h = 260, 180
    pad_box = (x, y, x + pad_w, y + pad_h)
    draw.rounded_rectangle(pad_box, radius=18, fill=(25, 36, 52), outline=(88, 110, 140), width=2)

    spacing = 10
    btn_w = 70
    btn_h = 52

    _draw_button(draw, (x + spacing, y + spacing, x + spacing + btn_w, y + spacing + btn_h), "Look ↑", font)
    _draw_button(
        draw,
        (x + spacing, y + pad_h - spacing - btn_h, x + spacing + btn_w, y + pad_h - spacing),
        "Look ↓",
        font,
    )
    _draw_button(
        draw,
        (x + pad_w - spacing - btn_w, y + spacing, x + pad_w - spacing, y + spacing + btn_h),
        "Turn →",
        font,
    )
    _draw_button(
        draw,
        (x + pad_w - spacing - btn_w, y + pad_h - spacing - btn_h, x + pad_w - spacing, y + pad_h - spacing),
        "Turn ←",
        font,
    )

    caption = "Mouse yaw/pitch mirrored by buttons"
    text_w, text_h = _text_size(draw, caption, font)
    draw.text(
        (x + (pad_w - text_w) / 2, y + pad_h - text_h - 8),
        caption,
        font=font,
        fill=(205, 216, 230),
    )


def render_overlay(path: Path = OUTPUT_PATH):
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), (12, 16, 28))
    draw = ImageDraw.Draw(image)

    title_font, label_font, body_font = _load_fonts()

    # Top banner
    banner_h = 110
    banner_box = (40, 30, width - 40, banner_h + 30)
    draw.rounded_rectangle(banner_box, radius=18, fill=(24, 36, 56), outline=(60, 96, 148), width=2)
    banner_text = "Manual control HUD: click to drive if keyboard/mouse are busy"
    text_w, text_h = _text_size(draw, banner_text, title_font)
    draw.text(
        (banner_box[0] + (banner_box[2] - banner_box[0] - text_w) / 2, banner_box[1] + (banner_h - text_h) / 2),
        banner_text,
        font=title_font,
        fill=(235, 242, 255),
    )

    # Movement section
    section_y = banner_box[3] + 30
    section_y = _draw_section_title(draw, "Movement", (60, section_y), title_font)
    _draw_movement_pad(draw, (60, section_y), label_font)

    # Look section
    section_y = banner_box[3] + 30
    section_y = _draw_section_title(draw, "Camera + turning", (430, section_y), title_font)
    _draw_look_pad(draw, (430, section_y), label_font)

    # Tips on the right
    tips_x = 770
    tips_y = section_y
    tips_box = (tips_x, tips_y, width - 60, tips_y + 360)
    draw.rounded_rectangle(tips_box, radius=18, fill=(22, 32, 48), outline=(60, 96, 148), width=2)
    tips = [
        "Overlay buttons mirror WASD/arrow movement and mouse look.",
        "Use them for trackpads, demos, or accessibility.",
        "Toggle overlay: --show-controls / --no-show-controls",
        "Hide HUD entirely: --hide-hud",
    ]
    padding = 24
    cursor_y = tips_box[1] + padding
    for tip in tips:
        draw.text((tips_box[0] + padding, cursor_y), f"• {tip}", font=body_font, fill=(215, 226, 240))
        _, tip_h = _font_text_size(body_font, tip)
        cursor_y += tip_h + 12

    # Footer caption
    footer = "Generated with scripts/generate_manual_control_overlay.py"
    footer_w, footer_h = _text_size(draw, footer, body_font)
    draw.text((width - footer_w - 24, height - footer_h - 24), footer, font=body_font, fill=(150, 170, 190))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"Saved overlay illustration to {path}")


def main():
    render_overlay()


if __name__ == "__main__":
    main()
