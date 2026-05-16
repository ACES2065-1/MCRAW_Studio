"""One-shot generator for MotionCam Tools' app icon.

Run: python tools/gen_icon.py
Outputs: gui/icon.png (256), gui/icon.ico (multi-size 16/32/48/64/128/256).

The design: a rounded square in MotionCam orange with a stylised aperture-blade
mark in the center (six blades arranged as a hexagonal star — a nod to camera
iris diaphragms without literally rendering one).
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


# ----- design constants -----
ORANGE = (232, 116, 59)      # #e8743b — matches QSS accent
ORANGE_DARK = (180, 80, 30)
BG_INNER = (28, 16, 8)        # subtle dark center for contrast
WHITE = (255, 255, 255)


def render(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Rounded-square background
    pad = max(1, size // 24)
    radius = max(2, size // 6)
    d.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=radius, fill=ORANGE,
    )

    # A faint dark inner glow for depth
    cx = cy = size / 2.0
    inner_r = size * 0.34
    d.ellipse(
        [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
        fill=BG_INNER,
    )

    # Six aperture blades: triangular wedges pointing inward, rotated 60° apart.
    blade_color = ORANGE
    blade_outer = size * 0.42
    blade_inner = size * 0.10
    blade_half_angle = math.radians(28)   # blade width in radians
    for i in range(6):
        center_angle = math.radians(-90 + i * 60)
        a1 = center_angle - blade_half_angle
        a2 = center_angle + blade_half_angle
        pts = [
            (cx + blade_outer * math.cos(a1), cy + blade_outer * math.sin(a1)),
            (cx + blade_outer * math.cos(a2), cy + blade_outer * math.sin(a2)),
            (cx + blade_inner * math.cos(center_angle + math.radians(20)),
             cy + blade_inner * math.sin(center_angle + math.radians(20))),
            (cx + blade_inner * math.cos(center_angle - math.radians(20)),
             cy + blade_inner * math.sin(center_angle - math.radians(20))),
        ]
        d.polygon(pts, fill=blade_color, outline=ORANGE_DARK)

    # Bright center pinhole
    pin_r = size * 0.06
    d.ellipse(
        [cx - pin_r, cy - pin_r, cx + pin_r, cy + pin_r],
        fill=WHITE,
    )

    # Light antialias smoothing for very small sizes (16/32 lose detail otherwise)
    if size <= 32:
        img = img.filter(ImageFilter.SMOOTH)

    return img


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "gui"
    out_dir.mkdir(exist_ok=True)

    # Render 256 master and save as PNG (used by QApplication.setWindowIcon).
    master = render(256)
    png_path = out_dir / "icon.png"
    master.save(png_path, "PNG")
    print(f"wrote {png_path}")

    # Multi-resolution ICO for the bundled .exe taskbar/file icon.
    # Pillow's ICO writer takes the master image and a list of (w,h) tuples,
    # then resizes the master to each. Pass distinct images via the file's
    # internal mode by saving each at native render resolution then letting
    # Pillow pick those entries.
    sizes = [16, 24, 32, 48, 64, 128, 256]
    ico_path = out_dir / "icon.ico"
    master.save(ico_path, format="ICO", sizes=[(s, s) for s in sizes])
    print(f"wrote {ico_path}  (sizes: {sizes})")


if __name__ == "__main__":
    main()
