import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import patheffects as pe
from PIL import Image, ImageFilter, ImageEnhance

# ------------------------------------------------------------
# configuration
# ------------------------------------------------------------
IMG_PATH = "/home/zanot/projects/tesseract2026/visual_abstract/t-logo-dark.png"
OUT_PATH = "fide_horizontal_diagram.png"

BG = "#000000"
BOX_FACE = "#0b0b0d"
BOX_EDGE = (1, 1, 1, 0.20)
TXT = "white"
SUBTXT = (1, 1, 1, 0.72)

NEON_MAGENTA = "#ff00b8"
NEON_CYAN = "#00e5ff"

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def prepare_motif(path):
    """
    Load the provided image and soften it so it works as a subtle in-box motif.
    """
    img = Image.open(path).convert("RGBA")

    # Slight blur so the source text doesn't fight with diagram text
    img = img.filter(ImageFilter.GaussianBlur(radius=1.4))

    # Slightly dim it
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(0.90)
    img = rgb.convert("RGBA")

    return img


def add_box(ax, x, y, w, h, title, formula="", subtitle="",
            title_size=14, formula_size=18, subtitle_size=10,
            facecolor=BOX_FACE, edgecolor=BOX_EDGE, lw=1.3,
            image=None, image_alpha=0.28, image_side="right",
            accent=None):
    """
    Draw a rounded box with optional decorative image inside.
    Coordinates are in data units.
    """
    # Main box
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.16",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=3
    )
    ax.add_patch(patch)

    # Optional accent edge
    if accent is not None:
        accent_patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.16",
            facecolor=(0, 0, 0, 0),
            edgecolor=accent,
            linewidth=1.7,
            zorder=4,
            alpha=0.65
        )
        ax.add_patch(accent_patch)

    # Optional motif image inside the box
    if image is not None:
        if image_side == "right":
            ix = x + 0.56 * w
            iy = y + 0.12 * h
            iw = 0.36 * w
            ih = 0.76 * h
            text_x = x + 0.08 * w
            text_ha = "left"
        elif image_side == "left":
            ix = x + 0.08 * w
            iy = y + 0.12 * h
            iw = 0.36 * w
            ih = 0.76 * h
            text_x = x + 0.50 * w
            text_ha = "left"
        else:
            ix = x + 0.15 * w
            iy = y + 0.12 * h
            iw = 0.70 * w
            ih = 0.76 * h
            text_x = x + 0.50 * w
            text_ha = "center"

        iax = ax.inset_axes([ix, iy, iw, ih], transform=ax.transData, zorder=4)
        iax.imshow(image, alpha=image_alpha)
        iax.set_axis_off()

        # Dark overlay behind text for readability
        overlay = FancyBboxPatch(
            (x + 0.04 * w, y + 0.14 * h),
            0.50 * w if image_side == "right" else 0.88 * w,
            0.72 * h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=(0, 0, 0, 0.40),
            edgecolor=(0, 0, 0, 0),
            zorder=5
        )
        ax.add_patch(overlay)
    else:
        text_x = x + 0.50 * w
        text_ha = "center"

    # Text vertical anchors
    title_y = y + 0.73 * h
    formula_y = y + 0.49 * h
    subtitle_y = y + 0.23 * h

    # Title
    t1 = ax.text(
        text_x, title_y, title,
        ha=text_ha, va="center",
        color=TXT, fontsize=title_size, zorder=6
    )
    t1.set_path_effects([pe.withStroke(linewidth=1.2, foreground=(0, 0, 0, 0.6))])

    # Formula
    if formula:
        t2 = ax.text(
            text_x, formula_y, formula,
            ha=text_ha, va="center",
            color=TXT, fontsize=formula_size, zorder=6
        )
        t2.set_path_effects([pe.withStroke(linewidth=1.2, foreground=(0, 0, 0, 0.6))])

    # Subtitle
    if subtitle:
        t3 = ax.text(
            text_x, subtitle_y, subtitle,
            ha=text_ha, va="center",
            color=SUBTXT, fontsize=subtitle_size, zorder=6, wrap=True
        )
        t3.set_path_effects([pe.withStroke(linewidth=1.0, foreground=(0, 0, 0, 0.5))])

    return patch


def add_arrow(ax, start, end, rad=0.0, lw=1.8, color=(1, 1, 1, 0.78), z=7):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=16,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z
    )
    ax.add_patch(arr)
    return arr


def box_center(x, y, w, h):
    return (x + 0.5 * w, y + 0.5 * h)


def right_mid(x, y, w, h):
    return (x + w, y + 0.5 * h)


def left_mid(x, y, w, h):
    return (x, y + 0.5 * h)


def top_mid(x, y, w, h):
    return (x + 0.5 * w, y + h)


def bottom_mid(x, y, w, h):
    return (x + 0.5 * w, y)


# ------------------------------------------------------------
# layout
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 8), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 27.5)
ax.set_ylim(0, 10)
ax.axis("off")

motif = prepare_motif(IMG_PATH)

# Main horizontal pipeline boxes
boxes = {
    "design":    (0.7, 5.0, 2.5, 2.1),
    "measure":   (3.9, 5.0, 3.1, 2.1),
    "fiber":     (7.7, 5.0, 2.9, 2.1),
    "iproj":     (11.3, 5.0, 3.8, 2.1),
    "residual":  (15.8, 5.0, 3.0, 2.1),
    "poisson":   (19.5, 5.0, 3.9, 2.1),
    "action":    (24.1, 5.0, 2.7, 2.1),
    "risk":      (11.3, 2.0, 3.8, 1.8),
    "objective": (17.8, 1.8, 4.1, 2.0),
    "update":    (22.5, 2.0, 3.1, 1.8),
    "ref":       (14.3, 8.0, 4.5, 1.3),
}

# Draw boxes
add_box(
    ax, *boxes["design"],
    title="Measurement design",
    formula=r"$\eta$",
    subtitle="sensor geometry / observable parameters",
    title_size=14, formula_size=21, subtitle_size=9
)

add_box(
    ax, *boxes["measure"],
    title="Measurements",
    formula=r"$\widehat{c}_\eta(t),\ \dot{\widehat{c}}_\eta(t)$",
    subtitle="finite acquisitions + moment reconstruction",
    title_size=14, formula_size=16, subtitle_size=9
)

add_box(
    ax, *boxes["fiber"],
    title="Moment fiber",
    formula=r"$\mathcal{F}_\eta(\widehat{c}_\eta(t))$",
    subtitle="laws matching the reconstructed moments",
    title_size=14, formula_size=15, subtitle_size=9
)

add_box(
    ax, *boxes["iproj"],
    title="I-projected law",
    formula=r"$Q_t^\eta$",
    subtitle="KL projection onto the moment fiber",
    title_size=14, formula_size=21, subtitle_size=9,
    image=motif, image_alpha=0.33, image_side="right",
    accent=NEON_MAGENTA
)

add_box(
    ax, *boxes["residual"],
    title="Continuity residual",
    formula=r"$h_{t,\eta}$",
    subtitle="forcing implied by the projected law",
    title_size=14, formula_size=19, subtitle_size=9
)

add_box(
    ax, *boxes["poisson"],
    title="Full-law correction",
    formula=r"$\delta^\star = -\nabla \psi^\star$",
    subtitle="weighted Poisson / Ritz solve",
    title_size=14, formula_size=16, subtitle_size=9,
    image=motif, image_alpha=0.33, image_side="right",
    accent=NEON_CYAN
)

add_box(
    ax, *boxes["action"],
    title="Full action",
    formula=r"$A(\eta)$",
    subtitle="energy of the correction",
    title_size=14, formula_size=20, subtitle_size=9
)

add_box(
    ax, *boxes["risk"],
    title="Scientific risk",
    formula=r"$R(\eta)$",
    subtitle="QoI error under the projected law",
    title_size=14, formula_size=20, subtitle_size=9
)

add_box(
    ax, *boxes["objective"],
    title="Risk-constrained objective",
    formula=r"$R(\eta) \leq R_{\mathrm{max}}$  |  $A(\eta)$",
    subtitle="feasibility first, then transportability",
    title_size=14, formula_size=16, subtitle_size=9
)

add_box(
    ax, *boxes["update"],
    title="Update design",
    formula=r"$\eta \leftarrow \eta - \alpha \nabla_\eta \mathcal{L}$",
    subtitle="differentiate through the pipeline",
    title_size=14, formula_size=13, subtitle_size=9
)

add_box(
    ax, *boxes["ref"],
    title="Frozen reference",
    formula=r"$(\widetilde{Q}_t,\ u_t)$",
    subtitle="common geometry held fixed during optimization",
    title_size=13, formula_size=16, subtitle_size=8
)

# ------------------------------------------------------------
# arrows: main horizontal flow
# ------------------------------------------------------------
main_flow = ["design", "measure", "fiber", "iproj", "residual", "poisson", "action"]
for a, b in zip(main_flow[:-1], main_flow[1:]):
    xa, ya, wa, ha = boxes[a]
    xb, yb, wb, hb = boxes[b]
    add_arrow(ax, right_mid(xa, ya, wa, ha), left_mid(xb, yb, wb, hb), rad=0.0)

# Branches to risk and objective
x, y, w, h = boxes["iproj"]
xr, yr, wr, hr = boxes["risk"]
add_arrow(ax, bottom_mid(x, y, w, h), top_mid(xr, yr, wr, hr), rad=0.0)

xa, ya, wa, ha = boxes["action"]
xo, yo, wo, ho = boxes["objective"]
xr, yr, wr, hr = boxes["risk"]
add_arrow(ax, bottom_mid(xa, ya, wa, ha), (xo + 0.72 * wo, yo + ho), rad=-0.05)
add_arrow(ax, bottom_mid(xr, yr, wr, hr), (xo + 0.28 * wo, yo + ho), rad=0.05)

# Objective -> update
xu, yu, wu, hu = boxes["update"]
add_arrow(ax, right_mid(*boxes["objective"]), left_mid(*boxes["update"]), rad=0.0)

# Loop back update -> design
add_arrow(
    ax,
    (boxes["update"][0] + 0.5 * boxes["update"][2], boxes["update"][1]),
    (boxes["design"][0] + 0.4 * boxes["design"][2], boxes["design"][1]),
    rad=-0.45,
    lw=2.0
)

# Frozen reference arrows
add_arrow(ax, bottom_mid(*boxes["ref"]), top_mid(*boxes["iproj"]), rad=0.12)
add_arrow(ax, bottom_mid(*boxes["ref"]), top_mid(*boxes["poisson"]), rad=-0.12)

# ------------------------------------------------------------
# subtle labels
# ------------------------------------------------------------
title = ax.text(
    13.75, 9.65,
    "FIDE overview",
    ha="center", va="center",
    color="white", fontsize=20, zorder=10
)
title.set_path_effects([pe.withStroke(linewidth=1.2, foreground=(0, 0, 0, 0.6))])

caption = ax.text(
    13.75, 0.55,
    "measurement-implied law reconstruction  →  minimum-energy physical correction  →  risk-constrained design update",
    ha="center", va="center",
    color=(1, 1, 1, 0.72), fontsize=11, zorder=10
)
caption.set_path_effects([pe.withStroke(linewidth=1.0, foreground=(0, 0, 0, 0.5))])

# ------------------------------------------------------------
# save
# ------------------------------------------------------------
plt.tight_layout(pad=0.4)
plt.savefig(OUT_PATH, dpi=300, facecolor=BG, bbox_inches="tight")
plt.show()
print(f"Saved to {OUT_PATH}")