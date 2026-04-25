# =============================================================================
# style/theme.py — Personal Design System | William Cheymol Portfolio
# =============================================================================
# Mirrors the portfolio CSS design system (style.css).
# Change the hex codes here once — everything updates everywhere.
#
# Usage in notebook:
#   from style.theme import COLORS, LAYOUT, apply_theme
#   apply_theme()   # call once at the top of the notebook
#
#   fig.update_layout(**LAYOUT, title="My chart")
#   fig.add_trace(go.Scatter(..., line=dict(color=COLORS["primary"])))
# =============================================================================

import plotly.graph_objects as go
import plotly.io as pio

# =============================================================================
# Palette — Quant Dark
# =============================================================================

COLORS = {
    # Core — mapped from portfolio :root variables
    "primary":    "#00e676",   # --accent  : green — main lines, calls
    "secondary":  "#00b0ff",   # --accent2 : blue — puts, secondary series
    "accent":     "#00e676",   # --accent  : green — highlights, strikes
    "negative":   "#ff6b6b",   # --accent3 : red/coral — losses, warnings
    "neutral":    "#898989",   # --muted   : grey — secondary elements

    # Backgrounds — mapped from --bg / --bg2 / --bg3
    "bg":         "#080808",   # --bg      : near-black — canvas
    "bg_panel":   "#111111",   # --bg2     : plot area
    "bg_card":    "#1a1a1a",   # --bg3     : card / annotation backgrounds

    # Text — mapped from --text / --muted2
    "text":       "#e8e8e8",   # --text    : main labels and titles
    "text_muted": "#acacac",   # --muted2  : axis labels, annotations

    # Specific use cases
    "yellow":       "#e1da02",   # yellow — call option
    "blue":        "#00b0ff",   # blue  — put option
    "red":  "#d33501",   # red   — theta P&L component
    "orange":  "#d77f03",
    "purple": "#6903d7",
    "pink": "#ff6ac3",
    "white": "#ededed"
}

# =============================================================================
# Font
# =============================================================================

FONT = dict(
    family="JetBrains Mono, Fira Code, monospace",   # --font-mono
    size=12,
    color=COLORS["text"],
)

FONT_TITLE = dict(
    family="Inter, system-ui, sans-serif",            # --font-sans
    size=14,
    color=COLORS["text"],
)

# =============================================================================
# Plotly layout defaults — spread with **LAYOUT in fig.update_layout()
# =============================================================================

LAYOUT = dict(
    paper_bgcolor = COLORS["bg"],
    plot_bgcolor  = COLORS["bg_panel"],
    font          = FONT,
    title_font    = FONT_TITLE,
    height        = 440,
    margin        = dict(l=60, r=30, t=60, b=50),
    legend        = dict(
        bgcolor     = COLORS["bg_card"],
        bordercolor = COLORS["neutral"],
        borderwidth = 1,
        font        = dict(color=COLORS["text_muted"], size=12),
    ),
    xaxis = dict(
        gridcolor     = "#222222",                    # --border
        zerolinecolor = "#333333",
        tickfont      = dict(color=COLORS["text_muted"], family="JetBrains Mono, Fira Code, monospace"),
        title_font    = dict(color=COLORS["text_muted"], family="JetBrains Mono, Fira Code, monospace"),
        linecolor     = "#222222",
    ),
    yaxis = dict(
        gridcolor     = "#222222",                    # --border
        zerolinecolor = "#333333",
        tickfont      = dict(color=COLORS["text_muted"], family="JetBrains Mono, Fira Code, monospace"),
        title_font    = dict(color=COLORS["text_muted"], family="JetBrains Mono, Fira Code, monospace"),
        linecolor     = "#222222",
    ),
)

# =============================================================================
# apply_theme() — call once at the top of the notebook
# =============================================================================

def apply_theme():
    """
    Register and activate the Quant Dark theme globally.
    After calling this, all plotly figures will use the theme by default.
    """
    pio.templates["portfolio"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor = COLORS["bg"],
            plot_bgcolor  = COLORS["bg_panel"],
            font          = FONT,
            colorway      = [
                COLORS["primary"], COLORS["secondary"],
                COLORS["accent"],  COLORS["negative"],
                COLORS["neutral"],
            ],
        )
    )
    pio.templates.default = "portfolio"
    print("Portfolio theme applied.")
