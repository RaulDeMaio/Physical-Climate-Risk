import json
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st


def load_design_tokens():
    tokens_path = Path(__file__).parent / "design_tokens.json"
    if tokens_path.exists():
        with open(tokens_path, "r") as f:
            return json.load(f)
    return None


def apply_oe_branding(fig, theme_color=None):
    """
    Applies Open Economics branding to a Plotly figure.

    Parameters:
    - fig: The Plotly figure.
    - theme_color: Optional. "primary" or "secondary" to force all traces to that color.
    """
    tokens = load_design_tokens()
    if not tokens:
        return fig

    colors = tokens.get("colors", {})
    palettes = colors.get("palettes", {})
    fonts = tokens.get("fonts", {})

    # 1. Fonts
    primary_font = fonts.get("primary", "Arial")
    fallback_font = fonts.get("fallback", "sans-serif")
    font_family = f"'{primary_font}', {fallback_font}"

    fig.update_layout(
        font_family=font_family,
        title_font_family=font_family,
        title_font_color=colors.get("primary", "#4400B3"),
        title_font_size=20,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # 2. Colors
    categorical = palettes.get("categorical", [])
    sequential = palettes.get("sequential", [])

    # Apply forced theme color if requested
    if theme_color in colors:
        target_color = colors[theme_color]
        fig.update_traces(marker_color=target_color)
    elif categorical and not hasattr(fig.data[0], "z"):
        fig.update_layout(colorway=categorical)

    # Apply sequential color scale for maps
    if sequential and any(hasattr(data, "z") for data in fig.data):
        fig.update_coloraxes(colorscale=sequential)
        for data in fig.data:
            if hasattr(data, "colorscale"):
                data.colorscale = sequential

    return fig


def set_streamlit_branding():
    """Sets Streamlit sidebar and main UI colors via custom CSS."""
    tokens = load_design_tokens()
    if not tokens:
        return

    colors = tokens.get("colors", {})
    primary = colors.get("primary", "#4400B3")
    font = tokens.get("fonts", {}).get("primary", "Atkinson Hyperlegible Next")

    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Next:wght@400;700&display=swap');
        
        html, body, [class*="css"] {{
            font-family: '{font}', sans-serif;
        }}
        
        :root {{
            --primary-color: {primary};
        }}
        
        .stButton>button {{
            background-color: {primary};
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }}
        
        .stButton>button:hover {{
            background-color: {primary}dd;
            color: white;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
