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
    primary_font = fonts.get("primary", "Atkinson Hyperlegible Next")
    fallback_font = fonts.get("fallback", "Arial")
    font_family = f"'{primary_font}', {fallback_font}"

    fig.update_layout(
        font_family=font_family,
        title_font_family=font_family,
        title_font_color=colors.get("primary", "#4400B3"),
        title_font_size=22,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_font_family=font_family,
        xaxis_title_font_family=font_family,
        yaxis_title_font_family=font_family,
    )

    # 2. Colors
    categorical = palettes.get("categorical", [])
    sequential = palettes.get("sequential", [])

    if theme_color in colors:
        target_color = colors[theme_color]
        fig.update_traces(marker_color=target_color)
    elif categorical and not any(hasattr(data, "z") for data in fig.data):
        fig.update_layout(colorway=categorical)

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

    # More aggressive CSS to force font application across Streamlit's shadow DOM and variables
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Next:ital,wght@0,200..800;1,200..800&display=swap');
        
        /* Apply to the whole app */
        html, body, [class*="css"], .stMarkdown, .stText, .stButton, .stSelectbox, .stSlider, .stHeader, .stMetric {{
            font-family: '{font}', 'Arial', sans-serif !important;
        }}
        
        /* Specifically target Streamlit headers and titles */
        h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm {{
            font-family: '{font}', 'Arial', sans-serif !important;
            font-weight: 700 !important;
        }}

        /* Target Streamlit's internal variable for fonts if possible */
        :root {{
            --st-font-sans: '{font}', 'Arial', sans-serif;
            --primary-color: {primary};
        }}
        
        /* Brand Color Button */
        .stButton>button {{
            background-color: {primary} !important;
            color: white !important;
            border-radius: 8px !important;
            font-family: '{font}', sans-serif !important;
        }}
        
        /* Metric Styling */
        [data-testid="stMetricValue"] {{
            font-family: '{font}', sans-serif !important;
            font-weight: 800 !important;
            color: {primary} !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
