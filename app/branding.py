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

    # 3. Value Labels (Roadmap Enhancement)
    for trace in list(fig.data):
        # Bar Charts (Sectors, Countries, Linkages)
        if trace.type == "bar" and getattr(trace, "orientation", None) == "h":
            # Determine format based on magnitude
            if trace.x is not None and len(trace.x) > 0:
                is_pp = any(0 < abs(val) < 1.0 for val in trace.x)
                fmt = ".3f" if is_pp else ",.1f"
                trace.text = trace.x
                trace.texttemplate = f"%{{text:{fmt}}}"
                trace.textposition = "outside"
                trace.textfont = dict(weight="bold")
                # Increase margin to avoid clipping
                fig.update_layout(margin=dict(r=80))

        # Choropleth Map Labels
        if trace.type == "choropleth":
            # Add static labels overlay
            if trace.locations is not None and trace.z is not None:
                label_text = []
                for val in trace.z:
                    # Check if percentage or absolute
                    is_pct_map = "loss_pct" in (fig.layout.title.text or "").lower()
                    if is_pct_map or (0 < abs(val) < 1.0):
                        # If value is > 1.0, it's likely already scaled (e.g. 5.1 for 5.1%)
                        display_val = val if abs(val) > 1.0 else val * 100
                        label_text.append(f"{display_val:.1f}%")
                    else:
                        label_text.append(f"{val:,.1f}")

                fig.add_trace(
                    go.Scattergeo(
                        locations=trace.locations,
                        locationmode=trace.locationmode,
                        text=label_text,
                        mode="text",
                        textfont=dict(color="black", size=12, weight="bold"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                # Update map look & center on Europe
                fig.update_geos(
                    showcountries=True,
                    countrycolor="LightGrey",
                    showland=True,
                    landcolor="white",
                    projection_type="mercator",
                    # European bounds - slightly wider
                    lataxis_range=[32, 72],
                    lonaxis_range=[-25, 45],
                    center=dict(lat=52, lon=10),
                    # Ensure it fits the container
                    fitbounds=False,
                )
                fig.update_layout(height=700, margin=dict(l=0, r=0, t=60, b=0))
                # Ensure hover is also formatted correctly
                if trace.hovertemplate:
                    trace.hovertemplate = trace.hovertemplate.replace(
                        "%{z}", "%{z:.2%}"
                    )

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
