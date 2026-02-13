import pandas as pd

from app.ui.table_styling import (
    build_alignment_map,
    build_global_table_css,
    resolve_table_theme,
)


def test_resolve_table_theme_uses_design_tokens_when_present():
    tokens = {
        "tables": {
            "header_bg": "token-header-bg",
            "header_text": "token-header-text",
            "row_odd_bg": "token-row-odd",
            "row_even_bg": "token-row-even",
            "border": "token-border",
        }
    }

    theme = resolve_table_theme(tokens, mode="light")

    assert theme["header_bg"] == "token-header-bg"
    assert theme["header_text"] == "token-header-text"
    assert theme["row_odd_bg"] == "token-row-odd"
    assert theme["row_even_bg"] == "token-row-even"
    assert theme["border"] == "token-border"


def test_resolve_table_theme_falls_back_for_dark_mode_when_missing_tokens():
    theme = resolve_table_theme(tokens={}, mode="dark")

    assert theme["header_bg"] == "var(--secondary-background-color)"
    assert theme["header_text"] == "var(--text-color)"
    assert theme["row_odd_bg"] == "var(--background-color)"
    assert theme["row_even_bg"] == "var(--secondary-background-color)"


def test_build_alignment_map_right_aligns_numeric_and_left_aligns_text():
    df = pd.DataFrame({"name": ["A"], "value": [1.23], "count": [4]})

    alignments = build_alignment_map(df)

    assert alignments["name"] == "left"
    assert alignments["value"] == "right"
    assert alignments["count"] == "right"


def test_build_global_table_css_includes_header_striping_border_and_key_metric_emphasis():
    theme = {
        "header_bg": "hb",
        "header_text": "ht",
        "row_odd_bg": "odd",
        "row_even_bg": "even",
        "border": "bd",
    }

    css = build_global_table_css(theme)

    assert "background: hb" in css
    assert "color: ht" in css
    assert "background: odd" in css
    assert "background: even" in css
    assert "border: 1px solid bd" in css
    assert ".key-metric" in css
