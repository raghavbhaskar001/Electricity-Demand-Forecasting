"""
Streamlit entry point for the Midnight Grid — 3D Glass research dashboard
and Interactive Electricity Demand Prediction Simulator.

Auto-builds `app/dashboard/data.js` and `app/dashboard/simulator_bundle.js`
from verified project artifacts if needed and mounts the full-bleed Midnight Grid
3D Glass interface inside Streamlit.

Run with:
    streamlit run app/app.py

Or open `app/dashboard/index.html` directly in any browser without a server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DASH_DIR = APP_DIR / "dashboard"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def ensure_bundle() -> None:
    if not (DASH_DIR / "data.js").exists():
        from app.build_dashboard_data import main as build_data
        build_data()
    if not (DASH_DIR / "simulator_bundle.js").exists():
        from app.build_simulator_bundle import main as build_sim_bundle
        build_sim_bundle()


@st.cache_data(show_spinner=False)
def compile_inline_document(mtime_key: float) -> str:
    """Inline styles.css, data.js, simulator_bundle.js, simulator.js, and app.js into index.html."""
    _ = mtime_key
    html = (DASH_DIR / "index.html").read_text(encoding="utf-8")
    css = (DASH_DIR / "styles.css").read_text(encoding="utf-8")
    data_js = (DASH_DIR / "data.js").read_text(encoding="utf-8")
    sim_bundle_js = (DASH_DIR / "simulator_bundle.js").read_text(encoding="utf-8")
    sim_js = (DASH_DIR / "simulator.js").read_text(encoding="utf-8")
    app_js = (DASH_DIR / "app.js").read_text(encoding="utf-8")

    html = html.replace('<link rel="stylesheet" href="styles.css" />', f"<style>{css}</style>")
    html = html.replace('<script src="data.js"></script>', f"<script>{data_js}</script>")
    html = html.replace('<script src="simulator_bundle.js"></script>', f"<script>{sim_bundle_js}</script>")
    html = html.replace('<script src="simulator.js"></script>', f"<script>{sim_js}</script>")
    html = html.replace('<script src="app.js"></script>', f"<script>{app_js}</script>")
    return html


def main() -> None:
    st.set_page_config(
        page_title="Midnight Grid · Electricity Demand Forecasting",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
          html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background: #071321 !important;
            overflow: hidden !important;
          }
          header[data-testid="stHeader"],
          [data-testid="stToolbar"],
          [data-testid="stDecoration"],
          footer { display: none !important; }
          .block-container, [data-testid="stMainBlockContainer"] {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
          }
          iframe {
            border: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            display: block !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    ensure_bundle()
    mtimes = sum(
        (DASH_DIR / name).stat().st_mtime
        for name in ("index.html", "styles.css", "data.js", "simulator_bundle.js", "simulator.js", "app.js")
    )
    doc = compile_inline_document(mtimes)
    components.html(doc, height=960, scrolling=True)


if __name__ == "__main__":
    main()
