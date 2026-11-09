"""
Local HTTP server + live PyTorch Simulator API for the Midnight Grid dashboard.

Refreshes `app/dashboard/data.js` and `app/dashboard/simulator_bundle.js` from
verified artifacts and serves `app/dashboard/` on localhost alongside JSON
endpoints backed by `src/simulator.py`:
  - GET  /api/availability
  - POST /api/scenario
  - POST /api/parse_csv
  - POST /api/predict

Usage:
    python app/serve_dashboard.py
    python app/serve_dashboard.py --port 8505 --no-browser
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DASH_DIR = (APP_DIR / "dashboard").resolve()
MAX_BODY_BYTES = 1_048_576  # 1 MB request payload limit

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.simulator import (  # noqa: E402
    PredictionSimulator,
    SimulatorValidationError,
    UnsupportedConfigurationError,
)


class PayloadTooLargeError(ValueError):
    """Raised when an HTTP request body exceeds MAX_BODY_BYTES."""


def make_handler(simulator: PredictionSimulator):
    class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(DASH_DIR), **kwargs)

        def end_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            super().end_headers()

        def list_directory(self, path: str):
            self.send_error(403, "Directory listing is disabled.")
            return None

        def translate_path(self, path: str) -> str:
            resolved = Path(super().translate_path(path)).resolve()
            if not resolved.is_relative_to(DASH_DIR):
                return str(DASH_DIR / "__nonexistent__")
            return str(resolved)

        def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            if length > MAX_BODY_BYTES:
                raise PayloadTooLargeError(
                    f"Request payload ({length} bytes) exceeds 1 MB limit."
                )
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw)

        def do_GET(self) -> None:
            route = self.path.split("?", 1)[0]
            if route == "/api/availability":
                self._send_json(200, {"ok": True, "availability": simulator.get_availability_matrix()})
                return
            if route in ("/", ""):
                self.path = "/index.html"
            super().do_GET()

        def do_POST(self) -> None:
            route = self.path.split("?", 1)[0]
            if not route.startswith("/api/"):
                self._send_json(404, {"ok": False, "error": f"Unknown endpoint {route}"})
                return

            try:
                body = self._read_json_body()
                if route == "/api/scenario":
                    seed_raw = body.get("seed", 42)
                    seed = int(seed_raw) if seed_raw is not None and str(seed_raw).strip() != "" else None
                    res = simulator.generate_scenario(
                        district=str(body.get("district", "mathura")),
                        scenario=str(body.get("scenario", "normal")),
                        seed=seed,
                    )
                    self._send_json(200, {"ok": True, "result": res})
                elif route == "/api/parse_csv":
                    res = simulator.parse_csv_input(
                        csv_text=str(body.get("csv_text", "")),
                        feature_config=str(body.get("feature_config", "consumption_only")),
                        row_offset=int(body.get("row_offset", 0)),
                    )
                    self._send_json(200, {"ok": True, "result": res})
                elif route == "/api/predict":
                    res = simulator.predict_next_interval(
                        scale=str(body.get("scale", "aggregate")),
                        district=str(body.get("district", "mathura")),
                        arch=str(body.get("arch", "LSTM")),
                        feature_config=str(body.get("feature_config", "consumption_only")),
                        consumption_kw=body.get("consumption_kw", []),
                        weather_rows=body.get("weather_rows"),
                    )
                    self._send_json(200, {"ok": True, "result": res})
                else:
                    self._send_json(404, {"ok": False, "error": f"Unknown endpoint {route}"})
            except PayloadTooLargeError as exc:
                self._send_json(413, {"ok": False, "error_type": "payload_too_large", "error": str(exc)})
            except SimulatorValidationError as exc:
                self._send_json(400, {"ok": False, "error_type": "validation", "error": str(exc)})
            except UnsupportedConfigurationError as exc:
                self._send_json(422, {"ok": False, "error_type": "unsupported", "error": str(exc)})
            except json.JSONDecodeError:
                self._send_json(400, {"ok": False, "error_type": "validation", "error": "Malformed JSON request body."})
            except Exception:  # pragma: no cover
                self._send_json(500, {"ok": False, "error_type": "internal", "error": "Internal server error during inference."})

    return DashboardRequestHandler


def main() -> None:
    default_host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    default_port = int(os.environ.get("DASHBOARD_PORT", "8505"))

    parser = argparse.ArgumentParser(description="Serve Midnight Grid 3D Glass research dashboard.")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--skip-rebuild", action="store_true")
    args = parser.parse_args()

    if not args.skip_rebuild or not (DASH_DIR / "data.js").exists():
        from app.build_dashboard_data import main as build_data
        build_data()
    if not args.skip_rebuild or not (DASH_DIR / "simulator_bundle.js").exists():
        from app.build_simulator_bundle import main as build_sim_bundle
        build_sim_bundle()

    simulator = PredictionSimulator(ROOT_DIR)
    handler = make_handler(simulator)
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        url = f"http://{args.host}:{args.port}/index.html"
        print(f"Midnight Grid — 3D Glass dashboard + PyTorch Simulator API serving at {url}")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard server.")


if __name__ == "__main__":
    main()
