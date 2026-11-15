"""Inference as a small web app: every page load (or the Refresh button) runs a fresh prediction.

Login, feature view and model are loaded once at startup, so a request only fetches the live
cloud_cover and the newest features. The feature pipeline runs separately (see docker-compose.yml).

Usage:
    python serve.py        # then open http://localhost:8000
"""
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv

from inference_pipeline import ForecastUnavailableError, connect, predict

PORT = 8000
RELOAD_S = 15  # how often the loading page reloads itself until the newest hour is in the feature store
# banner with a spinner; light-dark() follows the page's light/dark mode
LOADING = (
    "<style>.loading{display:flex;align-items:center;gap:10px;margin:0 0 16px;padding:10px 14px;border-radius:12px;"
    "font:500 14px/1.4 Inter,system-ui,sans-serif;background:light-dark(#e3eefb,#1a2533);"
    "color:light-dark(#0f1720,#eef0f3)}.loading i{flex:none;width:16px;height:16px;border:2px solid currentColor;"
    "border-right-color:transparent;border-radius:50%;animation:spin .8s linear infinite}"
    "@keyframes spin{to{transform:rotate(360deg)}}</style>"
    '<div class="loading" role="status"><i></i>Loading the newest weather data. The page refreshes by itself.</div>'
)
# shown with LOADING when the server has no earlier page yet (right after start)
EMPTY = (
    '<!doctype html><html lang="en"><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1"><title>Zurich Rain Forecast</title>'
    "<style>:root{color-scheme:light dark}body{margin:0;background:light-dark(#f5f6f8,#101214)}</style>"
    '<main style="max-width:820px;margin:0 auto;padding:32px 16px">'
)
LAST_PAGE = ""  # last successful page, shown with LOADING while the newest hour is missing


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global LAST_PAGE
        if self.path != "/":
            self.send_error(404)
            return
        loading = False
        try:
            page = LAST_PAGE = predict(*CONTEXT)
        except ForecastUnavailableError:
            # start of the hour, before the feature pipeline wrote it: keep the last page, reload until fresh
            loading = True
            page = LAST_PAGE.replace("<main>", "<main>" + LOADING, 1) if LAST_PAGE else EMPTY + LOADING
        except Exception as e:  # show the error instead of a dropped connection
            self.send_error(500, explain=str(e))
            raise
        body = page.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if loading:
            self.send_header("Refresh", str(RELOAD_S))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    load_dotenv()
    # ponytail: model is fixed at startup, restart the server to pick up a newly trained better model
    CONTEXT = connect()
    print(f"serving on http://localhost:{PORT}")
    # single-threaded: one inference at a time, enough for a demo page
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
