"""Inference as a small web app: every page load (or the Refresh button) runs a fresh prediction.

Login, feature view and model are loaded once at startup, so a request only fetches the live
cloud_cover and the newest features. The feature pipeline runs separately (see docker-compose.yml).

Usage:
    python serve.py        # then open http://localhost:8000
"""
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv

from inference_pipeline import connect, predict

PORT = 8000


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        try:
            body = predict(*CONTEXT).encode()
        except Exception as e:  # show the error instead of a dropped connection
            self.send_error(500, explain=str(e))
            raise
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    load_dotenv()
    # ponytail: model is fixed at startup, restart the server to pick up a newly trained better model
    CONTEXT = connect()
    print(f"serving on http://localhost:{PORT}")
    # single-threaded: one inference at a time, enough for a demo page
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
