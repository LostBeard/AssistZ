"""Static server for a published Blazor wwwroot with SPA fallback (unknown paths without an extension -> index.html).
Usage: python spa_serve.py <wwwroot> <port>"""
import http.server, os, sys, functools

ROOT, PORT = sys.argv[1], int(sys.argv[2])

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript",
                      ".json": "application/json", ".dat": "application/octet-stream", ".blat": "application/octet-stream"}
    def send_head(self):
        path = self.translate_path(self.path.split("?")[0])
        if not os.path.exists(path) and "." not in os.path.basename(path):
            self.path = "/index.html"
        return super().send_head()
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def log_message(self, *a):
        pass

class Server(http.server.ThreadingHTTPServer):
    # Blazor fetches ~100 framework files at once; socketserver's default listen backlog of 5 overflows on Windows and
    # the browser gets ERR_CONNECTION_REFUSED for some of them ("Error in mono_download_assets ... Failed to fetch").
    request_queue_size = 256
    daemon_threads = True

Server(("127.0.0.1", PORT), functools.partial(Handler, directory=ROOT)).serve_forever()
