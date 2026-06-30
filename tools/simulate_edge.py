#!/usr/bin/env python3
# simulate_edge.py — pretend to be a reCamera edge device, so you can exercise the
# server with NO hardware. It (1) serves a JPEG at http://<this-host>/vis/<cam>.jpg
# on port 80 (run with sudo, or pass --http-port and a matching server reachability),
# and (2) sends start/active/end UDP events to the server's event port.
#
# Because the server pulls the frame from the *source IP of the datagram* on port 80,
# the realistic test needs port 80. Easiest: run this on the same host as the server
# with sudo. For a quick GPU-only check without any of this, just:
#   curl --data-binary @some.jpg -H 'Content-Type: image/jpeg' http://localhost:8092/detect
#
# Usage: sudo python3 simulate_edge.py --img path/to.jpg --server 127.0.0.1 [--cam sim] [--evt-port 23906]
import argparse, json, socket, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--img", required=True, help="JPEG to serve as the gated frame")
ap.add_argument("--server", default="127.0.0.1", help="server IP to send events to")
ap.add_argument("--evt-port", type=int, default=23906)
ap.add_argument("--http-port", type=int, default=80, help="port to serve /vis/<cam>.jpg (server pulls :80)")
ap.add_argument("--cam", default="sim")
ap.add_argument("--secs", type=float, default=15, help="how long to stay 'active'")
args = ap.parse_args()

JPEG = open(args.img, "rb").read()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == f"/vis/{args.cam}.jpg":
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(JPEG))); self.end_headers()
            self.wfile.write(JPEG)
        else:
            self.send_response(404); self.end_headers()


threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", args.http_port), H).serve_forever(),
                 daemon=True).start()
print(f"serving /vis/{args.cam}.jpg on :{args.http_port}; sending events to {args.server}:{args.evt_port}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def send(ev, **kw):
    d = {"cam": args.cam, "event": ev, "ts": time.time(), "seq": send.n, **kw}; send.n += 1
    sock.sendto(json.dumps(d).encode(), (args.server, args.evt_port))
send.n = 0

box = [760, 420, 1120, 980]
send("start", label="person", conf=0.82, box=box, threat=0.7, fps=8,
     dets=[{"l": "person", "c": 0.82, "b": box}], img=f"{args.cam}.jpg")
t0 = time.time()
while time.time() - t0 < args.secs:
    time.sleep(2)
    send("active", label="person", conf=0.8, box=box, threat=0.6, fps=8,
         dets=[{"l": "person", "c": 0.8, "b": box}], img=f"{args.cam}.jpg")
send("end", dur=time.time() - t0)
print("done (sent end)")
