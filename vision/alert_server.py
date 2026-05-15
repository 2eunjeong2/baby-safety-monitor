"""
Baby Safety Alert Server
캡처된 경고 이미지 조회 + 실시간 카메라 스트리밍

단독 실행: python3 vision/alert_server.py
baby_monitor_v4.py 에서 import해 사용하는 경우 자동으로 프레임 공유됨

접속:
  http://<IP>:8080/       — 경고 이미지 목록
  http://<IP>:8080/live   — 실시간 스트리밍
  http://<IP>:8080/api/alerts — JSON API
"""

import time as _time
import threading
from flask import Flask, send_file, jsonify, abort, Response
from pathlib import Path
from datetime import datetime

ROOT        = Path(__file__).resolve().parent.parent
CAPTURE_DIR = ROOT / "captures"

app = Flask(__name__)

# ── 프레임 공유 (baby_monitor_v4 → 스트리밍) ───────────────
_frame_lock   = threading.Lock()
_latest_frame = None   # JPEG bytes


def set_frame(jpeg_bytes: bytes):
    """렌더링된 프레임을 스트리밍용으로 업데이트."""
    global _latest_frame
    with _frame_lock:
        _latest_frame = jpeg_bytes


def _mjpeg_generator():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        _time.sleep(0.04)   # ~25 fps cap


# ── 라우트 ────────────────────────────────────────────────

STATE_LABEL = {
    "DANGER_BACK":  "후면 위험",
    "DANGER_NODET": "미감지 위험",
    "CAUTION":      "측면 경고",
}
STATE_COLOR = {
    "DANGER_BACK":  "#e53935",
    "DANGER_NODET": "#e53935",
    "CAUTION":      "#fb8c00",
}


def parse_filename(name: str) -> dict:
    stem  = Path(name).stem
    parts = stem.rsplit("_", 2)
    try:
        state    = parts[0] if len(parts) == 3 else stem
        dt_str   = f"{parts[1]}_{parts[2]}" if len(parts) == 3 else ""
        dt       = datetime.strptime(dt_str, "%Y%m%d_%H%M%S") if dt_str else None
        dt_label = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""
    except Exception:
        state, dt_label = stem, ""
    return {
        "filename": name,
        "state":    state,
        "label":    STATE_LABEL.get(state, state),
        "color":    STATE_COLOR.get(state, "#757575"),
        "datetime": dt_label,
        "url":      f"/image/{name}",
    }


@app.route("/stream")
def stream():
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live")
def live():
    return """<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baby Safety — Live</title>
<style>
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:#000; display:flex; flex-direction:column;
         align-items:center; font-family:sans-serif; }
  header { width:100%; max-width:640px; padding:12px 16px;
           background:#1c1c1e; color:#fff; font-size:15px;
           font-weight:600; display:flex; justify-content:space-between; }
  header a { color:#aaa; font-size:13px; font-weight:400; text-decoration:none; }
  img { width:100%; max-width:640px; display:block; }
</style>
</head><body>
<header>
  <span>&#9679; Live Feed</span>
  <a href="/">Alert 이미지 보기</a>
</header>
<img src="/stream" alt="Live">
</body></html>"""


@app.route("/api/alerts")
def api_alerts():
    if not CAPTURE_DIR.exists():
        return jsonify([])
    files = sorted(CAPTURE_DIR.glob("*.jpg"), reverse=True)
    return jsonify([parse_filename(f.name) for f in files])


@app.route("/api/alerts/<filename>", methods=["DELETE"])
def delete_alert(filename):
    path = CAPTURE_DIR / filename
    if not path.exists():
        abort(404)
    path.unlink()
    return jsonify({"deleted": filename})


@app.route("/image/<filename>")
def get_image(filename):
    path = CAPTURE_DIR / filename
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype="image/jpeg")


@app.route("/")
def index():
    files    = sorted(CAPTURE_DIR.glob("*.jpg"), reverse=True) if CAPTURE_DIR.exists() else []
    captures = [parse_filename(f.name) for f in files]

    cards = ""
    for c in captures:
        cards += f"""
        <div class="card">
          <div class="badge" style="background:{c['color']};">
            {c['label']} &nbsp; <span class="ts">{c['datetime']}</span>
          </div>
          <img src="{c['url']}" loading="lazy">
        </div>"""

    if not cards:
        cards = '<p class="empty">캡처된 이미지가 없습니다.</p>'

    return f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baby Safety Alerts</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0f0f0f; color:#fff; font-family:sans-serif; }}
  header {{ padding:14px 16px; background:#1c1c1e; font-size:17px; font-weight:600;
            display:flex; align-items:center; justify-content:space-between; }}
  header a {{ color:#0a84ff; font-size:13px; font-weight:500; text-decoration:none; }}
  .dot {{ width:10px; height:10px; border-radius:50%; background:#e53935;
          display:inline-block; margin-right:8px; }}
  .list  {{ max-width:520px; margin:0 auto; padding:16px; }}
  .card  {{ border-radius:12px; overflow:hidden; margin-bottom:16px; background:#1c1c1e; }}
  .badge {{ padding:8px 14px; font-size:13px; font-weight:600; color:#fff;
            display:flex; justify-content:space-between; align-items:center; }}
  .ts    {{ font-weight:400; font-size:12px; opacity:.85; }}
  img    {{ width:100%; display:block; }}
  .empty {{ color:#555; text-align:center; margin-top:60px; font-size:15px; }}
</style>
</head><body>
<header>
  <span><span class="dot"></span>Baby Safety — Alert Captures</span>
  <a href="/live">&#9654; Live</a>
</header>
<div class="list">{cards}</div>
</body></html>"""


if __name__ == "__main__":
    CAPTURE_DIR.mkdir(exist_ok=True)
    print("[서버] http://0.0.0.0:8080")
    print("[라이브] http://0.0.0.0:8080/live")
    print(f"[캡처 폴더] {CAPTURE_DIR}")
    app.run(host="0.0.0.0", port=8080, debug=False)
