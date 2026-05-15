"""
Baby Safety Alert Server
캡처된 경고 이미지를 앱/브라우저에서 확인하는 서버

실행: python3 vision/alert_server.py
접속: http://<Pi_IP>:5000/
API:  http://<Pi_IP>:5000/api/alerts
"""

from flask import Flask, send_file, jsonify, abort
from pathlib import Path
from datetime import datetime

ROOT        = Path(__file__).resolve().parent.parent
CAPTURE_DIR = ROOT / "captures"

app = Flask(__name__)

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
    """DANGER_BACK_20260515_143022.jpg → {state, datetime, filename, url}"""
    stem   = Path(name).stem          # e.g. DANGER_BACK_20260515_143022
    # 마지막 두 토큰이 날짜·시간, 나머지가 상태명
    parts  = stem.rsplit("_", 2)
    try:
        state  = parts[0] if len(parts) == 3 else stem
        dt_str = f"{parts[1]}_{parts[2]}" if len(parts) == 3 else ""
        dt     = datetime.strptime(dt_str, "%Y%m%d_%H%M%S") if dt_str else None
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
    if not CAPTURE_DIR.exists():
        captures = []
    else:
        files    = sorted(CAPTURE_DIR.glob("*.jpg"), reverse=True)
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
  *    {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0f0f0f; color:#fff; font-family:sans-serif; }}
  header {{ padding:16px; background:#1c1c1e; font-size:17px; font-weight:600;
            display:flex; align-items:center; gap:10px; }}
  header span {{ width:10px; height:10px; border-radius:50%; background:#e53935;
                 display:inline-block; }}
  .list  {{ max-width:520px; margin:0 auto; padding:16px; }}
  .card  {{ border-radius:12px; overflow:hidden; margin-bottom:16px;
            background:#1c1c1e; }}
  .badge {{ padding:8px 14px; font-size:13px; font-weight:600; color:#fff;
            display:flex; justify-content:space-between; align-items:center; }}
  .ts    {{ font-weight:400; font-size:12px; opacity:.85; }}
  img    {{ width:100%; display:block; }}
  .empty {{ color:#555; text-align:center; margin-top:60px; font-size:15px; }}
</style>
</head><body>
<header><span></span>Baby Safety — Alert Captures</header>
<div class="list">{cards}</div>
</body></html>"""


if __name__ == "__main__":
    CAPTURE_DIR.mkdir(exist_ok=True)
    print(f"[서버] http://0.0.0.0:8080")
    print(f"[캡처 폴더] {CAPTURE_DIR}")
    app.run(host="0.0.0.0", port=8080, debug=False)
