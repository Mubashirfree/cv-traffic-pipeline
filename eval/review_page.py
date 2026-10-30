"""Builds report/figures/rag_review.html: for each summary, the keyframes, the metadata counts and the text."""
import base64, csv, html, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.retrieval.service import keyframe_jpeg

CSV = ROOT / "report" / "figures" / "rag_manual_check.csv"
OUT = ROOT / "report" / "figures" / "rag_review.html"


def find(clip_id):
    for ext in ("jpg", "mp4", "png"):
        for p in (ROOT / "data").rglob(f"{clip_id}.{ext}"):
            return p
    return None


def main():
    rows = list(csv.DictReader(open(CSV)))
    parts = ["<meta charset=utf-8><style>body{font-family:sans-serif;max-width:1100px;margin:20px auto}"
             ".row{border:2px solid #888;padding:12px;margin:18px 0;border-radius:8px}"
             "img{width:32%;margin:2px}.sum{background:#fffbe0;padding:8px;font-size:17px}"
             "table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:3px 8px}</style>",
             "<h2>Check each summary against the images and counts. Then put 1 (all claims supported) or 0 (anything invented) "
             "in manual_ok in the CSV, in the same row number.</h2>"]
    for n, r in enumerate(rows, 1):
        meta = json.loads(r["metadata"])
        parts.append(f"<div class=row><h3>Row {n}: [{html.escape(r['dataset'])}] {html.escape(r['query'])}</h3>")
        for m in meta:
            p = find(m["clip_id"])
            try:
                b = base64.b64encode(keyframe_jpeg(p)).decode()
                parts.append(f"<img src='data:image/jpeg;base64,{b}' title='{m['clip_id']}'>")
            except Exception:
                parts.append(f"<i>[image for {m['clip_id']} not found]</i>")
        parts.append("<table><tr><th>clip</th><th>camera</th><th>t (s)</th><th>objects detected by the system</th></tr>")
        for m in meta:
            parts.append(f"<tr><td>{m['clip_id']}</td><td>{m['camera_id']}</td><td>{m['timestamp']}</td>"
                         f"<td>{html.escape(str(m['counts']))}</td></tr>")
        parts.append("</table>")
        parts.append(f"<p class=sum><b>Summary:</b> {html.escape(r['summary'])}</p>")
        parts.append(f"<p>Automatic flags: <b>{html.escape(r['auto_flags']) or 'none'}</b></p></div>")
    OUT.write_text("\n".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
