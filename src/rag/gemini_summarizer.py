"""Phase 6: grounded summaries with Gemini (google-genai SDK; key in .env as GEMINI_API_KEY)."""
import os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.rag.prompt_templates import build_prompt, parse_report, report_to_text
from src.retrieval.service import keyframe_jpeg

load_dotenv(ROOT / ".env")


def model_name():
    if os.getenv("GEMINI_MODEL"):
        return os.environ["GEMINI_MODEL"]
    t = ROOT / "test_gemini.py"
    if t.exists():
        m = re.search(r"""model\s*=\s*["']([\w\.\-]+)["']""", t.read_text())
        if m:
            return m.group(1)
    return "gemini-2.5-flash"


def summarize(query, hits, max_clips=3):
    from google import genai
    from google.genai import types
    hits = hits[:max_clips]
    if not hits:
        return {"summary": "No matching clips were found.", "report": None, "clips_used": [], "model": None}
    client = genai.Client()
    parts = [build_prompt(query, hits)]
    used = []
    for i, h in enumerate(hits, 1):
        try:
            parts.append(f"Keyframe of Result {i}:")
            parts.append(types.Part.from_bytes(data=keyframe_jpeg(h["clip_path"]), mime_type="image/jpeg"))
            used.append(h["clip_id"])
        except Exception as e:
            parts.append(f"(keyframe for Result {i} unavailable: {e})")
    name = model_name()
    last = None
    for attempt in range(4):                      # retry transient 429/503 errors
        try:
            resp = client.models.generate_content(model=name, contents=parts,
                                                  config=types.GenerateContentConfig(
                                                      temperature=0.2, response_mime_type="application/json"))
            text = (resp.text or "").strip()
            report = parse_report(text)
            summary = report_to_text(report) if report else re.sub(r"\*\*", "", text)
            return {"summary": summary, "report": report, "clips_used": used, "model": name}
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:300]}")
