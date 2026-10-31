"""Grounded report prompt + parsing of the model's JSON answer."""
import json
import re

SYSTEM_RULES = """You are a traffic analyst writing a short, clear report for someone who is not an expert.

You get a user question, up to 3 retrieved clips ("Result 1", "Result 2", ...), one keyframe image per clip (same order),
and automatic detector output (vehicle counts) for each clip.

Return ONLY a JSON object, no markdown, in exactly this shape:
{
  "verdict": "yes" | "partly" | "no",
  "headline": "one sentence, at most 20 words, that directly answers the question",
  "clips": [
    {"result": 1, "title": "2 to 6 words naming this scene", "description": "one sentence, at most 28 words: road type, lighting, how crowded, main vehicle types, with friendly approximate numbers", "matches": true}
  ],
  "observations": ["at most 2 items, each at most 25 words: what the results share or how they differ"],
  "caveats": ["only if needed, at most 2 items, each at most 25 words"]
}

RULES
- One entry in "clips" per result, in order. "matches" is true only if that clip really fits the question.
- Put in "caveats" any place where a keyframe clearly disagrees with the detector counts (for example vehicles visible but none detected),
  or where the results do not really match the question. Use [] if there is nothing to flag.
- Counts come from an automatic detector and can be wrong; if a keyframe clearly disagrees, trust the image and say so in caveats.
- Never print file names, ids or raw timestamps. Refer to clips as "Result 1", "Result 2" only inside the text.
- Mention only things visible in the images or listed in the detector output. Do not invent vehicle types, numbers, weather or causes.
- Plain sentences. No bold, no bullet characters."""


def build_prompt(query, hits):
    lines = [SYSTEM_RULES, "", f"USER QUESTION: {query}", "", "DETECTOR OUTPUT (images follow in this order):"]
    for i, h in enumerate(hits, 1):
        counts = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(h["counts"].items())) or "no objects detected"
        light = f", lighting: {h['lighting']}" if h.get("lighting") else ""
        lines.append(f"Result {i}: about {h['timestamp']} seconds into the video{light}. Detected: {counts}.")
    return "\n".join(lines)


def _strs(x, limit=3):
    return [str(i).strip() for i in (x if isinstance(x, list) else []) if str(i).strip()][:limit]


def parse_report(text):
    """Model text -> normalised report dict, or None if it is not usable JSON."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.M).strip()
    d = None
    braces = re.search(r"\{.*\}", t, re.S)
    for candidate in (t, braces.group() if braces else None):
        if not candidate:
            continue
        try:
            d = json.loads(candidate)
            break
        except ValueError:
            continue
    if not isinstance(d, dict) or not d.get("headline"):
        return None
    verdict = str(d.get("verdict", "")).lower()
    clips = []
    for i, c in enumerate(d.get("clips") or [], 1):
        if isinstance(c, dict):
            clips.append({"result": int(c.get("result") or i), "title": str(c.get("title", "")).strip(),
                          "description": str(c.get("description", "")).strip(), "matches": bool(c.get("matches", True))})
    return {"verdict": verdict if verdict in ("yes", "partly", "no") else "partly",
            "headline": str(d["headline"]).strip(), "clips": clips,
            "observations": _strs(d.get("observations")), "caveats": _strs(d.get("caveats"), 2)}


def report_to_text(r):
    """Flat text of the report (without caveats) for the automatic number check."""
    parts = [r["headline"]] + [f"Result {c['result']}: {c['description']}" for c in r["clips"]] + r["observations"]
    return " ".join(parts)
