"""Automated first-pass screen for summaries (a manual check is still required)."""
import re

VEHICLE_WORDS = {
    "car": ["car", "cars", "hatchback", "sedan", "suv"],
    "bus": ["bus", "buses"],
    "truck": ["truck", "trucks", "lorry"],
    "auto_rickshaw": ["rickshaw", "rickshaws", "auto-rickshaw", "auto-rickshaws", "three-wheeler"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes", "scooter", "scooters", "two-wheeler"],
    "bicycle": ["bicycle", "bicycles", "cyclist", "cyclists"],
    "person": ["person", "people", "pedestrian", "pedestrians"],
    "van": ["van", "vans"],
}
NEG = re.compile(r"\b(no|not|without|none|nor|neither)\b", re.I)


def check(summary, hits):
    text = re.sub(r"\[[^\]]*\]", " ", summary)
    for h in hits:
        if h.get("clip_id"):
            text = text.replace(str(h["clip_id"]), " ")
    text = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", " ", text)
    text = re.sub(r"\b[Rr]esults?\s+\d+(\s*(,|and|&)\s*\d+)*", " ", text)
    present, allowed = set(), {0, 1, 2, 3}
    for h in hits:
        present |= set(h["counts"])
        allowed |= set(h["counts"].values())
        allowed.add(sum(h["counts"].values()))
        try:
            allowed |= {int(float(h["timestamp"]))}
        except (TypeError, ValueError):
            pass
    allowed.add(len(hits))
    allowed.add(sum(sum(h["counts"].values()) for h in hits))
    problems = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if NEG.search(sent):
            continue
        low = sent.lower()
        for cls, words in VEHICLE_WORDS.items():
            if cls not in present and any(re.search(rf"\b{re.escape(w)}\b", low) for w in words):
                problems.append(f"mentions '{cls}' not in detector output")
    for m in re.finditer(r"\b\d+\b", text):
        n = int(m.group())
        if not any(abs(n - a) <= max(1, 0.15 * a) for a in allowed):
            problems.append(f"number {n} not in detector output")
    return sorted(set(problems))
