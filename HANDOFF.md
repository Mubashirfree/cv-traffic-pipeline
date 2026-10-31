# Handoff
Project: ~/projects/Cv_project (WSL2, venv/, Node via nvm). Internship: search Indian traffic clips by plain-English text.

## State
- Detectors in models/: detrac_best.pt, indian_best.pt (mAP 0.409 COCO baseline -> 0.798 on UVH-26 test).
- Qdrant collections: UA-DETRAC (`clips`), `uvh26_items` (800 pts), `indian_clips` (sample videos + uploads).
- Backend (FastAPI, src/api/main.py): /search, /analyze (structured report), /cameras, /keyframe, /media, /metrics,
  /library (list), /library/upload (raw body, one file per request), DELETE /library/{id}. Background worker in src/library/worker.py.
- Search (src/retrieval/service.py): CLIP + count/lighting filters; understands counts, "no X", "only/mostly X", night/day, empty/heavy.
- Report (src/rag/): Gemini returns JSON -> verdict, per-clip rows, notes; checked automatically against detector counts.
- Frontend: React + Vite in frontend/ (tabs Search, Library, How well it works). Build: `cd frontend && npm run build`.
- Library on disk (git-ignored): data/indian_videos (originals), data/indian_clips/<id> (clips), data/library/<id>.json (status).

## Rules
Create files with heredocs; run from project root with `python -m ...`; never commit .env; Docker Desktop open and `docker-compose`;
no paid services, no HF account; keep test_gemini.py (model name is read from it) or set GEMINI_MODEL.

## Open items
1. Detector misses vehicles in fog/night/portrait clips; trucks sometimes confused with buses.
2. Confirm Scene + vehicles beats scene alone: `python -m indian.retrieval eval`.
3. Pin requirements.txt (was overwritten by pip freeze); verify with a clean-clone test.
4. Remove unused weights/yolo26n.pt; final report/slides.
5. Library upload/processing was tested with a stand-in detector; test with a real video on the GPU.

## Every new chat
Start: "Read HANDOFF.md, run bash scripts/status.sh and paste the output." End: git add/commit/push and update this file.
