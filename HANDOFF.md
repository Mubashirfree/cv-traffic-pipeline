# Handoff
Project: ~/projects/Cv_project (WSL2, venv/). Internship: search Indian CCTV-style traffic clips by text.

## Done
- Detectors: models/detrac_best.pt, models/indian_best.pt (mAP 0.409 COCO -> 0.798 Indian fine-tune, UVH-26).
- Qdrant collections: UA-DETRAC, uvh26_items (800 pts), indian_clips (16 videos, 94 clips).
- Hybrid search (CLIP + count/lighting filters + re-rank), Gemini explanations + number check.
- Web frontend at / (web/index.html), /media, /metrics endpoints.
## Rules
Create files with heredocs; run from project root with `python -m ...`; never commit .env; Docker Desktop open, `docker-compose`; no paid services, no HF account.
## Open items
1. Trucks look confused with buses in video (detector, not search). Need a screenshot of a bad truck result.
2. Confirm "Scene + vehicles together" beats scene alone: `python -m indian.retrieval eval`.
3. Re-run `python -m eval.rag_eval` with the new prompt; pin requirements.txt.
4. Report/slides later.
## Each new chat
Start with: "Read HANDOFF.md, run bash scripts/status.sh (paste output)". End with git add/commit/push and update this file.
