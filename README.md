# Traffic clip search

Search Indian traffic footage in plain English. Pipeline: YOLOv8 + ByteTrack detection, CLIP embeddings of 3-second clips (scene and per-vehicle), Qdrant vector search, rule-based query understanding, Gemini explanations.

## Run
1. Open Docker Desktop, then `docker-compose up -d`
2. `source venv/bin/activate`
3. `uvicorn src.api.main:app --port 8000`
4. Open http://localhost:8000

`.env` needs `GEMINI_API_KEY=...` (never commit it; see `.env.example`).

## Rebuild the data
- Add videos to `data/indian_videos/`, then `python -m indian.ingest_video`
- Detection scores: `python -m indian.eval_detection --model models/indian_best.pt --label indian_ft --idmap ours`
- Image search scores: `python -m indian.retrieval eval`
- Checks: `python -m eval.integration_test`

## Layout
`web/index.html` frontend · `src/api/main.py` API · `src/retrieval/service.py` hybrid search · `src/rag/` Gemini summaries and number check · `indian/` dataset, training, eval, video ingest · `eval/` tests · `models/` trained weights.

## Limits
Counts are detector output and can be off in dense or night scenes. UVH-26 has no video, so clip search was checked by eye. Stock clips (Pexels) need credit.
