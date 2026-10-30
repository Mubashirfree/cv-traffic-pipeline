#!/usr/bin/env bash
# Shows what is on this machine. Run from the project root:  bash scripts/status.sh
ok(){ printf "  [ok]      %s\n" "$1"; }; no(){ printf "  [MISSING] %s\n" "$1"; }
echo "== Files"
for f in models/detrac_best.pt models/indian_best.pt src/api/main.py src/retrieval/service.py src/rag/gemini_summarizer.py \
         indian/ingest_video.py indian/retrieval.py web/index.html .env .gitignore; do [ -e "$f" ] && ok "$f" || no "$f"; done
echo "== Data"
echo "  clips:  $(find data/indian_clips -name '*.mp4' 2>/dev/null | wc -l)"
echo "  videos: $(find data/indian_videos -type f 2>/dev/null | wc -l)"
echo "== Docker / Qdrant"
if curl -s localhost:6333/collections >/dev/null 2>&1; then
  curl -s localhost:6333/collections | python3 -c "import sys,json;[print('  ',c['name']) for c in json.load(sys.stdin)['result']['collections']]"
else echo "  Qdrant not reachable. Open Docker Desktop, then: docker-compose up -d"; fi
echo "== Git"
git status -sb 2>/dev/null | head -15 || echo "  not a git repo"
git log --oneline -5 2>/dev/null
git check-ignore -q .env && echo "  .env is ignored (good)" || echo "  WARNING: .env is NOT ignored"
