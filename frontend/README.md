# Frontend

React + Vite single-page app for the traffic clip search API.

    src/
      main.jsx            entry point
      App.jsx             header + the two pages
      pages/              SearchPage, EvaluationPage
      components/         Header, SearchForm, SearchOptions, ResultCard, CountMix, ExplainPanel, StatusBanner
      lib/                api.js (backend calls), constants.js, format.js
      styles.css          all styles

## Commands (run inside `frontend/`)
- `npm install` once
- `npm run dev` develop at http://localhost:5173 (needs the API running on :8000)
- `npm run build` writes `dist/`, which the API serves at http://localhost:8000
