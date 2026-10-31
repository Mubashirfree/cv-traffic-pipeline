import { useEffect, useState } from "react";
import SearchForm from "../components/SearchForm.jsx";
import FilterBar from "../components/FilterBar.jsx";
import ResultCard from "../components/ResultCard.jsx";
import ReportPanel from "../components/ReportPanel.jsx";
import { search, getCameras } from "../lib/api.js";

const DEFAULT_OPTIONS = { dataset: "indian_video", camera: "", mode: "both", topK: 6, smart: true };

export default function SearchPage({ header, libraryVersion, focusCamera }) {
  const [query, setQuery] = useState("");
  const [options, setOptions] = useState(DEFAULT_OPTIONS);
  const [cameras, setCameras] = useState([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);     // { kind: "info"|"warn"|"err", title, text }
  const [response, setResponse] = useState(null); // { hits, request, understood, relaxed }

  // Reload the video list when the footage source changes or the library changes.
  useEffect(() => {
    getCameras(options.dataset).then((d) => setCameras([...d.cameras].sort())).catch(() => setCameras([]));
  }, [options.dataset, libraryVersion]);

  // "Search this video" in the library jumps here with that video selected.
  useEffect(() => {
    if (focusCamera) setOptions((o) => ({ ...o, dataset: "indian_video", camera: focusCamera }));
  }, [focusCamera]);

  async function run(text) {
    const q = (text ?? query).trim();
    if (text) setQuery(text);
    if (!q) {
      setNotice({ kind: "warn", title: "Type a description first.", text: "Try one of the examples under the search box." });
      return;
    }
    const request = {
      query: q,
      dataset: options.dataset,
      embedding_type: options.mode,
      top_k: Number(options.topK),
      rerank: options.smart,
      camera_id: options.camera || null,
    };
    setBusy(true); setResponse(null); setNotice(null);
    try {
      const d = await search(request);
      if (!d.results.length) {
        setNotice({ kind: "warn", title: "No clips matched.", text: "Try fewer conditions, pick All cameras, or turn off smart understanding." });
      } else {
        if (d.relaxed) setNotice({ kind: "warn", title: "Nothing matched every condition.", text: "These are the closest clips instead." });
        setResponse({ hits: d.results, request, understood: d.understood });
      }
    } catch (e) {
      setNotice({ kind: "err", title: "Search failed.", text: e.message });
    } finally {
      setBusy(false);
    }
  }

  const understood = response?.understood?.split(", ") ?? [];

  return (
    <>
      <div className="hero">
        <div className="wrap">
          {header}
          <div className="hero-copy">
            <h2>Search traffic footage the way you would describe it</h2>
            <p className="sub">Ask for vehicle counts, vehicle types, night or day, and empty or busy roads. Results are 3-second clips from Indian roads.</p>
          </div>
          <SearchForm query={query} setQuery={setQuery} busy={busy} onSubmit={run} />
        </div>
      </div>

      <main className="wrap">
        <FilterBar options={options} setOptions={setOptions} cameras={cameras} />
        {notice && (
          <div className={notice.kind === "info" ? "notice" : `notice ${notice.kind}`} role="status">
            <b>{notice.title}</b> {notice.text}
          </div>
        )}
        {response && (
          <>
            <div className="summary-row">
              <span className="lbl">{response.hits.length} clip{response.hits.length > 1 ? "s" : ""} for</span>
              {understood.map((u) => <span className="chip" key={u}>{u}</span>)}
            </div>
            <ul className="results">
              {response.hits.map((h, i) => <ResultCard key={h.clip_path + i} hit={h} rank={i + 1} />)}
            </ul>
            <ReportPanel key={response.hits.map((h) => h.clip_path).join()} request={response.request} />
          </>
        )}
      </main>
    </>
  );
}
