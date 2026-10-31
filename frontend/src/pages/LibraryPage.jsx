import { useCallback, useEffect, useState } from "react";
import UploadZone from "../components/UploadZone.jsx";
import VideoCard from "../components/VideoCard.jsx";
import { getLibrary, deleteVideo } from "../lib/api.js";

const WORKING = ["uploading", "queued", "processing"];

export default function LibraryPage({ header, active, onChanged, onSearchVideo }) {
  const [items, setItems] = useState(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try { setItems((await getLibrary()).items); setError(""); }
    catch (e) { setError(e.message); }
  }, []);

  useEffect(() => { if (active) refresh(); }, [active, refresh]);

  // Poll while anything is still being processed; tell the search page when something finishes.
  const working = items?.some((v) => WORKING.includes(v.status));
  useEffect(() => {
    if (!active || !working) return;
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, [active, working, refresh]);

  const readyCount = items?.filter((v) => v.status === "ready").length ?? 0;
  useEffect(() => { onChanged(); }, [readyCount]); // eslint-disable-line react-hooks/exhaustive-deps

  async function remove(video) {
    if (!window.confirm(`Delete "${video.name}" and its clips from the library? This cannot be undone.`)) return;
    try { await deleteVideo(video.id); } catch (e) { setError(e.message); }
    refresh();
  }

  return (
    <>
      <div className="hero" style={{ paddingBottom: 28 }}>
        <div className="wrap">
          {header}
          <div className="hero-copy" style={{ paddingBottom: 0 }}>
            <h2>Your video library</h2>
            <p className="sub">Add your own traffic videos. Each one is cut into 3-second clips, analysed, and becomes searchable. Videos stay here between visits.</p>
          </div>
        </div>
      </div>
      <main className="wrap lib">
        <UploadZone onUploaded={refresh} />
        {error && <div className="notice err" role="alert"><b>Library problem.</b> {error}</div>}
        <h2 className="lib-title">{items ? `${items.length} video${items.length === 1 ? "" : "s"} in the library` : "Loading…"}</h2>
        {items && items.length === 0 && <p className="none">The library is empty. Add a video above to get started.</p>}
        <ul className="results">
          {items?.map((v) => <VideoCard key={v.id} video={v} onSearch={onSearchVideo} onDelete={remove} />)}
        </ul>
      </main>
    </>
  );
}
