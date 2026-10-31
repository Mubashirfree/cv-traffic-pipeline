import { useState } from "react";
import { analyze } from "../lib/api.js";
import ReportView from "./ReportView.jsx";

// "Write a report" button + the report it produces. Resets when the search results change (via key in the parent).
export default function ReportPanel({ request }) {
  const [busy, setBusy] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  async function write() {
    setBusy(true); setError(""); setData(null);
    try { setData(await analyze(request)); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  return (
    <>
      <div className="explain-bar">
        <button className="btn dark" type="button" onClick={write} disabled={busy}>
          {busy ? "Writing report…" : "Write a report on these results"}
        </button>
        <p>Gemini reads the top 3 clips and the detector counts, then summarises what they show.</p>
      </div>
      {error && <div className="notice err"><b>Could not write the report.</b> {error}</div>}
      {data && <ReportView query={request.query} data={data} />}
    </>
  );
}
