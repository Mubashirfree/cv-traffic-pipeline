import { useEffect, useState } from "react";
import { getMetrics } from "../lib/api.js";
import { MODEL_LABELS } from "../lib/constants.js";

const isNum = (x) => typeof x === "number" && Number.isFinite(x);
const pickMap = (d) => ["map", "mAP", "map50_95", "mAP50-95", "map_50_95", "overall_map"].map((k) => d[k]).find(isNum);
const INDEX_NAMES = { indian_video: "Video clips", indian: "Street photos", detrac: "Highway clips" };

function Table({ rows }) {
  const [head, ...body] = rows;
  return (
    <div className="tblwrap">
      <table className="tbl">
        <thead><tr>{head.map((c, i) => <th key={i} scope="col" className={i ? "n" : ""}>{c}</th>)}</tr></thead>
        <tbody>
          {body.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className={j ? "n" : ""}>{c}</td>)}</tr>)}
        </tbody>
      </table>
    </div>
  );
}

function DetectionBars({ detection }) {
  const rows = detection.map((d) => ({ d, v: pickMap(d) })).filter((r) => r.v != null);
  if (!rows.length) return <p className="none">Detection results are not available yet. Run the evaluation step in the README.</p>;
  return rows.map(({ d, v }, i) => (
    <div className="bar" key={d.label || i}>
      <span className="l">{MODEL_LABELS[d.label] || d.label || "Model"}</span>
      <div className="track"><div className={i === rows.length - 1 ? "fill best" : "fill"} style={{ width: `${Math.min(100, v * 100)}%` }} /></div>
      <b>{v.toFixed(2)}</b>
    </div>
  ));
}

export default function EvaluationPage({ header, active }) {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");

  // Load once, the first time the tab is opened.
  useEffect(() => {
    if (active && !metrics && !error) getMetrics().then(setMetrics).catch((e) => setError(e.message));
  }, [active, metrics, error]);

  const shell = (content) => (
    <>
      <div className="hero" style={{ paddingBottom: 28 }}>
        <div className="wrap">
          {header}
          <div className="hero-copy" style={{ paddingBottom: 0 }}>
            <h2>How well it works</h2>
            <p className="sub">Measured results for the detector and the search, plus what the system cannot do yet.</p>
          </div>
        </div>
      </div>
      <main className="wrap eval">{content}</main>
    </>
  );

  if (error) return shell(<p className="none">{error}</p>);
  if (!metrics) return shell(<p className="none">Loading results…</p>);

  const index = Object.entries(metrics.index || {});
  const retrievalRows = metrics.retrieval?.rows;

  return shell(
    <>
      <h2>Vehicle detection accuracy</h2>
      <p className="sub">Measured on held-out Indian street photos. A higher score means the detector finds and labels vehicles more reliably. 1.00 would be perfect.</p>
      <div className="panel"><DetectionBars detection={metrics.detection} /></div>

      <h2>Search quality</h2>
      <p className="sub">How often the right images come back for each way of searching.</p>
      {retrievalRows?.length > 1
        ? <div className="panel"><Table rows={retrievalRows} /></div>
        : <p className="none">Search results are not available yet. Run: python -m indian.retrieval eval</p>}

      <h2>What is indexed</h2>
      {index.length ? (
        <div className="stats">
          {index.map(([k, v]) => <div className="stat" key={k}><b>{v}</b>{INDEX_NAMES[k] || k}</div>)}
        </div>
      ) : <p className="none">Index counts need Qdrant running (docker-compose up -d).</p>}

      <h2>Known limits</h2>
      <ul className="limits">
        <li>Vehicle counts come from an automatic detector, so they can be off in dense traffic or at night.</li>
        <li>The Indian training photos contain no video, so tracking and clip search on videos was checked by eye, not scored.</li>
        <li>Colour is easily lost when many vehicles are averaged together.</li>
        <li>Explanations are written by an AI model from the detector's counts. They are checked automatically for numbers that do not match.</li>
      </ul>
    </>
  );
}
