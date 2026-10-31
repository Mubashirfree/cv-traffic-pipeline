import { CLASSES } from "../lib/constants.js";
import { countLabel } from "../lib/format.js";
import { keyframeUrl } from "../lib/api.js";

const VERDICT = { yes: "Yes", partly: "Partly", no: "No" };

function MiniCounts({ counts }) {
  const present = CLASSES.filter((c) => counts?.[c]);
  if (!present.length) return <div className="mini"><span>No vehicles detected</span></div>;
  return (
    <div className="mini">
      {present.map((c) => (
        <span key={c}><i className="dot" style={{ background: `var(--${c})` }} />{countLabel(c, counts[c])}</span>
      ))}
    </div>
  );
}

export default function ReportView({ query, data }) {
  const { report, summary, flags = [], results = [] } = data;

  // The model did not return usable structure: show its text plainly rather than nothing.
  if (!report) {
    return (
      <article className="report">
        <div className="report-head"><div><div className="report-kicker">Report</div><div className="report-q">{query}</div></div></div>
        <p className="plain">{summary}</p>
      </article>
    );
  }

  return (
    <article className="report" aria-label="Report">
      <header className="report-head">
        <div>
          <div className="report-kicker">Report for your search</div>
          <div className="report-q">“{query}”</div>
        </div>
        <span className={`verdict ${report.verdict}`}>{VERDICT[report.verdict]}</span>
      </header>

      <h3 className="report-headline">{report.headline}</h3>

      {report.clips.length > 0 && (
        <section className="report-sec">
          <h3>Clip by clip</h3>
          {report.clips.map((c) => {
            const hit = results[c.result - 1];
            return (
              <div className="clip-row" key={c.result}>
                {hit ? <img src={keyframeUrl(hit.clip_path)} alt={`Still from result ${c.result}`} loading="lazy" /> : <div />}
                <div>
                  <div className="clip-title">
                    Result {c.result}{c.title ? ` · ${c.title}` : ""}
                    <span className={c.matches ? "fit ok" : "fit weak"}>{c.matches ? "Fits your search" : "Weak match"}</span>
                  </div>
                  <p className="clip-desc">{c.description}</p>
                  {hit && <MiniCounts counts={hit.counts} />}
                </div>
              </div>
            );
          })}
        </section>
      )}

      {report.observations.length > 0 && (
        <section className="report-sec">
          <h3>Overall</h3>
          <ul className="points">{report.observations.map((o, i) => <li key={i}>{o}</li>)}</ul>
        </section>
      )}

      {report.caveats.length > 0 && (
        <aside className="notes">
          <h3>Data notes</h3>
          <ul>{report.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </aside>
      )}

      <footer className="report-foot">
        {flags.length ? (
          <span className="check bad">Check before using: {flags.join("; ")}</span>
        ) : (
          <span className="check ok">Every number matches the detector’s counts</span>
        )}
        <span className="no-print"><button className="btn line" type="button" onClick={() => window.print()}>Print or save as PDF</button></span>
      </footer>
    </article>
  );
}
