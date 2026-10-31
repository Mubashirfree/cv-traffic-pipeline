import CountMix from "./CountMix.jsx";
import { mediaUrl } from "../lib/api.js";
import { niceCamera, niceTime } from "../lib/format.js";

const isImage = (path) => /\.(jpe?g|png)$/i.test(path || "");

export default function ResultCard({ hit, rank }) {
  const camera = niceCamera(hit.camera_id);
  const night = hit.lighting === "night";
  const pct = Math.round((hit.score || 0) * 100);

  return (
    <li className="card">
      <div className="media">
        {isImage(hit.clip_path) ? (
          <img src={mediaUrl(hit.clip_path)} alt={`Frame from ${camera}`} loading="lazy" />
        ) : (
          <video src={mediaUrl(hit.clip_path)} controls muted loop playsInline preload="metadata"
                 aria-label={`Clip ${rank} from ${camera}`} />
        )}
        <div className="badges">
          <span className="badge">#{rank}</span>
          {hit.lighting && <span className={night ? "badge night" : "badge"}>{night ? "Night" : "Day"}</span>}
        </div>
      </div>
      <div className="body">
        <div className="row">
          <span className="cam" title={hit.camera_id}>{camera}</span>
          <span className="when">{niceTime(hit.timestamp)}</span>
        </div>
        <CountMix counts={hit.counts || {}} />
        <div className="match">
          <span>Match {pct}%</span>
          <div className="meter" aria-hidden="true"><i style={{ width: `${pct}%` }} /></div>
        </div>
      </div>
    </li>
  );
}
