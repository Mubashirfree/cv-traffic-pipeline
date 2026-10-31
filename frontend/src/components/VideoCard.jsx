import { mediaUrl, keyframeUrl } from "../lib/api.js";

const LABEL = { uploading: "Uploading", queued: "In the queue", processing: "Processing", ready: "Ready", failed: "Failed" };

export default function VideoCard({ video, onSearch, onDelete }) {
  const working = ["uploading", "queued", "processing"].includes(video.status);
  const pct = video.total ? Math.round((100 * video.done) / video.total) : 0;

  return (
    <li className="card vcard">
      <div className="media">
        {video.thumb
          ? <img src={keyframeUrl(video.thumb)} alt="" loading="lazy" />
          : <div className="placeholder">{working ? "Preparing…" : "No preview"}</div>}
        <div className="badges"><span className={`badge st-${video.status}`}>{LABEL[video.status] || video.status}</span></div>
      </div>
      <div className="body">
        <div className="row">
          <span className="cam" title={video.name}>{video.name}</span>
          <span className="when">{video.source === "upload" ? "Uploaded" : "Sample"}</span>
        </div>

        {working && (
          <div>
            <div className="total">{video.stage}{video.total > 1 ? ` · ${video.done} of ${video.total}` : ""}</div>
            <div className="meter big"><i style={{ width: `${pct}%` }} /></div>
          </div>
        )}
        {video.status === "failed" && <p className="nodet err-text">{video.error || "Processing failed."}</p>}
        {video.status === "ready" && (
          <p className="muted small">{video.clips} clips · about {video.seconds} seconds{video.size_mb ? ` · ${video.size_mb} MB` : ""}</p>
        )}

        <div className="actions">
          {video.status === "ready" && <button className="btn dark" type="button" onClick={() => onSearch(video.id)}>Search this video</button>}
          {!working && <button className="btn line" type="button" onClick={() => onDelete(video)}>Delete</button>}
        </div>
      </div>
    </li>
  );
}
