import { useRef, useState } from "react";
import { uploadVideo } from "../lib/api.js";

const ACCEPT = ".mp4,.mov,.avi,.mkv,.webm";
let nextId = 1;

// Drop area + upload queue. Files upload one after another; the server processes them in the background.
export default function UploadZone({ onUploaded }) {
  const [uploads, setUploads] = useState([]);   // { id, name, pct, state: "waiting"|"uploading"|"done"|"error", message }
  const [drag, setDrag] = useState(false);
  const input = useRef(null);
  const busy = useRef(false);
  const queue = useRef([]);

  const patch = (id, change) => setUploads((u) => u.map((x) => (x.id === id ? { ...x, ...change } : x)));

  async function drain() {
    if (busy.current) return;
    busy.current = true;
    while (queue.current.length) {
      const { id, file } = queue.current.shift();
      patch(id, { state: "uploading" });
      try {
        await uploadVideo(file, (p) => patch(id, { pct: Math.round(p * 100) }));
        patch(id, { state: "done", pct: 100 });
      } catch (e) {
        patch(id, { state: "error", message: e.message });
      }
      onUploaded();
    }
    busy.current = false;
  }

  function add(files) {
    const list = [...files].map((file) => ({ id: nextId++, file }));
    setUploads((u) => [...u, ...list.map(({ id, file }) => ({ id, name: file.name, pct: 0, state: "waiting" }))]);
    queue.current.push(...list);
    drain();
  }

  return (
    <section>
      <div
        className={drag ? "drop over" : "drop"}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); }}
      >
        <p className="drop-title">Drag videos here</p>
        <p className="muted">MP4, MOV, AVI, MKV or WEBM, up to 4 GB each. You can add several at once.</p>
        <button className="btn" type="button" onClick={() => input.current.click()}>Choose videos</button>
        <input ref={input} type="file" accept={ACCEPT} multiple hidden
               onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
      </div>

      {uploads.length > 0 && (
        <ul className="uploads" aria-label="Uploads">
          {uploads.map((u) => (
            <li key={u.id} className={`up ${u.state}`}>
              <span className="up-name">{u.name}</span>
              <span className="up-state">
                {u.state === "waiting" && "Waiting"}
                {u.state === "uploading" && `Uploading ${u.pct}%`}
                {u.state === "done" && "Uploaded. Processing continues below."}
                {u.state === "error" && u.message}
              </span>
              {u.state === "uploading" && <div className="meter"><i style={{ width: `${u.pct}%` }} /></div>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
