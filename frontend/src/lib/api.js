// Thin wrapper around the FastAPI backend. Every function throws an Error with a message fit to show the user.

async function request(path, options) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error("Cannot reach the server. Start it with: uvicorn src.api.main:app --port 8000");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

const post = (path, body) =>
  request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const search = (params) => post("/search", params);
export const analyze = (params) => post("/analyze", { ...params, max_clips: 3 });
export const getCameras = (dataset) => request(`/cameras?dataset=${dataset}`);
export const getMetrics = () => request("/metrics");
export const mediaUrl = (path) => `/media?path=${encodeURIComponent(path)}`;
export const keyframeUrl = (path) => `/keyframe?path=${encodeURIComponent(path)}`;

// ---- video library ----
export const getLibrary = () => request("/library");
export const deleteVideo = (id) => request(`/library/${encodeURIComponent(id)}`, { method: "DELETE" });

// Sends one file as the raw request body. fetch() cannot report upload progress, so this uses XMLHttpRequest.
export function uploadVideo(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/library/upload?name=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error(data.detail || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Cannot reach the server. Is it running?"));
    xhr.send(file);
  });
}
