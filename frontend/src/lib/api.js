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
