import { CLASSES, CLASS_NAMES } from "./constants.js";

// Turns a file-style source name ("15128491_2160_3840_60fps") into something readable.
export function niceCamera(id) {
  if (!id) return "Unknown camera";
  const words = id.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim().split(" ");
  const kept = words.filter((w) => !/^(\d{3,4}|\d+fps|uhd|hd|fhd)$/i.test(w) || w === words[0]);
  const text = kept.join(" ");
  if (/^\d+$/.test(kept[0])) return `Video ${kept[0]}`;
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function niceTime(t) {
  if (t == null || t === "") return "";
  if (typeof t === "number") return `at ${t.toFixed(0)} s`;
  const m = String(t).match(/\d+:\d+:\d+/);
  return m ? m[0] : `at ${t} s`;
}

export function countLabel(cls, n) {
  return `${n} ${CLASS_NAMES[cls][n === 1 ? 0 : 1]}`;
}

export function totalVehicles(counts) {
  return CLASSES.reduce((sum, c) => sum + (counts[c] || 0), 0);
}
