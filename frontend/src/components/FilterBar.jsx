import { DATASETS, MODES } from "../lib/constants.js";
import { niceCamera } from "../lib/format.js";

export default function FilterBar({ options, setOptions, cameras }) {
  const set = (key) => (e) =>
    setOptions({ ...options, [key]: e.target.type === "checkbox" ? e.target.checked : e.target.value });

  return (
    <section className="filters" aria-label="Search filters">
      <label className="field">Footage
        <select value={options.dataset} onChange={set("dataset")}>
          {DATASETS.map((d) => <option key={d.value} value={d.value}>{d.label}</option>)}
        </select>
      </label>
      <label className="field">Camera
        <select value={options.camera} onChange={set("camera")}>
          <option value="">All cameras</option>
          {cameras.map((c) => <option key={c} value={c}>{niceCamera(c)}</option>)}
        </select>
      </label>
      <label className="field">Compare
        <select value={options.mode} onChange={set("mode")}>
          {MODES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
        </select>
      </label>
      <label className="field">Results
        <select value={options.topK} onChange={set("topK")}>
          {[3, 6, 9, 12].map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
      </label>
      <label className="switch">
        <input type="checkbox" checked={options.smart} onChange={set("smart")} />
        Smart understanding
      </label>
      <p className="filters-hint">
        Scene finds the overall look (night, crowded junction). Vehicles finds what a vehicle looks like (a bus).
        Smart understanding turns counts and conditions in your text into filters; turn it off to compare image similarity only.
      </p>
    </section>
  );
}
