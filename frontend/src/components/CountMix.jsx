import { CLASSES } from "../lib/constants.js";
import { countLabel, totalVehicles } from "../lib/format.js";

// Stacked bar + legend showing the vehicle mix the detector found in one clip.
export default function CountMix({ counts }) {
  const total = totalVehicles(counts);
  const present = CLASSES.filter((c) => counts[c]);

  if (total === 0) {
    return (
      <p className="nodet" title="The detector can miss vehicles in fog, darkness or unusual camera angles.">
        The detector found no vehicles in this clip.
      </p>
    );
  }
  const summary = present.map((c) => countLabel(c, counts[c])).join(", ");

  return (
    <div>
      <div className="total">{total} vehicle{total === 1 ? "" : "s"} detected</div>
      <div className="mix" role="img" aria-label={summary}>
        {present.map((c) => (
          <i key={c} style={{ width: `${(100 * counts[c]) / total}%`, background: `var(--${c})` }} />
        ))}
      </div>
      <ul className="legend">
        {present.map((c) => (
          <li key={c}><span className="dot" style={{ background: `var(--${c})` }} />{countLabel(c, counts[c])}</li>
        ))}
      </ul>
    </div>
  );
}
