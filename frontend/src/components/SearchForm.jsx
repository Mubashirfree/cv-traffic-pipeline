import { EXAMPLES } from "../lib/constants.js";

export default function SearchForm({ query, setQuery, busy, onSubmit }) {
  return (
    <>
      <form className="searchbar" onSubmit={(e) => { e.preventDefault(); onSubmit(); }}>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Describe what you are looking for"
          placeholder="For example: more than 3 trucks at night"
          autoComplete="off"
        />
        <button className="btn" type="submit" disabled={busy}>{busy ? "Searching…" : "Search clips"}</button>
      </form>
      <div className="examples" aria-label="Example searches">
        {EXAMPLES.map((text) => (
          <button key={text} type="button" onClick={() => onSubmit(text)}>{text}</button>
        ))}
      </div>
    </>
  );
}
