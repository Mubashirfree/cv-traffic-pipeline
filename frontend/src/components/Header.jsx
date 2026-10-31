const TABS = [
  { id: "search", label: "Search" },
  { id: "eval", label: "How well it works" },
];

function Logo() {
  // Lane markings on a yellow tile.
  return (
    <span className="logo" aria-hidden="true">
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="#231a00" strokeWidth="2" strokeLinecap="round">
        <path d="M9 1v3M9 7.5v3M9 14v3" />
        <path d="M3 1l-2 16M15 1l2 16" opacity=".55" />
      </svg>
    </span>
  );
}

export default function Header({ tab, onTab }) {
  return (
    <div className="topbar">
      <div className="brand"><Logo />Traffic clip search</div>
      <nav className="tabs" role="tablist" aria-label="Sections">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => onTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
    </div>
  );
}
