// kind: "info" | "warn" | "err"
export default function StatusBanner({ kind = "info", title, children }) {
  const cls = kind === "info" ? "status" : `status ${kind}`;
  return (
    <div role="status" aria-live="polite">
      <div className={cls}>
        {title && <b>{title} </b>}
        {children}
      </div>
    </div>
  );
}
