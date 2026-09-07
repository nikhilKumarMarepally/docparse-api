type Props = {
  dragOver: boolean;
  busy: boolean;
  onBrowse: () => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: () => void;
  onDrop: (e: React.DragEvent) => void;
};

const FEATURES = [
  {
    title: "Any document",
    desc: "Upload forms, applications, invoices, and scans — PDF or image.",
    icon: "◎",
  },
  {
    title: "Structured output",
    desc: "Get clean field names and values ready to use in your apps.",
    icon: "▦",
  },
  {
    title: "Smart parsing",
    desc: "Ignores fine print and disclaimers so you only see what matters.",
    icon: "⊘",
  },
  {
    title: "Export anywhere",
    desc: "Review in the browser or download JSON in one click.",
    icon: "✦",
  },
];

export default function HomePage({
  dragOver,
  busy,
  onBrowse,
  onDragOver,
  onDragLeave,
  onDrop,
}: Props) {
  return (
    <div className="home">
      <section className="hero">
        <div className="hero-inner">
          <p className="eyebrow">Document intelligence</p>
          <h1>
            Turn documents into
            <br />
            <span className="gradient-text">structured data</span>
          </h1>
          <p className="hero-sub">
            Upload a file and get back the fields you care about — names, dates,
            IDs, amounts, and more — in a clear, readable format.
          </p>

          <div
            className={`hero-upload ${dragOver ? "drag-over" : ""} ${busy ? "busy" : ""}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={() => !busy && onBrowse()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && !busy && onBrowse()}
          >
            <div className="upload-icon">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            {busy ? (
              <p className="upload-title">Reading your document…</p>
            ) : (
              <>
                <p className="upload-title">Drop your document here</p>
                <p className="upload-hint">PDF, PNG, JPG, TIFF — up to 20 MB</p>
                <button type="button" className="btn-primary" onClick={(e) => { e.stopPropagation(); onBrowse(); }}>
                  Choose file
                </button>
              </>
            )}
          </div>
        </div>
      </section>

      <section className="features">
        <h2>Why DocParse</h2>
        <div className="feature-grid">
          {FEATURES.map((f) => (
            <article key={f.title} className="feature-card">
              <span className="feature-icon">{f.icon}</span>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
