import { useEffect, useMemo, useRef, useState } from "react";
import {
  JobResponse,
  PageResult,
  pageSourceUrl,
  SectionResult,
  YoloSectionsResponse,
} from "../api";
import DocumentSectionOverlay from "./DocumentSectionOverlay";
import YoloSectionsPanel from "./YoloSectionsPanel";
import {
  collectScalarFields,
  collectTables,
  DisplayTable,
  formatCell,
  formatFieldLabel,
  sectionDisplayTitle,
  tableFromFields,
} from "../resultsFormat";

type Tab = "sections" | "kv" | "tables" | "json" | "yolo";

type Props = {
  job: JobResponse;
  onNewDoc: () => void;
  onCopyJson: () => void;
  onDownloadJson: () => void;
};

function yoloDetectionsToOverlay(response: YoloSectionsResponse): SectionResult[] {
  return response.detections.map((detection) => ({
    index: detection.index,
    label: detection.label || detection.class,
    bounds: detection.bounds,
    kept: true,
    filter_reason: null,
    ocr_preview: `${detection.class} · ${(detection.confidence * 100).toFixed(0)}%`,
    fields: null,
  }));
}

function KeyValueTable({ rows }: { rows: [string, string][] }) {
  if (!rows.length) return <p className="empty">No key-value fields in this region.</p>;
  return (
    <table className="kv-table">
      <thead>
        <tr>
          <th>Field</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td className="field-key">{formatFieldLabel(k)}</td>
            <td className="field-val">{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function TablesBlock({ tables }: { tables: DisplayTable[] }) {
  if (!tables.length) return null;
  return (
    <>
      {tables.map((table) => (
        <div key={table.id} className="section-table-block">
          <p className="section-table-label">Table · {table.rows.length} rows</p>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  {table.columns.map((col) => (
                    <th key={col}>{formatFieldLabel(col)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, rowIdx) => (
                  <tr key={rowIdx}>
                    {table.columns.map((col) => (
                      <td key={col}>{formatCell(row[col])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </>
  );
}

function SectionResultCard({
  section,
  pageIndex,
  selected,
  onSelect,
  setCardRef,
}: {
  section: SectionResult;
  pageIndex: number;
  selected: boolean;
  onSelect: () => void;
  setCardRef: (el: HTMLElement | null) => void;
}) {
  const fields = section.fields as Record<string, unknown> | null;
  const scalars = collectScalarFields(fields ?? undefined);
  const table = tableFromFields(
    fields,
    sectionDisplayTitle(section),
    `p${pageIndex}-s${section.index}`,
  );
  const preview = section.ocr_preview?.trim();

  return (
    <article
      ref={setCardRef}
      className={`section-result-card${selected ? " section-result-card--selected" : ""}`}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
      role="button"
      tabIndex={0}
      aria-pressed={selected}
    >
      <h3 className="section-result-card__title">{sectionDisplayTitle(section)}</h3>
      {scalars.length > 0 && <KeyValueTable rows={scalars} />}
      {table && <TablesBlock tables={[table]} />}
      {!scalars.length && !table && preview && (
        <p className="section-result-card__preview">{preview}</p>
      )}
      {!scalars.length && !table && !preview && (
        <p className="empty">No extracted fields for this region.</p>
      )}
    </article>
  );
}

export default function ResultsView({ job, onNewDoc, onCopyJson, onDownloadJson }: Props) {
  const [tab, setTab] = useState<Tab>("sections");
  const [pageIdx, setPageIdx] = useState(0);
  const [selectedSection, setSelectedSection] = useState<number | null>(null);
  const [yoloResponse, setYoloResponse] = useState<YoloSectionsResponse | null>(null);
  const [yoloImageUrl, setYoloImageUrl] = useState<string | null>(null);
  const sectionRefs = useRef<Record<number, HTMLElement | null>>({});
  const pages = job.pages || [];
  const page: PageResult | undefined =
    pages.find((p) => p.page_index === pageIdx) ?? pages[0];

  const scalarFields = useMemo(() => collectScalarFields(job.merged_fields), [job.merged_fields]);
  const tables = useMemo(() => collectTables(job), [job]);

  useEffect(() => {
    setSelectedSection(null);
    setYoloResponse(null);
    setYoloImageUrl(null);
  }, [pageIdx]);

  useEffect(() => {
    if (selectedSection === null) return;
    const el = sectionRefs.current[selectedSection];
    el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selectedSection]);

  const tabs: { id: Tab; label: string }[] = [
    { id: "sections", label: "By region" },
    { id: "yolo", label: "YOLO Sections" },
    { id: "kv", label: "Key value" },
    { id: "tables", label: "Tables" },
    { id: "json", label: "JSON" },
  ];

  const jsonText = useMemo(() => JSON.stringify(job, null, 2), [job]);
  const yoloOverlaySections = useMemo(
    () => (yoloResponse ? yoloDetectionsToOverlay(yoloResponse) : []),
    [yoloResponse],
  );

  const selectSection = (index: number) => {
    setSelectedSection((prev) => (prev === index ? null : index));
    if (tab !== "sections" && tab !== "yolo") setTab("sections");
  };

  const docImageUrl =
    tab === "yolo" && yoloImageUrl?.startsWith("blob:")
      ? yoloImageUrl
      : page
        ? pageSourceUrl(job.job_id, page.page_index)
        : "";

  return (
    <div className="results-view">
      <header className="results-topbar">
        <div className="results-title">
          <button type="button" className="btn-ghost" onClick={onNewDoc}>
            ← New document
          </button>
          <div>
            <h1>{job.source?.filename || "Document"}</h1>
            <p className="meta">
              {job.source?.pages || pages.length} page{(job.source?.pages || 1) !== 1 ? "s" : ""}
              {page ? ` · ${page.sections.length} regions` : ""}
            </p>
          </div>
        </div>
      </header>

      <div className="stats-row">
        <div className="stat">
          <span className="stat-val">{scalarFields.length}</span>
          <span className="stat-label">Fields</span>
        </div>
        <div className="stat">
          <span className="stat-val">{tables.length}</span>
          <span className="stat-label">Tables</span>
        </div>
        <div className="stat">
          <span className="stat-val">{page?.sections.length ?? 0}</span>
          <span className="stat-label">Regions</span>
        </div>
      </div>

      <div className="textract-layout">
        <div className="doc-panel">
          <div className="panel-head">
            <span>Document</span>
            {pages.length > 1 && (
              <div className="page-tabs">
                {pages.map((p) => (
                  <button
                    key={p.page_index}
                    type="button"
                    className={pageIdx === p.page_index ? "active" : ""}
                    onClick={() => setPageIdx(p.page_index)}
                  >
                    Page {p.page_index + 1}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="doc-preview doc-preview--interactive">
            {page && (
              <DocumentSectionOverlay
                imageUrl={docImageUrl}
                sections={tab === "yolo" && yoloOverlaySections.length ? yoloOverlaySections : page.sections}
                selectedIndex={selectedSection}
                onSelect={selectSection}
                boxClassName={tab === "yolo" ? "doc-section-box doc-section-box--yolo" : "doc-section-box"}
              />
            )}
          </div>
          <p className="legend">
            <span className="legend-hint">
              {tab === "yolo"
                ? "YOLO-format section bounds (pixel xyxy + normalized cxcywh)"
                : "Click a region to highlight its extracted fields"}
            </span>
          </p>
        </div>

        <div className="data-panel">
          <div className="panel-head tabs">
            {tabs.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                className={tab === id ? "active" : ""}
                onClick={() => setTab(id)}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="data-body">
            {tab === "sections" && page && (
              <div className="sections-results-panel">
                {page.sections.length === 0 ? (
                  <p className="empty">No regions detected on this page.</p>
                ) : (
                  page.sections.map((section) => (
                    <SectionResultCard
                      key={section.index}
                      section={section}
                      pageIndex={page.page_index}
                      selected={selectedSection === section.index}
                      onSelect={() => selectSection(section.index)}
                      setCardRef={(el) => {
                        sectionRefs.current[section.index] = el;
                      }}
                    />
                  ))
                )}
              </div>
            )}

            {tab === "yolo" && page && (
              <YoloSectionsPanel
                jobId={job.job_id}
                pageIndex={page.page_index}
                onResponse={({ response, imageUrl }) => {
                  setYoloResponse(response);
                  setYoloImageUrl(imageUrl);
                }}
              />
            )}

            {tab === "kv" && (
              <div className="all-fields-panel">
                <KeyValueTable rows={scalarFields} />
              </div>
            )}

            {tab === "tables" && (
              <div className="all-fields-panel">
                {tables.length > 0 ? (
                  tables.map((table) => (
                    <section key={table.id} className="fields-section">
                      <h3 className="fields-section-title">{table.label}</h3>
                      <TablesBlock tables={[table]} />
                    </section>
                  ))
                ) : (
                  <p className="empty">No tables found.</p>
                )}
              </div>
            )}

            {tab === "json" && (
              <div className="json-panel">
                <div className="json-toolbar">
                  <span className="json-filename mono">{job.job_id || "result"}.json</span>
                  <div className="json-actions">
                    <button type="button" className="btn-compact" onClick={onCopyJson}>
                      Copy
                    </button>
                    <button type="button" className="btn-compact btn-compact--primary" onClick={onDownloadJson}>
                      Download
                    </button>
                  </div>
                </div>
                <pre className="json-pre">{jsonText}</pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
