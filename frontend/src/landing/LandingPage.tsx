import { useEffect, useMemo, useRef, useState } from "react";
import type { AuthUser } from "../auth";
import DocumentPreview from "./DocumentPreview";
import FounderContactModal from "./FounderContactModal";
import {
  DEMO_DATA,
  DOC_TYPES,
  DocType,
  FEATURES,
  PRICING,
  STEPS,
  USE_CASES,
} from "./demoData";

type Props = {
  onStartExtracting: () => void;
  onUploadFile: (file: File) => void;
  onLogin: () => void;
  onSignOut?: () => void;
  user?: AuthUser | null;
  onFileDrop: (e: React.DragEvent) => void;
  dragOver: boolean;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: () => void;
  ready?: boolean;
};

type OutputTab = "extracted" | "json" | "confidence";

function scrollTo(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
}

function avgConfidence(conf: Record<string, number>) {
  const vals = Object.values(conf);
  if (!vals.length) return 0;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function fieldLabel(key: string) {
  return key
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (c) => c.toUpperCase())
    .trim();
}

const CURL_SNIPPET = `curl -X POST https://api.extracta.dev/v1/extract \\
  -H "Authorization: Bearer sk_..." \\
  -F file=@invoice.pdf \\
  -F type=invoice

{
  "invoice_number": "INV-1842",
  "total": 4250.00,
  "confidence": 0.97
}`;

export default function LandingPage({
  onStartExtracting,
  onUploadFile,
  onLogin,
  onSignOut,
  user,
  onFileDrop,
  dragOver,
  onDragOver,
  onDragLeave,
  ready,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [docType, setDocType] = useState<DocType>("invoice");
  const [hasUploaded, setHasUploaded] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState(100);
  const [yearly, setYearly] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);
  const [outputTab, setOutputTab] = useState<OutputTab>("extracted");
  const [toast, setToast] = useState<string | null>(null);
  const [founderOpen, setFounderOpen] = useState(false);
  const [editField, setEditField] = useState<string | null>(null);
  const [editedFields, setEditedFields] = useState<Record<string, string>>({});

  const demo = DEMO_DATA[docType];
  const fields = useMemo(
    () => ({ ...demo.fields, ...editedFields }),
    [demo.fields, editedFields],
  );
  const overallConf = useMemo(() => avgConfidence(demo.confidence), [demo]);

  const showToast = (msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast(null), 2800);
  };

  const runScan = () => {
    setScanning(true);
    setProgress(0);
    const timers: number[] = [];
    const tick = () => {
      setProgress((p) => {
        const next = p + 10 + Math.random() * 14;
        if (next < 100) timers.push(window.setTimeout(tick, 100));
        else {
          setScanning(false);
          setHasUploaded(true);
        }
        return Math.min(100, next);
      });
    };
    timers.push(window.setTimeout(tick, 60));
    return () => timers.forEach(clearTimeout);
  };

  useEffect(() => {
    setEditedFields({});
    setEditField(null);
    const cleanup = runScan();
    return cleanup;
  }, [docType]);

  const switchDocType = (id: DocType) => {
    setDocType(id);
    setHasUploaded(true);
    showToast(`Switched to ${DOC_TYPES.find((t) => t.id === id)?.label ?? id}`);
  };

  const handleDemoFile = (file?: File) => {
    if (file) {
      showToast(`Uploaded ${file.name}`);
      onUploadFile(file);
      return;
    }
    setHasUploaded(true);
    runScan();
    showToast("Sample document loaded");
  };

  const onDemoDrop = (e: React.DragEvent) => {
    e.preventDefault();
    onDragLeave();
    const file = e.dataTransfer.files[0];
    if (file) {
      onFileDrop(e);
      return;
    }
    handleDemoFile();
  };

  const copyJson = async () => {
    const payload = {
      document_type: docType,
      ...fields,
      line_items: demo.lineItems,
      metadata: { pages: 1, time_ms: 1142 },
    };
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    showToast("JSON copied to clipboard");
  };

  const statusText =
    progress < 40 ? "Detecting layout" : progress < 75 ? "Reading fields" : "Validating output";

  return (
    <div className="min-h-screen bg-[#fafaf9] text-zinc-900">
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-[#fafaf9]/80 border-b border-black/[0.06]">
        <div className="max-w-[1280px] mx-auto px-5 sm:px-8 h-[68px] flex items-center justify-between">
          <div className="flex items-center gap-10">
            <div className="flex items-center gap-2.5">
              <div className="w-[28px] h-[28px] rounded-[9px] bg-black text-white grid place-items-center font-bold text-[15px]">
                E
              </div>
              <span className="font-semibold tracking-[-0.02em] text-[18px]">Extracta</span>
              <span className="w-1.5 h-1.5 rounded-full bg-violet-600 ml-0.5 animate-pulse" />
            </div>
            <nav className="hidden lg:flex items-center gap-7 text-[14px] font-[500] text-zinc-500">
              <button type="button" onClick={() => scrollTo("product")} className="hover:text-zinc-900 transition-colors">
                Product
              </button>
              <button type="button" onClick={() => scrollTo("usecases")} className="hover:text-zinc-900 transition-colors">
                Solutions
              </button>
              <button type="button" onClick={() => scrollTo("demo")} className="hover:text-zinc-900 transition-colors">
                Developers
              </button>
              <button type="button" onClick={() => scrollTo("pricing")} className="hover:text-zinc-900 transition-colors">
                Pricing
              </button>
            </nav>
          </div>
          <div className="flex items-center gap-2.5">
            {ready === false && (
              <span className="hidden sm:inline text-[12px] text-amber-600 font-medium">Offline</span>
            )}
            <button
              type="button"
              onClick={() => {
                if (user && onSignOut) onSignOut();
                else onLogin();
              }}
              className="hidden sm:inline-flex h-9 px-4 rounded-full text-[14px] font-medium hover:bg-black/5 transition-colors"
            >
              {user ? user.name || user.email : "Login"}
            </button>
            <button
              type="button"
              onClick={onStartExtracting}
              className="h-9 px-5 rounded-full bg-black text-white text-[13.5px] font-semibold hover:bg-zinc-800 transition-colors shadow-[0_1px_2px_rgba(0,0,0,0.12)]"
            >
              Start Extracting
            </button>
            <button
              type="button"
              className="lg:hidden w-9 h-9 grid place-items-center rounded-full border border-black/10 bg-white"
              onClick={() => setMobileNav((v) => !v)}
              aria-label="Menu"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d={mobileNav ? "M6 18L18 6M6 6l12 12" : "M4 7h16M4 12h16M4 17h16"} />
              </svg>
            </button>
          </div>
        </div>
        {mobileNav && (
          <div className="lg:hidden border-t border-black/5 bg-white px-5 py-4 flex flex-col gap-3 text-[15px] font-medium">
            {[
              ["product", "Product"],
              ["usecases", "Solutions"],
              ["demo", "Developers"],
              ["pricing", "Pricing"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                className="text-left py-2"
                onClick={() => {
                  scrollTo(id);
                  setMobileNav(false);
                }}
              >
                {label}
              </button>
            ))}
          </div>
        )}
      </header>

      <main>
        {/* Hero */}
        <section className="relative overflow-hidden">
          <div className="max-w-[1280px] mx-auto px-5 sm:px-8 pt-10 sm:pt-[72px] pb-12 sm:pb-16">
            <div className="grid lg:grid-cols-[1.1fr_0.9fr] gap-10 lg:gap-8 items-center">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-black/[0.07] shadow-[0_1px_2px_rgba(0,0,0,0.04)] text-[12px] font-medium">
                  <span className="inline-flex w-5 h-5 rounded-full bg-violet-600 text-white grid place-items-center text-[10px]">
                    ✦
                  </span>
                  <span className="tracking-tight">New: Supports 100+ document types</span>
                  <span className="w-4 h-4 rounded-full bg-zinc-100 grid place-items-center">↗</span>
                </div>
                <h1 className="mt-6 text-[36px] sm:text-[56px] font-[700] leading-[0.95] tracking-[-0.04em]">
                  Extract data from any document.{" "}
                  <span className="text-zinc-400">In seconds.</span>
                </h1>
                <p className="mt-5 text-[16px] sm:text-[18px] leading-[1.5] text-zinc-500 max-w-[520px]">
                  AI that actually understands documents. No templates, no rules. Just upload and get structured data.
                </p>
                <div className="mt-7 flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={onStartExtracting}
                    className="h-11 px-6 rounded-full bg-black text-white text-[14px] font-semibold shadow-[0_4px_16px_rgba(0,0,0,0.15)] hover:bg-zinc-800 transition-all flex items-center gap-2"
                  >
                    Start free trial <span>→</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => showToast("Demo video — coming soon. Try live demo below")}
                    className="h-11 px-6 rounded-full bg-white border border-black/10 text-[14px] font-semibold hover:bg-zinc-50 transition-colors flex items-center gap-2"
                  >
                    <span className="w-6 h-6 rounded-full bg-black text-white grid place-items-center text-[10px]">▶</span>
                    Watch demo
                  </button>
                </div>
                <div className="mt-8 flex items-center gap-3 text-[12px] text-zinc-500">
                  <div className="flex -space-x-2">
                    {["A", "L", "S"].map((letter) => (
                      <div
                        key={letter}
                        className="w-7 h-7 rounded-full border-2 border-white bg-zinc-200 grid place-items-center text-[10px]"
                      >
                        {letter}
                      </div>
                    ))}
                  </div>
                  <span>
                    <b className="text-zinc-900">2,400+</b> teams extract 12M pages/mo • No credit card
                  </span>
                </div>
              </div>

              <div className="relative lg:h-[520px] h-[460px] flex items-center justify-center overflow-hidden lg:overflow-visible">
                <div className="absolute inset-0 bg-gradient-to-br from-violet-200/40 via-transparent to-indigo-100/30 blur-[40px] rounded-[32px]" />
                <div className="relative w-full max-w-[420px]">
                  <div
                    className="absolute -left-6 top-6 w-[220px] rounded-[14px] bg-white border border-black/[0.06] shadow-[0_12px_32px_rgba(0,0,0,0.08)] p-4 rotate-[-6deg]"
                    style={{ animation: "float 5s ease-in-out infinite" }}
                  >
                    <div className="text-[10px] font-bold tracking-widest text-zinc-400">RECEIPT</div>
                    <div className="mt-2 h-2 w-3/4 bg-zinc-100 rounded-full" />
                    <div className="mt-2 text-[22px] font-bold">$8.50</div>
                  </div>
                  <div
                    className="absolute -right-2 top-20 w-[200px] rounded-[14px] bg-white border border-black/[0.06] shadow-[0_12px_32px_rgba(0,0,0,0.08)] p-4 rotate-[5deg]"
                    style={{ animation: "float 6s ease-in-out infinite 0.5s" }}
                  >
                    <div className="text-[10px] font-bold tracking-widest text-zinc-400">PASSPORT</div>
                    <div className="mt-2 mono text-[10px] text-zinc-500">P&lt;USAMORGAN&lt;&lt;ALEX</div>
                  </div>
                  <div className="relative rounded-[18px] bg-white border border-black/[0.06] shadow-[0_24px_64px_rgba(0,0,0,0.12)] p-5 scale-[1.02]">
                    <p className="text-[11px] text-zinc-400">Invoice_2024.pdf</p>
                    <p className="mt-1 font-bold text-[16px]">ACME CORP</p>
                    <p className="text-[11px] text-zinc-400">EST 2018 • SF</p>
                    <div className="mt-4 grid grid-cols-2 gap-3 text-[12px]">
                      <div>
                        <span className="text-zinc-400">Vendor</span>
                        <p className="font-medium">Acme Corp</p>
                      </div>
                      <div>
                        <span className="text-zinc-400">Total</span>
                        <p className="font-medium">$4,250</p>
                      </div>
                    </div>
                    <p className="mt-3 text-[11px] text-emerald-600 font-medium">extracted.json ✓</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* How it works */}
        <section id="product" className="border-t border-black/[0.06] bg-white">
          <div className="max-w-[1280px] mx-auto px-5 sm:px-8 py-14 sm:py-20">
            <div className="max-w-[640px]">
              <p className="inline-flex px-2.5 py-1 rounded-full bg-violet-50 border border-violet-100 text-violet-700 text-[11px] font-semibold tracking-wide">
                HOW IT WORKS
              </p>
              <h2 className="mt-4 text-[32px] sm:text-[44px] font-bold tracking-[-0.03em] leading-[0.95]">
                From messy PDFs to structured data. <span className="text-zinc-400">3 steps.</span>
              </h2>
            </div>
            <div className="mt-12 grid md:grid-cols-3 gap-6">
              {STEPS.map((s) => (
                <div
                  key={s.n}
                  className="group relative rounded-[20px] bg-white border border-black/[0.06] p-7 shadow-[0_1px_2px_rgba(0,0,0,0.04),0_8px_24px_rgba(0,0,0,0.03)] hover:shadow-[0_8px_32px_rgba(0,0,0,0.08)] transition-all"
                >
                  <div className="flex items-start justify-between">
                    <div className={`w-10 h-10 rounded-[12px] ${s.color} text-white grid place-items-center`}>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                        <path d={s.icon} />
                      </svg>
                    </div>
                    <span className="mono text-[12px] font-medium text-zinc-300">{s.n}</span>
                  </div>
                  <h3 className="mt-5 text-[18px] font-semibold tracking-tight">{s.title}</h3>
                  <p className="mt-2 text-[14px] leading-[1.6] text-zinc-500">{s.desc}</p>
                  <div className="mt-6 flex items-center gap-2 text-[12px] font-medium">
                    <span className="w-6 h-6 rounded-full bg-zinc-100 grid place-items-center">→</span>
                    Learn more
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Live demo */}
        <section id="demo" className="bg-[#111] text-white relative overflow-hidden">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(124,58,237,0.15),transparent_60%)]" />
          <div className="relative max-w-[1280px] mx-auto px-3 sm:px-8 py-10 sm:py-16">
            <div className="flex flex-wrap items-end justify-between gap-4 px-2 sm:px-0">
              <div>
                <p className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 border border-white/10 text-[11px] font-medium">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  LIVE DEMO • No signup
                </p>
                <h2 className="mt-4 text-[28px] sm:text-[40px] font-bold tracking-[-0.03em] leading-[0.95]">
                  Try it live
                </h2>
                <p className="mt-2 text-[14px] text-zinc-400 max-w-[420px]">
                  Upload a document or switch type. Watch Extracta extract fields with confidence scores.
                </p>
              </div>
              <div className="w-full lg:w-auto overflow-x-auto scrollbar-none">
                <div className="flex items-center gap-1.5 p-1 rounded-full bg-white/10 border border-white/10 w-max">
                  {DOC_TYPES.map((t) => (
                    <button
                      key={t.id}
                      type="button"
                      onClick={() => switchDocType(t.id)}
                      className={`px-3 sm:px-4 h-8 rounded-full text-[12px] sm:text-[13px] font-medium transition-all whitespace-nowrap ${
                        docType === t.id ? "bg-white text-black shadow" : "text-zinc-400 hover:text-white"
                      }`}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-8 grid grid-cols-1 lg:grid-cols-[0.95fr_1.15fr] gap-4 items-start">
              {/* Document panel */}
              <div className="bg-[#1a1a1a] rounded-[20px] border border-white/10 shadow-[0_20px_60px_rgba(0,0,0,0.4)] overflow-hidden">
                <div className="h-[48px] flex items-center justify-between px-5 border-b border-white/10 bg-white/[0.03]">
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1.5">
                      <span className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                      <span className="w-3 h-3 rounded-full bg-[#ffbd2e]" />
                      <span className="w-3 h-3 rounded-full bg-[#28ca42]" />
                    </div>
                    <span className="mono text-[11px] text-zinc-400">
                      {demo.fileName} • {docType}
                    </span>
                  </div>
                  <div className="hidden sm:flex items-center gap-2 text-[11px] text-zinc-500">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    1 page • OCR ready
                  </div>
                </div>
                <div className="p-3">
                  {!hasUploaded && !scanning ? (
                    <div
                      onDragOver={onDragOver}
                      onDragLeave={onDragLeave}
                      onDrop={onDemoDrop}
                      onClick={() => fileRef.current?.click()}
                      className={`rounded-[14px] border-2 border-dashed p-10 text-center cursor-pointer transition-all ${
                        dragOver
                          ? "border-violet-500 bg-violet-500/10"
                          : "border-white/10 bg-white/[0.02] hover:bg-white/[0.04]"
                      }`}
                    >
                      <input
                        ref={fileRef}
                        type="file"
                        className="hidden"
                        accept=".pdf,.png,.jpg,.jpeg"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) handleDemoFile(file);
                          e.target.value = "";
                        }}
                      />
                      <div className="w-12 h-12 rounded-[12px] bg-white text-black grid place-items-center mx-auto mb-4">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
                          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                          <polyline points="14 2 14 8 20 8" />
                          <line x1="12" y1="18" x2="12" y2="12" />
                          <line x1="9" y1="15" x2="15" y2="15" />
                        </svg>
                      </div>
                      <p className="text-[14px] font-medium">Drop your document here</p>
                      <p className="mt-1 text-[12px] text-zinc-500">PDF, JPG up to 25MB</p>
                    </div>
                  ) : (
                    <div className="rounded-[14px] bg-white text-zinc-900 overflow-hidden min-h-[420px]">
                      {scanning ? (
                        <div className="p-8 flex flex-col items-center justify-center min-h-[420px] text-center">
                          <div className="w-12 h-12 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
                          <p className="mt-4 text-[14px] font-medium">{statusText}</p>
                          <p className="mt-1 text-[12px] text-zinc-500">{Math.round(progress)}%</p>
                          <div className="mt-4 w-full max-w-[200px] h-1 rounded-full bg-zinc-100 overflow-hidden">
                            <div
                              className="h-full bg-violet-500 transition-all"
                              style={{ width: `${progress}%` }}
                            />
                          </div>
                        </div>
                      ) : (
                        <DocumentPreview docType={docType} demo={{ ...demo, fields }} />
                      )}
                    </div>
                  )}
                </div>
                {hasUploaded && !scanning && (
                  <div className="px-4 pb-4 flex items-center justify-between text-[11px] text-zinc-500">
                    <span>Drag new file to replace • Auto-detects type</span>
                    <button
                      type="button"
                      onClick={() => {
                        setHasUploaded(false);
                        setScanning(false);
                      }}
                      className="px-2.5 py-1 rounded-full bg-white text-black text-[11px] font-medium"
                    >
                      Replace
                    </button>
                  </div>
                )}
              </div>

              {/* Output panel */}
              <div className="bg-white rounded-[20px] border border-black/[0.06] shadow-[0_20px_60px_rgba(0,0,0,0.25)] overflow-hidden min-h-[560px] flex flex-col text-zinc-900">
                <div className="h-[52px] flex items-center justify-between px-5 border-b border-black/[0.06] bg-[#fcfcfa]">
                  <div className="flex gap-1 p-1 rounded-full bg-zinc-100">
                    {(
                      [
                        { id: "extracted" as const, label: "Extracted Data" },
                        { id: "json" as const, label: "JSON" },
                        { id: "confidence" as const, label: "Confidence" },
                      ] as const
                    ).map((tab) => (
                      <button
                        key={tab.id}
                        type="button"
                        onClick={() => setOutputTab(tab.id)}
                        className={`px-3.5 h-7 rounded-full text-[13px] font-medium transition-all ${
                          outputTab === tab.id
                            ? "bg-black text-white shadow"
                            : "text-zinc-500 hover:text-zinc-900"
                        }`}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => void copyJson()}
                      className="h-8 px-3 rounded-full bg-white border border-black/10 text-[12px] font-medium hover:bg-zinc-50"
                    >
                      Copy JSON
                    </button>
                    <button
                      type="button"
                      onClick={() => showToast("CSV exported")}
                      className="hidden sm:inline-flex h-8 px-3 rounded-full bg-black text-white text-[12px] font-medium"
                    >
                      Export
                    </button>
                  </div>
                </div>

                <div className="flex-1 overflow-y-auto">
                  {outputTab === "extracted" && (
                    <div className="p-5 space-y-5">
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {Object.entries(fields).map(([key, value]) => {
                          const conf = demo.confidence[key] ?? 96;
                          const editing = editField === key;
                          return (
                            <div
                              key={key}
                              className="group rounded-[14px] border border-black/[0.06] bg-[#fcfcfa] p-4 hover:bg-white hover:border-black/10 transition-colors"
                            >
                              <div className="flex justify-between items-start gap-2">
                                <span className="text-[11px] text-zinc-400 uppercase tracking-wide">
                                  {fieldLabel(key)}
                                </span>
                                <span
                                  className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full ${
                                    conf >= 95
                                      ? "bg-emerald-50 text-emerald-700"
                                      : conf >= 90
                                        ? "bg-amber-50 text-amber-700"
                                        : "bg-zinc-100 text-zinc-600"
                                  }`}
                                >
                                  {conf}%
                                </span>
                              </div>
                              {editing ? (
                                <input
                                  autoFocus
                                  className="mt-1 w-full text-[13px] font-medium bg-white border border-black/10 rounded-[8px] px-2 py-1"
                                  value={value}
                                  onChange={(e) =>
                                    setEditedFields((prev) => ({ ...prev, [key]: e.target.value }))
                                  }
                                  onBlur={() => setEditField(null)}
                                  onKeyDown={(e) => e.key === "Enter" && setEditField(null)}
                                />
                              ) : (
                                <p
                                  className="mt-1 text-[13px] font-medium cursor-pointer"
                                  onClick={() => setEditField(key)}
                                >
                                  {value}
                                </p>
                              )}
                            </div>
                          );
                        })}
                      </div>
                      {demo.lineItems.length > 0 && (
                        <div>
                          <p className="text-[11px] font-bold tracking-wider text-zinc-400 mb-2">LINE ITEMS</p>
                          <div className="rounded-[14px] border border-black/[0.06] overflow-hidden">
                            {demo.lineItems.map((item) => (
                              <div
                                key={item.id}
                                className="flex justify-between px-4 py-2.5 text-[12px] border-b border-black/[0.04] last:border-0"
                              >
                                <span className="font-medium">{item.product}</span>
                                {item.amount !== 0 && (
                                  <span className="mono text-zinc-500">
                                    {item.amount < 0 ? "-" : ""}${Math.abs(item.amount).toLocaleString()}
                                  </span>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {outputTab === "json" && (
                    <div className="p-4">
                      <div className="rounded-[12px] bg-[#0a0a0b] border border-white/10 overflow-hidden">
                        <div className="h-8 flex items-center justify-between px-4 bg-white/[0.04] border-b border-white/10">
                          <span className="mono text-[11px] text-zinc-400">{docType}.json</span>
                          <span className="text-[10px] text-emerald-400">Valid</span>
                        </div>
                        <pre className="p-4 mono text-[11px] leading-[1.7] text-zinc-300 overflow-x-auto">
                          {JSON.stringify(
                            {
                              document_type: docType,
                              ...fields,
                              line_items: demo.lineItems,
                              metadata: { pages: 1, time_ms: 1142 },
                            },
                            null,
                            2,
                          )}
                        </pre>
                      </div>
                      <div className="mt-3 flex gap-2">
                        <button
                          type="button"
                          onClick={() => void copyJson()}
                          className="h-9 px-4 rounded-full bg-black text-white text-[13px] font-medium"
                        >
                          Copy JSON
                        </button>
                        <button
                          type="button"
                          onClick={() => showToast("JSON downloaded")}
                          className="h-9 px-4 rounded-full bg-white border border-black/10 text-[13px] font-medium"
                        >
                          Download .json
                        </button>
                      </div>
                    </div>
                  )}

                  {outputTab === "confidence" && (
                    <div className="p-5 space-y-5">
                      <div className="rounded-[14px] bg-[#fcfcfa] border p-4">
                        <div className="flex justify-between text-[13px] font-semibold">
                          <span>Overall confidence</span>
                          <span>{overallConf.toFixed(1)}%</span>
                        </div>
                        <div className="mt-3 h-2 bg-black/10 rounded-full overflow-hidden flex">
                          <div className="h-full bg-emerald-500" style={{ width: "78%" }} />
                          <div className="h-full bg-amber-400" style={{ width: "18%" }} />
                          <div className="h-full bg-zinc-300" style={{ width: "4%" }} />
                        </div>
                      </div>
                      {Object.entries(demo.confidence).map(([key, val]) => (
                        <div key={key} className="flex items-center gap-3 text-[12px]">
                          <span className="w-[110px] text-zinc-500 capitalize">{fieldLabel(key)}</span>
                          <div className="flex-1 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${val >= 95 ? "bg-emerald-500" : val >= 90 ? "bg-amber-400" : "bg-zinc-400"}`}
                              style={{ width: `${val}%` }}
                            />
                          </div>
                          <span className="w-8 text-right mono text-[11px]">{val}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="px-5 py-3 border-t border-black/[0.06] flex items-center justify-between text-[11px] text-zinc-400">
                  <span>Report issue</span>
                  <span className="text-emerald-600 font-medium">200 OK • 142ms</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features */}
        <section className="border-t border-black/[0.06] bg-[#fafaf9]">
          <div className="max-w-[1280px] mx-auto px-5 sm:px-8 py-14 sm:py-20">
            <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
              <h2 className="text-[28px] sm:text-[36px] font-bold tracking-[-0.03em]">Everything you need.</h2>
              <p className="text-[14px] text-zinc-500 max-w-[320px]">
                No templates to build. No rules to maintain. Our model understands documents like a human, but faster.
              </p>
            </div>
            <div className="mt-10 grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {FEATURES.map((f) => (
                <div
                  key={f.title}
                  className="rounded-[20px] bg-white border border-black/[0.06] p-6 shadow-[0_1px_2px_rgba(0,0,0,0.04)] hover:shadow-[0_8px_24px_rgba(0,0,0,0.06)] transition-all"
                >
                  <div className="w-9 h-9 rounded-[10px] bg-zinc-900 text-white grid place-items-center text-[16px]">
                    {f.icon}
                  </div>
                  <h3 className="mt-4 text-[15px] font-semibold tracking-tight">{f.title}</h3>
                  <p className="mt-1.5 text-[13px] leading-[1.6] text-zinc-500">{f.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Use cases */}
        <section id="usecases" className="border-y border-black/[0.06] bg-white">
          <div className="max-w-[1280px] mx-auto px-5 sm:px-8 py-14 sm:py-20">
            <div className="flex items-center gap-3">
              <span className="w-2 h-2 rounded-full bg-violet-600 animate-pulse" />
              <span className="text-[11px] font-semibold tracking-widest uppercase text-zinc-500">
                Use cases • 100+ types
              </span>
            </div>
            <div className="mt-8 grid md:grid-cols-4 gap-4">
              {USE_CASES.map((u) => (
                <div
                  key={u.k}
                  className="group relative rounded-[20px] overflow-hidden border border-black/[0.06] p-[1px]"
                >
                  <div
                    className={`absolute inset-0 bg-gradient-to-br ${u.color} opacity-[0.08] group-hover:opacity-[0.12] transition-opacity`}
                  />
                  <div className="relative rounded-[19px] bg-white p-6 h-full">
                    <div
                      className={`w-10 h-10 rounded-[12px] bg-gradient-to-br ${u.color} text-white grid place-items-center font-bold text-[14px]`}
                    >
                      {u.k[0]}
                    </div>
                    <h4 className="mt-4 font-semibold tracking-tight">{u.k}</h4>
                    <p className="mt-1.5 text-[13px] leading-[1.5] text-zinc-500">{u.desc}</p>
                    <div className="mt-6 inline-flex px-2.5 py-1 rounded-full bg-zinc-900 text-white text-[11px] font-medium mono">
                      {u.metric}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Pricing */}
        <section id="pricing" className="max-w-[1280px] mx-auto px-5 sm:px-8 py-16 sm:py-24">
          <div className="text-center max-w-[640px] mx-auto">
            <h2 className="text-[32px] sm:text-[44px] font-bold tracking-[-0.03em] leading-[0.95]">
              Simple pricing. <span className="text-zinc-400">Start free.</span>
            </h2>
            <p className="mt-3 text-[15px] text-zinc-500">
              Pay for what you extract. No hidden fees. Cancel anytime.
            </p>
            <div className="mt-6 inline-flex p-1 rounded-full bg-zinc-100 border border-black/5">
              <button
                type="button"
                onClick={() => setYearly(false)}
                className={`px-4 h-8 rounded-full text-[13px] font-medium transition-all ${
                  !yearly ? "bg-white shadow text-black" : "text-zinc-500"
                }`}
              >
                Monthly
              </button>
              <button
                type="button"
                onClick={() => setYearly(true)}
                className={`px-4 h-8 rounded-full text-[13px] font-medium transition-all flex items-center gap-1.5 ${
                  yearly ? "bg-white shadow text-black" : "text-zinc-500"
                }`}
              >
                Yearly{" "}
                <span className="px-1.5 py-0.5 rounded-full bg-emerald-500 text-white text-[10px]">-20%</span>
              </button>
            </div>
          </div>
          <div className="mt-12 grid lg:grid-cols-3 gap-5 max-w-[1040px] mx-auto">
            {PRICING.map((plan) => {
              const monthlyPrice =
                plan.price === null ? null : yearly ? (plan.name === "Starter" ? 23 : 79) : plan.price;
              const displayPrice = plan.price === null ? "Custom" : `$${monthlyPrice}`;
              return (
                <div
                  key={plan.name}
                  className={`relative rounded-[24px] border p-7 flex flex-col ${
                    plan.popular
                      ? "bg-black text-white border-black shadow-[0_20px_60px_rgba(0,0,0,0.2)] scale-[1.02]"
                      : "bg-white border-black/[0.06] shadow-[0_1px_2px_rgba(0,0,0,0.04)]"
                  }`}
                >
                  {plan.popular && (
                    <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-violet-600 text-white text-[11px] font-semibold tracking-wide">
                      MOST POPULAR
                    </div>
                  )}
                  <div className="flex justify-between items-start">
                    <h3 className="text-[18px] font-semibold tracking-tight">{plan.name}</h3>
                    <span
                      className={`px-2.5 py-1 rounded-full text-[11px] font-medium mono ${
                        plan.popular ? "bg-white/10 text-white" : "bg-zinc-100 text-zinc-600"
                      }`}
                    >
                      {plan.pages}
                    </span>
                  </div>
                  <p className={`mt-2 text-[13px] leading-[1.5] ${plan.popular ? "text-zinc-400" : "text-zinc-500"}`}>
                    {plan.desc}
                  </p>
                  <div className="mt-5 flex items-baseline gap-1">
                    <span className="text-[40px] font-bold tracking-tight">{displayPrice}</span>
                    {plan.price !== null && (
                      <span className={`text-[14px] ${plan.popular ? "text-zinc-500" : "text-zinc-400"}`}>/mo</span>
                    )}
                  </div>
                  <ul className="mt-6 space-y-2.5 flex-1">
                    {plan.features.map((f) => (
                      <li key={f} className="text-[13px] flex items-start gap-2">
                        <span className={plan.popular ? "text-emerald-400" : "text-emerald-500"}>✓</span>
                        <span className={plan.popular ? "text-zinc-300" : "text-zinc-600"}>{f}</span>
                      </li>
                    ))}
                  </ul>
                  <button
                    type="button"
                    onClick={onStartExtracting}
                    className={`mt-7 h-11 rounded-full text-[14px] font-semibold transition-colors ${
                      plan.popular
                        ? "bg-white text-black hover:bg-zinc-100"
                        : "bg-black text-white hover:bg-zinc-800"
                    }`}
                  >
                    {plan.cta}
                  </button>
                </div>
              );
            })}
          </div>
        </section>

        {/* Dark CTA footer */}
        <section className="bg-[#111] text-white relative overflow-hidden">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom,_rgba(124,58,237,0.12),transparent_60%)]" />
          <div className="relative max-w-[1280px] mx-auto px-5 sm:px-8 py-16 sm:py-24">
            <div className="grid lg:grid-cols-2 gap-12 items-center">
              <div>
                <h2 className="text-[32px] sm:text-[44px] font-bold tracking-[-0.03em] leading-[0.95]">
                  Extract your first document today.
                </h2>
                <p className="mt-4 text-[15px] text-zinc-400 max-w-[420px]">
                  No credit card. No template setup. Just upload and get structured JSON in seconds.
                </p>
                <div className="mt-7 flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={onStartExtracting}
                    className="h-11 px-6 rounded-full bg-white text-black font-semibold text-[14px]"
                  >
                    Start Extracting free
                  </button>
                  <button
                    type="button"
                    onClick={() => setFounderOpen(true)}
                    className="h-11 px-6 rounded-full bg-white/10 border border-white/10 text-white font-medium text-[14px]"
                  >
                    Talk to founder
                  </button>
                </div>
                <div className="mt-6 flex items-center gap-2 text-[11px] text-zinc-500">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  SOC2 certified • GDPR • Zero retention available
                </div>
              </div>
              <div className="lg:justify-self-end w-full max-w-[360px] rounded-[16px] bg-white/[0.06] border border-white/10 p-4 backdrop-blur">
                <div className="flex items-center justify-between text-[11px] mono text-zinc-400">
                  <span>api.extracta.dev</span>
                  <span className="px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/20">
                    200 OK • 142ms
                  </span>
                </div>
                <pre className="mt-3 p-3 rounded-[10px] bg-black/50 mono text-[11px] leading-[1.6] text-zinc-300 overflow-x-auto whitespace-pre-wrap">
                  {CURL_SNIPPET}
                </pre>
              </div>
            </div>
          </div>
          <div className="relative border-t border-white/10 px-8 sm:px-12 h-[56px] flex items-center justify-between text-[12px] text-zinc-500">
            <div className="flex items-center gap-6">
              <span className="flex items-center gap-2 font-semibold text-white">
                <span className="w-6 h-6 rounded-[7px] bg-white text-black grid place-items-center text-[12px] font-bold">
                  E
                </span>
                Extracta
              </span>
              <span className="hidden sm:inline">© 2025 Extracta Inc.</span>
            </div>
            <div className="flex gap-5">
              <button type="button" className="hover:text-white">
                Privacy
              </button>
              <button type="button" className="hover:text-white">
                Terms
              </button>
              <button type="button" className="hover:text-white">
                Security
              </button>
              <button type="button" className="hover:text-white">
                Changelog
              </button>
            </div>
          </div>
        </section>
      </main>

      <FounderContactModal
        open={founderOpen}
        onClose={() => setFounderOpen(false)}
        onCopied={showToast}
      />

      {toast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[60] px-4 py-2.5 rounded-full bg-zinc-900 text-white text-[13px] font-medium shadow-[0_8px_24px_rgba(0,0,0,0.3)] flex items-center gap-2">
          <span className="w-5 h-5 rounded-full bg-white/15 grid place-items-center">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
              <path d="M5 12l5 5l10-10" />
            </svg>
          </span>
          {toast}
        </div>
      )}
    </div>
  );
}
