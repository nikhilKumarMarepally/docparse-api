import { useCallback, useEffect, useRef, useState } from "react";
import LandingPage from "./landing/LandingPage";
import UploadPage from "./pages/UploadPage";
import ProcessingBar from "./components/ProcessingBar";
import ResultsView from "./components/ResultsView";
import LoginModal from "./components/LoginModal";
import { useAuth } from "./auth";
import { outOfCreditsMessage, estimateExtractionCost } from "./creditsCopy";
import { countUploadPages } from "./pageCount";
import {
  JobResponse,
  pollJob,
  stepLabel,
  uploadFile,
} from "./api";
import { fetchApi } from "./apiBase";

const ACCEPT = ".pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp";

function friendlyError(raw: string): string {
  const msg = raw.toLowerCase();
  if (msg.includes("429") || msg.includes("rate limit") || msg.includes("too many requests")) {
    return "We're reading documents a bit too quickly right now. Please wait a few seconds and try again.";
  }
  if (msg.includes("403") && msg.includes("vision")) {
    return "Google Vision API rejected the request (403). Enable Cloud Vision API and billing on your personal GCP project, or update GOOGLE_CLOUD_API_KEY in .env.local.";
  }
  if (msg.includes("billing") || msg.includes("403")) {
    return "We couldn't read this document — your Google Cloud API key may need billing enabled or the Vision API turned on.";
  }
  if (msg.includes("api key") || msg.includes("credentials") || msg.includes("not set")) {
    return "Document parsing is temporarily unavailable. Please try again later.";
  }
  if (msg.includes("exceeds") || msg.includes("20 mb")) {
    return "This file is too large. Please use a file under 20 MB.";
  }
  if (msg.includes("unsupported")) {
    return "This file type isn't supported. Try PDF, PNG, or JPG.";
  }
  if (msg.includes("sign in") || msg.includes("401")) {
    return "Please sign in with email or Google before uploading.";
  }
  if (msg.includes("ran out of credits") || msg.includes("credits")) {
    return raw;
  }
  if (msg.includes("not found")) {
    return "Something went wrong. Please upload your document again.";
  }
  return "Something went wrong while reading your document. Please try again.";
}

type Health = {
  status: string;
};

type View = "home" | "upload" | "processing" | "results";

function viewFromPath(path: string): View {
  return path === "/upload" ? "upload" : "home";
}

function pathForView(view: View): string {
  return view === "upload" ? "/upload" : "/";
}

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const { user, token, authRequired, signOut, setUserCredits, creditsPerPage, initialCredits } = useAuth();
  const [job, setJob] = useState<JobResponse | null>(null);
  const [view, setView] = useState<View>(() => viewFromPath(window.location.pathname));
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [loginOpen, setLoginOpen] = useState(false);
  const [pendingUpload, setPendingUpload] = useState(false);
  const pendingFileRef = useRef<File | null>(null);
  const [uploadBusy, setUploadBusy] = useState(false);

  useEffect(() => {
    fetchApi("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "error" }));
  }, []);

  useEffect(() => {
    const onPopState = () => {
      const next = viewFromPath(window.location.pathname);
      setView((current) => {
        if (current === "processing" || current === "results") return current;
        return next;
      });
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((next: View) => {
    setView(next);
    const path = pathForView(next);
    if (window.location.pathname !== path) {
      window.history.pushState(null, "", path);
    }
  }, []);

  useEffect(() => {
    if (!job?.job_id) return;
    if (job.status === "completed" || job.status === "failed") return;

    let cancelled = false;

    const tick = async () => {
      try {
        const next = await pollJob(job.job_id);
        if (cancelled) return;
        setJob(next);
        if (next.status === "completed") {
          setView("results");
        }
        if (next.status === "failed") {
          setError(next.error || "Processing failed");
        }
      } catch (e) {
        if (cancelled) return;
        setError(String(e));
        setView("upload");
      }
    };

    void tick();
    const id = window.setInterval(() => void tick(), 1200);

    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [job?.job_id, job?.status]);

  const goToUpload = useCallback(() => {
    setError(null);
    if (authRequired && !token) {
      setPendingUpload(true);
      setLoginOpen(true);
      return;
    }
    navigate("upload");
  }, [authRequired, token, navigate]);

  const goHome = useCallback(() => {
    setError(null);
    navigate("home");
  }, [navigate]);

  const openFilePicker = useCallback(() => {
    if (authRequired && !token) {
      setPendingUpload(true);
      setLoginOpen(true);
      return;
    }
    if (
      authRequired &&
      user &&
      typeof user.credits === "number" &&
      user.credits < creditsPerPage
    ) {
      setError(outOfCreditsMessage(creditsPerPage, initialCredits));
      return;
    }
    inputRef.current?.click();
  }, [authRequired, token, user, creditsPerPage, initialCredits]);

  const handleFile = useCallback(
    async (file: File) => {
      if (authRequired && !token) {
        pendingFileRef.current = file;
        setPendingUpload(true);
        setLoginOpen(true);
        navigate("upload");
        return;
      }
      const pageCount = await countUploadPages(file);
      const cost = estimateExtractionCost(pageCount, creditsPerPage);
      if (
        authRequired &&
        user &&
        typeof user.credits === "number" &&
        user.credits < cost
      ) {
        const pageLabel = pageCount === 1 ? "page" : "pages";
        setError(
          `Not enough credits. This document has ${pageCount} ${pageLabel} and costs ${cost} credits (${creditsPerPage} per page). Your balance is ${user.credits}.`,
        );
        navigate("upload");
        return;
      }
      setError(null);
      setUploadBusy(true);
      setView("processing");
      setJob(null);
      try {
        const { jobId, creditsRemaining } = await uploadFile(file, token);
        if (creditsRemaining !== undefined) {
          setUserCredits(creditsRemaining);
        }
        const initial = await pollJob(jobId);
        setJob(initial);
        setUploadBusy(false);
        if (initial.status === "completed") {
          setView("results");
        } else if (initial.status === "failed") {
          setError(initial.error || "Processing failed");
        }
      } catch (e) {
        setUploadBusy(false);
        setError(e instanceof Error ? e.message : String(e));
        setView("upload");
      }
    },
    [authRequired, token, navigate, setUserCredits, user, creditsPerPage, initialCredits],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) void handleFile(file);
    },
    [handleFile],
  );

  const reset = () => {
    setJob(null);
    setError(null);
    setUploadBusy(false);
    navigate("upload");
  };

  const onLoginSuccess = () => {
    const pendingFile = pendingFileRef.current;
    pendingFileRef.current = null;
    if (pendingUpload) {
      setPendingUpload(false);
      if (pendingFile) {
        void handleFile(pendingFile);
        return;
      }
      navigate("upload");
    }
  };

  const downloadJson = () => {
    if (!job) return;
    const blob = new Blob([JSON.stringify(job, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${job.job_id || "result"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyJson = async () => {
    if (!job) return;
    await navigator.clipboard.writeText(JSON.stringify(job, null, 2));
  };

  const processingStep = job?.step ?? (view === "processing" ? "upload" : undefined);
  const processingLabel = stepLabel(processingStep);
  const ready = health?.status === "ok";

  return (
    <div className="app-shell">
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void handleFile(file);
          e.target.value = "";
        }}
      />

      <LoginModal
        open={loginOpen}
        onClose={() => {
          setLoginOpen(false);
          setPendingUpload(false);
          pendingFileRef.current = null;
        }}
        onSuccess={onLoginSuccess}
      />

      {error && (view === "home" || view === "upload") && (
        <div className="fixed top-20 left-1/2 -translate-x-1/2 z-[70] max-w-md w-[calc(100%-2rem)] px-4 py-3 rounded-[12px] bg-red-50 border border-red-200 text-red-800 text-[14px] shadow-lg">
          {friendlyError(error)}
        </div>
      )}
      {job?.error && view === "upload" && (
        <div className="fixed top-20 left-1/2 -translate-x-1/2 z-[70] max-w-md w-[calc(100%-2rem)] px-4 py-3 rounded-[12px] bg-red-50 border border-red-200 text-red-800 text-[14px] shadow-lg">
          {friendlyError(job.error)}
        </div>
      )}

      {view === "home" && (
        <LandingPage
          onStartExtracting={goToUpload}
          onUploadFile={(file) => void handleFile(file)}
          onLogin={() => setLoginOpen(true)}
          onSignOut={signOut}
          user={user}
          onFileDrop={onDrop}
          dragOver={dragOver}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          ready={health ? ready : undefined}
        />
      )}

      {view === "upload" && (
        <UploadPage
          dragOver={dragOver}
          busy={uploadBusy}
          ready={health ? ready : undefined}
          user={user}
          authRequired={authRequired}
          creditsPerPage={creditsPerPage}
          initialCredits={initialCredits}
          onBrowse={openFilePicker}
          onLogin={() => {
            setPendingUpload(true);
            setLoginOpen(true);
          }}
          onSignOut={signOut}
          onHome={goHome}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        />
      )}

      {view === "processing" && (
        <div className="min-h-screen bg-[#fafaf9] text-zinc-900">
          <header className="sticky top-0 z-40 backdrop-blur-xl bg-[#fafaf9]/80 border-b border-black/[0.06]">
            <div className="max-w-[1280px] mx-auto px-5 sm:px-8 h-[68px] flex items-center justify-between">
              <button type="button" onClick={reset} className="flex items-center gap-2.5">
                <div className="w-[28px] h-[28px] rounded-[9px] bg-black text-white grid place-items-center font-bold text-[15px]">
                  E
                </div>
                <span className="font-semibold tracking-[-0.02em] text-[18px]">Extracta</span>
              </button>
              <span className="text-[13px] text-zinc-500">
                {job?.status === "failed" ? "Processing failed" : processingLabel}
              </span>
            </div>
          </header>
          {job?.status === "failed" ? (
            <div className="max-w-[640px] mx-auto px-5 py-16">
              <div className="rounded-[16px] border border-red-200 bg-red-50 p-6 text-left">
                <p className="text-[16px] font-semibold text-red-900">Couldn&apos;t process this document</p>
                <p className="mt-2 text-[14px] text-red-800 leading-relaxed">
                  {friendlyError(error || job.error || "Unknown error")}
                </p>
                <button
                  type="button"
                  onClick={reset}
                  className="mt-5 h-10 px-4 rounded-[10px] bg-black text-white text-[14px] font-semibold"
                >
                  Try another file
                </button>
              </div>
            </div>
          ) : (
            <>
              <ProcessingBar step={processingStep} light />
              <div className="max-w-[640px] mx-auto px-5 py-20 text-center">
                <p className="text-[15px] text-zinc-500">This usually takes a few seconds…</p>
                {job?.filename && (
                  <p className="mt-2 text-[13px] text-zinc-400">{job.filename}</p>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {view === "results" && job?.status === "completed" && job.pages && (
        <div className="results-light-wrap">
          <ResultsView
            job={job}
            onNewDoc={reset}
            onCopyJson={copyJson}
            onDownloadJson={downloadJson}
          />
        </div>
      )}
    </div>
  );
}
