import type { AuthUser } from "../auth";
import { outOfCreditsMessage } from "../creditsCopy";

type Props = {
  dragOver: boolean;
  busy: boolean;
  ready?: boolean;
  user?: AuthUser | null;
  authRequired: boolean;
  creditsPerPage?: number;
  initialCredits?: number;
  onBrowse: () => void;
  onLogin: () => void;
  onSignOut?: () => void;
  onHome: () => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: () => void;
  onDrop: (e: React.DragEvent) => void;
};

const FORMATS = [
  { ext: "PDF", desc: "Invoices, contracts, forms" },
  { ext: "PNG / JPG", desc: "Photos and scans" },
  { ext: "TIFF / WEBP", desc: "High-res and web images" },
];

export default function UploadPage({
  dragOver,
  busy,
  ready,
  user,
  authRequired,
  creditsPerPage = 2,
  initialCredits = 20,
  onBrowse,
  onLogin,
  onSignOut,
  onHome,
  onDragOver,
  onDragLeave,
  onDrop,
}: Props) {
  const needsLogin = authRequired && !user;
  const credits = user?.credits;
  const outOfCredits =
    !needsLogin && typeof credits === "number" && credits < creditsPerPage;

  return (
    <div className="min-h-screen bg-[#fafaf9] text-zinc-900 flex flex-col">
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-[#fafaf9]/80 border-b border-black/[0.06]">
        <div className="max-w-[1280px] mx-auto px-5 sm:px-8 h-[68px] flex items-center justify-between">
          <button type="button" onClick={onHome} className="flex items-center gap-2.5">
            <div className="w-[28px] h-[28px] rounded-[9px] bg-black text-white grid place-items-center font-bold text-[15px]">
              E
            </div>
            <span className="font-semibold tracking-[-0.02em] text-[18px]">Extracta</span>
          </button>
          <div className="flex items-center gap-2.5">
            {ready === false && (
              <span className="hidden sm:inline text-[12px] text-amber-600 font-medium">Offline</span>
            )}
            {user && onSignOut ? (
              <>
                {typeof user.credits === "number" && (
                  <span className="hidden sm:inline text-[13px] text-zinc-600 font-medium">
                    {user.credits} credit{user.credits === 1 ? "" : "s"}
                  </span>
                )}
                <span className="hidden sm:inline text-[13px] text-zinc-500">{user.name || user.email}</span>
                <button
                  type="button"
                  onClick={onSignOut}
                  className="h-9 px-4 rounded-full text-[14px] font-medium hover:bg-black/5 transition-colors"
                >
                  Sign out
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={onLogin}
                className="h-9 px-4 rounded-full text-[14px] font-medium hover:bg-black/5 transition-colors"
              >
                Login
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="flex-1 flex items-center justify-center px-5 py-12 sm:py-16">
        <div className="w-full max-w-[640px]">
          <div className="text-center mb-8">
            <p className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-50 border border-violet-100 text-violet-700 text-[11px] font-semibold tracking-wide">
              UPLOAD
            </p>
            <h1 className="mt-4 text-[32px] sm:text-[40px] font-bold tracking-[-0.03em] leading-[0.95]">
              Upload your document
            </h1>
            <p className="mt-3 text-[15px] text-zinc-500 max-w-[420px] mx-auto">
              Drop a file or browse from your computer. We&apos;ll extract fields, tables, and line items automatically.
            </p>
          </div>

          {needsLogin && (
            <div className="mb-6 rounded-[14px] border border-amber-200 bg-amber-50 px-4 py-3 text-[14px] text-amber-900 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <span>Sign up or sign in with email and password to extract documents.</span>
              <button
                type="button"
                onClick={onLogin}
                className="h-9 px-4 rounded-full bg-black text-white text-[13px] font-semibold shrink-0"
              >
                Sign up / Sign in
              </button>
            </div>
          )}

          {outOfCredits && (
            <div className="mb-6 rounded-[14px] border border-red-200 bg-red-50 px-4 py-3 text-[14px] text-red-900">
              {outOfCreditsMessage(creditsPerPage, initialCredits)}
            </div>
          )}

          <div
            onDragOver={outOfCredits ? undefined : onDragOver}
            onDragLeave={outOfCredits ? undefined : onDragLeave}
            onDrop={outOfCredits ? undefined : onDrop}
            onClick={() => {
              if (busy) return;
              if (needsLogin) {
                onLogin();
                return;
              }
              if (outOfCredits) return;
              onBrowse();
            }}
            onKeyDown={(e) => {
              if (e.key !== "Enter" || busy) return;
              if (needsLogin) {
                onLogin();
                return;
              }
              if (!outOfCredits) onBrowse();
            }}
            role="button"
            tabIndex={0}
            className={`rounded-[24px] border-2 border-dashed p-10 sm:p-14 text-center transition-all cursor-pointer ${
              outOfCredits
                ? "border-zinc-200 bg-zinc-50 cursor-not-allowed opacity-70"
                : needsLogin
                  ? "border-amber-200 bg-amber-50/40 hover:border-amber-300"
                : dragOver
                  ? "border-violet-500 bg-violet-50/80 scale-[1.01]"
                  : busy
                    ? "border-zinc-200 bg-white cursor-wait"
                    : "border-black/10 bg-white hover:border-black/20 hover:shadow-[0_8px_32px_rgba(0,0,0,0.06)]"
            }`}
          >
            {busy ? (
              <>
                <div className="w-14 h-14 rounded-full border-2 border-violet-500 border-t-transparent animate-spin mx-auto" />
                <p className="mt-5 text-[16px] font-semibold">Uploading your document…</p>
                <p className="mt-1 text-[13px] text-zinc-500">This usually takes a few seconds</p>
              </>
            ) : (
              <>
                <div className="w-14 h-14 rounded-[16px] bg-black text-white grid place-items-center mx-auto">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
                    <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
                <p className="mt-5 text-[18px] font-semibold">
                  {dragOver ? "Drop to upload" : "Drop your document here"}
                </p>
                <p className="mt-1 text-[14px] text-zinc-500">or click to browse files</p>
                <button
                  type="button"
                  disabled={needsLogin || outOfCredits}
                  onClick={(e) => {
                    e.stopPropagation();
                    onBrowse();
                  }}
                  className="mt-6 h-11 px-6 rounded-full bg-black text-white text-[14px] font-semibold hover:bg-zinc-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Choose file
                </button>
                <p className="mt-4 text-[12px] text-zinc-400">PDF, PNG, JPG, TIFF, WEBP — up to 20 MB</p>
              </>
            )}
          </div>

          <div className="mt-8 grid sm:grid-cols-3 gap-3">
            {FORMATS.map((f) => (
              <div
                key={f.ext}
                className="rounded-[14px] border border-black/[0.06] bg-white px-4 py-3 text-center"
              >
                <p className="text-[13px] font-semibold">{f.ext}</p>
                <p className="mt-0.5 text-[11px] text-zinc-500">{f.desc}</p>
              </div>
            ))}
          </div>

          <button
            type="button"
            onClick={onHome}
            className="mt-8 mx-auto flex items-center gap-2 text-[14px] text-zinc-500 hover:text-zinc-900 transition-colors"
          >
            <span>←</span> Back to home
          </button>
        </div>
      </main>
    </div>
  );
}
