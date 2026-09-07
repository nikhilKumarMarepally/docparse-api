import { FOUNDER_CONTACT } from "./founderContact";

type Props = {
  open: boolean;
  onClose: () => void;
  onCopied: (message: string) => void;
};

export default function FounderContactModal({ open, onClose, onCopied }: Props) {
  if (!open) return null;

  const linkedinLabel = FOUNDER_CONTACT.linkedin.replace(/^https?:\/\/(www\.)?linkedin\.com\//, "");

  const copyEmail = async () => {
    await navigator.clipboard.writeText(FOUNDER_CONTACT.email);
    onCopied("Email copied");
  };

  return (
    <div
      className="fixed inset-0 z-[80] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="w-full max-w-[400px] rounded-[20px] bg-white border border-black/[0.08] shadow-[0_24px_64px_rgba(0,0,0,0.2)] p-6"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-labelledby="founder-contact-title"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-[11px] font-semibold tracking-widest uppercase text-violet-600">Founder</p>
            <h2 id="founder-contact-title" className="mt-1 text-[20px] font-bold tracking-tight">
              Talk to {FOUNDER_CONTACT.name.split(" ")[0]}
            </h2>
            <p className="mt-1 text-[14px] text-zinc-500">Reach out on LinkedIn or email.</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 rounded-full border border-black/10 text-zinc-500 hover:bg-zinc-50"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="mt-6 space-y-3">
          <a
            href={FOUNDER_CONTACT.linkedin}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-[14px] border border-black/[0.08] bg-[#fafaf9] px-4 py-3 hover:border-black/15 hover:bg-white transition-colors"
          >
            <span className="w-9 h-9 rounded-[10px] bg-[#0A66C2] text-white grid place-items-center text-[13px] font-bold">
              in
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[12px] text-zinc-400">LinkedIn</p>
              <p className="text-[14px] font-medium text-zinc-900 truncate">linkedin.com/{linkedinLabel}</p>
            </div>
            <span className="text-zinc-400">↗</span>
          </a>

          <div className="flex items-center gap-3 rounded-[14px] border border-black/[0.08] bg-[#fafaf9] px-4 py-3">
            <span className="w-9 h-9 rounded-[10px] bg-zinc-900 text-white grid place-items-center text-[16px]">
              @
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[12px] text-zinc-400">Email</p>
              <a
                href={`mailto:${FOUNDER_CONTACT.email}`}
                className="text-[14px] font-medium text-zinc-900 hover:underline break-all"
              >
                {FOUNDER_CONTACT.email}
              </a>
            </div>
            <button
              type="button"
              onClick={() => void copyEmail()}
              className="h-8 px-3 rounded-full border border-black/10 text-[12px] font-medium hover:bg-white shrink-0"
            >
              Copy
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
