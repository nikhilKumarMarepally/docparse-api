import type { DocDemo, DocType } from "./demoData";

type Props = {
  docType: DocType;
  demo: DocDemo;
};

export default function DocumentPreview({ docType, demo }: Props) {
  const v = demo.fields;

  if (docType === "invoice") {
    return (
      <div className="p-6 text-[11px] leading-[1.5]">
        <div className="flex justify-between">
          <div>
            <div className="text-[18px] font-bold">ACME CORP</div>
            <div className="text-zinc-500 mono text-[9px]">INDUSTRIES</div>
            <div className="mt-3 text-zinc-600">
              548 Market St
              <br />
              SF, CA 94104
            </div>
          </div>
          <div className="text-right">
            <div className="text-[18px] font-semibold">INVOICE</div>
            <div className="mt-1 inline-flex px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 text-[9px] border">
              {v.invoiceNumber}
            </div>
          </div>
        </div>
        <div className="mt-6 grid grid-cols-2 gap-4">
          <div>
            <div className="text-[9px] uppercase text-zinc-400">Bill To</div>
            <div className="font-medium">Linear Inc.</div>
          </div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-zinc-400">Date</span>
              <span>{v.date}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Due</span>
              <span>{v.dueDate}</span>
            </div>
          </div>
        </div>
        <div className="mt-5 border-t pt-3">
          <div className="grid grid-cols-[1fr_40px_60px_60px] gap-2 text-[9px] uppercase text-zinc-400">
            <span>Desc</span>
            <span>Qty</span>
            <span className="text-right">Rate</span>
            <span className="text-right">Amt</span>
          </div>
          {demo.lineItems.map((h) => (
            <div key={h.id} className="grid grid-cols-[1fr_40px_60px_60px] gap-2 py-2 border-t border-zinc-50">
              <span className="font-medium truncate">{h.product}</span>
              <span className="text-center">{h.qty}</span>
              <span className="text-right mono">${h.rate}</span>
              <span className="text-right mono font-medium">${h.amount}</span>
            </div>
          ))}
          <div className="mt-4 flex justify-end border-t border-black pt-2 font-bold">
            <span>Total</span>
            <span className="ml-8">{v.total}</span>
          </div>
        </div>
      </div>
    );
  }

  if (docType === "receipt") {
    return (
      <div className="p-6 text-center">
        <div className="inline-flex w-10 h-10 rounded-full bg-black text-white grid place-items-center font-bold">B</div>
        <div className="mt-2 font-bold tracking-widest text-[12px]">BLUE BOTTLE COFFEE</div>
        <div className="text-[10px] text-zinc-500">Mint Plaza • San Francisco</div>
        <div className="mt-4 border-t border-dashed pt-4 text-left text-[11px] space-y-2">
          {demo.lineItems.map((h) => (
            <div key={h.id} className="flex justify-between">
              <span>{h.product}</span>
              <span className="mono">${h.amount}</span>
            </div>
          ))}
          <div className="flex justify-between font-bold pt-2 border-t border-dashed">
            <span>Total</span>
            <span>{v.total}</span>
          </div>
        </div>
        <div className="mt-4 text-[10px] text-zinc-400">
          {v.date} • {v.payment}
        </div>
      </div>
    );
  }

  if (docType === "passport") {
    return (
      <div className="bg-[#0f172a] text-white p-6">
        <div className="flex justify-between items-start">
          <div>
            <div className="text-[10px] tracking-[0.2em] opacity-60">UNITED STATES OF AMERICA</div>
            <div className="mt-1 text-[14px] font-bold">PASSPORT</div>
          </div>
          <div className="w-12 h-12 rounded-[8px] bg-white/10 grid place-items-center text-[20px]">◐</div>
        </div>
        <div className="mt-6 flex gap-4">
          <div className="w-20 h-24 rounded-[8px] bg-white/15" />
          <div className="flex-1 space-y-2 text-[11px]">
            <div>
              <span className="opacity-50 text-[9px] uppercase">Name</span>
              <div className="font-semibold">{v.fullName}</div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <span className="opacity-50 text-[9px] uppercase">Passport No</span>
                <div className="mono font-medium">{v.passportNo}</div>
              </div>
              <div>
                <span className="opacity-50 text-[9px] uppercase">Nationality</span>
                <div>{v.nationality}</div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <span className="opacity-50 text-[9px] uppercase">DOB</span>
                <div>{v.dob}</div>
              </div>
              <div>
                <span className="opacity-50 text-[9px] uppercase">Expiry</span>
                <div>{v.expiry}</div>
              </div>
            </div>
          </div>
        </div>
        <div className="mt-6 pt-3 border-t border-white/10 mono text-[9px] opacity-40 leading-[1.4]">
          P&lt;USA{v.fullName?.replace(/ /g, "&lt;")}&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;
          <br />
          N4822917&lt;2USA920414&lt;...
        </div>
      </div>
    );
  }

  if (docType === "contract") {
    return (
      <div className="p-6 text-[11px] leading-[1.7]">
        <div className="text-[16px] font-bold tracking-tight">Master Service Agreement</div>
        <div className="text-[10px] text-zinc-500 mono">MSA • {v.effectiveDate}</div>
        <div className="mt-4 space-y-3 text-zinc-700">
          <p>
            <b>Parties:</b> {v.parties}
          </p>
          <p>
            <b>Term:</b> {v.term}. This agreement commences on {v.effectiveDate} and shall continue for the initial term...
          </p>
          <p>
            <b>Fees:</b> Client shall pay {v.value} as set forth in Exhibit A.
          </p>
          <div className="mt-4 p-3 rounded-[10px] bg-amber-50 border border-amber-200 text-amber-900">
            <b className="text-[10px] uppercase tracking-widest">Extracted Clauses</b>
            <div className="mt-1 space-y-1">
              {demo.lineItems.map((h) => (
                <div key={h.id} className="flex justify-between">
                  <span>• {h.product}</span>
                  <span className="mono text-[10px] opacity-60">{h.sku}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex justify-between">
        <div className="font-bold">MERCURY • Business Checking ••8821</div>
        <div className="text-[10px] px-2 py-1 rounded-full bg-zinc-100 mono">{v.period}</div>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-[11px]">
        <div className="rounded-[10px] bg-zinc-50 p-3 border">
          <div className="text-zinc-400 text-[10px] uppercase">Opening</div>
          <div className="font-semibold">{v.opening}</div>
        </div>
        <div className="rounded-[10px] bg-zinc-900 text-white p-3">
          <div className="text-zinc-400 text-[10px] uppercase">Closing</div>
          <div className="font-semibold">{v.closing}</div>
        </div>
      </div>
      <div className="mt-4 text-[11px]">
        <div className="text-[10px] uppercase tracking-widest text-zinc-400 mb-2">Transactions</div>
        {demo.lineItems.map((h) => (
          <div key={h.id} className="flex justify-between py-2 border-b border-zinc-100">
            <div>
              <div className="font-medium">{h.product}</div>
              <div className="text-[10px] text-zinc-500 mono">{h.sku}</div>
            </div>
            <div className={`mono font-medium ${h.amount < 0 ? "text-red-600" : ""}`}>
              {h.amount < 0 ? "-" : "+"}${Math.abs(h.amount).toLocaleString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
