export type DocType = "invoice" | "receipt" | "passport" | "contract" | "bank";

export type LineItem = {
  id: number;
  product: string;
  sku: string;
  qty: number;
  rate: number;
  amount: number;
};

export type DocDemo = {
  fileName: string;
  displayName: string;
  fields: Record<string, string>;
  lineItems: LineItem[];
  confidence: Record<string, number>;
};

export const DOC_TYPES: { id: DocType; label: string; icon: string }[] = [
  { id: "invoice", label: "Invoice", icon: "🧾" },
  { id: "receipt", label: "Receipt", icon: "🧮" },
  { id: "passport", label: "Passport / ID", icon: "🛂" },
  { id: "contract", label: "Contract", icon: "📄" },
  { id: "bank", label: "Bank Statement", icon: "🏦" },
];

export const DEMO_DATA: Record<DocType, DocDemo> = {
  invoice: {
    fileName: "Invoice_2024_Acme.pdf",
    displayName: "ACME CORP",
    fields: {
      invoiceNumber: "INV-2024-1842",
      vendor: "Acme Corp Industries",
      date: "Mar 15, 2024",
      total: "$4,250.00",
      dueDate: "Apr 14, 2024",
      poNumber: "PO-883291",
    },
    lineItems: [
      { id: 1, product: "Design System Audit", sku: "SRV-001", qty: 1, rate: 2500, amount: 2500 },
      { id: 2, product: "Component Library - Pro", sku: "LIC-PRO", qty: 2, rate: 499, amount: 998 },
      { id: 3, product: "Implementation Support", sku: "SRV-042", qty: 4, rate: 188, amount: 752 },
    ],
    confidence: {
      invoiceNumber: 98,
      vendor: 99,
      date: 96,
      total: 97,
      dueDate: 94,
      poNumber: 92,
      lineItems: 95,
    },
  },
  receipt: {
    fileName: "Receipt_BlueBottle_0315.jpg",
    displayName: "BLUE BOTTLE",
    fields: {
      merchant: "Blue Bottle Coffee - Mint Plaza",
      date: "Mar 14, 2024 • 9:42 AM",
      total: "$8.50",
      payment: "Visa •• 4242",
      location: "San Francisco, CA",
      tax: "$0.72",
    },
    lineItems: [
      { id: 1, product: "Oat Milk Latte - Large", sku: "DRK-12", qty: 1, rate: 5.5, amount: 5.5 },
      { id: 2, product: "Almond Croissant", sku: "BAK-04", qty: 1, rate: 3, amount: 3 },
    ],
    confidence: {
      merchant: 97,
      date: 99,
      total: 98,
      payment: 95,
      location: 92,
      tax: 96,
      lineItems: 96,
    },
  },
  passport: {
    fileName: "Passport_USA_Morgan.pdf",
    displayName: "UNITED STATES",
    fields: {
      fullName: "ALEX R. MORGAN",
      passportNo: "N 4822917",
      dob: "14 APR 1992",
      nationality: "USA",
      expiry: "13 APR 2034",
      authority: "US Department of State",
      placeOfBirth: "San Francisco, CA",
    },
    lineItems: [],
    confidence: {
      fullName: 99,
      passportNo: 98,
      dob: 97,
      nationality: 99,
      expiry: 98,
      authority: 93,
      placeOfBirth: 94,
    },
  },
  contract: {
    fileName: "MSA_Linear_Acme.pdf",
    displayName: "MASTER AGREEMENT",
    fields: {
      parties: "Linear Inc. ↔ Acme Corp",
      effectiveDate: "Jan 01, 2024",
      term: "12 months auto-renew",
      value: "$120,000 / yr",
      governingLaw: "California, USA",
      signedBy: "Alex Rivera, CEO",
    },
    lineItems: [
      { id: 1, product: "Confidentiality Clause", sku: "P.3 • §4.2", qty: 1, rate: 0, amount: 0 },
      { id: 2, product: "Payment Terms - Net 30", sku: "P.5 • §7.1", qty: 1, rate: 0, amount: 0 },
      { id: 3, product: "Termination for Convenience", sku: "P.9 • §11.3", qty: 1, rate: 0, amount: 0 },
    ],
    confidence: {
      parties: 97,
      effectiveDate: 96,
      term: 95,
      value: 98,
      governingLaw: 92,
      signedBy: 94,
      lineItems: 90,
    },
  },
  bank: {
    fileName: "Statement_Mercury_Feb.pdf",
    displayName: "MERCURY",
    fields: {
      accountHolder: "Alex Morgan • Linear Inc.",
      bankName: "Mercury •• 8821",
      period: "Feb 1 - Feb 29, 2024",
      opening: "$42,102.00",
      closing: "$48,291.20",
      currency: "USD",
    },
    lineItems: [
      { id: 1, product: "Stripe Payout", sku: "Feb 04 • Income", qty: 1, rate: 12400, amount: 12400 },
      { id: 2, product: "AWS EMEA", sku: "Feb 12 • Expense", qty: 1, rate: 842, amount: -842 },
      { id: 3, product: "Deel - Contractor Payroll", sku: "Feb 20 • Expense", qty: 1, rate: 5368, amount: -5368 },
    ],
    confidence: {
      accountHolder: 98,
      bankName: 99,
      period: 97,
      opening: 96,
      closing: 98,
      currency: 99,
      lineItems: 94,
    },
  },
};

export const FEATURES = [
  { title: "No-code templates", desc: "Upload 5 samples and we learn. No regex, no drag-to-tag. Works from day one.", icon: "⚡" },
  { title: "99.2% accuracy", desc: "Finetuned vision + LLM validation. Totals are checked, dates normalized, duplicates flagged.", icon: "🎯" },
  { title: "API-first", desc: "REST, webhooks, SDKs in TS/Python/Go. 50ms p95 latency. Scale to millions.", icon: "🔌" },
  { title: "50+ languages", desc: "English to Japanese to Arabic. Handwriting, low-res scans, photos — all supported.", icon: "🌍" },
  { title: "SOC2 & GDPR", desc: "Encrypted at rest, zero data retention option. EU data residency. Your docs are yours.", icon: "🔒" },
  { title: "Human review", desc: "Optional human-in-the-loop for edge cases. 99.9% SLA with fallback queue.", icon: "👁️" },
];

export const USE_CASES = [
  { k: "Finance", desc: "Invoices, receipts, bank statements. Reconcile in seconds.", metric: "12M pages/mo", color: "from-violet-600 to-indigo-600" },
  { k: "HR", desc: "Resumes, IDs, contracts. Onboard faster without manual entry.", metric: "89% faster", color: "from-emerald-600 to-teal-600" },
  { k: "Logistics", desc: "BOLs, packing lists, customs forms. Track shipments auto.", metric: "4.2k docs/day", color: "from-amber-500 to-orange-600" },
  { k: "Healthcare", desc: "Lab reports, insurance claims, patient forms. HIPAA ready.", metric: "SOC2 + HIPAA", color: "from-zinc-900 to-zinc-700" },
];

export const STEPS = [
  {
    n: "01",
    title: "Upload any doc",
    desc: "PDF, JPG, PNG, even photos of paper. Drag & drop or use API. We handle 100+ types out of the box.",
    color: "bg-zinc-900",
    icon: "M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z",
  },
  {
    n: "02",
    title: "AI extracts & validates",
    desc: "Vision model + cross-field validation. Totals match line items, dates make sense, confidence scored.",
    color: "bg-violet-600",
    icon: "M9.5 2A7.5 7.5 0 002 9.5c0 5.424 7.5 11.5 7.5 11.5S17 14.924 17 9.5A7.5 7.5 0 009.5 2z",
  },
  {
    n: "03",
    title: "Export to API / Excel / Webhook",
    desc: "Get JSON, CSV, or push directly to your stack. Webhooks, Zapier, or our SDK.",
    color: "bg-emerald-600",
    icon: "M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3",
  },
];

export const PRICING = [
  {
    name: "Starter",
    price: 29,
    pages: "500 pages",
    desc: "For freelancers & small teams getting started.",
    features: ["500 pages / mo", "5 doc types", "API access", "Email support", "JSON / CSV export"],
    cta: "Start free",
    popular: false,
  },
  {
    name: "Pro",
    price: 99,
    pages: "5,000 pages",
    desc: "For growing teams automating ops.",
    features: ["5,000 pages / mo", "100+ doc types", "Webhook + Zapier", "Priority support", "Human review add-on", "SOC2 logs"],
    cta: "Start free trial",
    popular: true,
  },
  {
    name: "Enterprise",
    price: null,
    pages: "Unlimited",
    desc: "For companies extracting at scale.",
    features: ["Unlimited pages", "Custom models", "On-prem / VPC", "Dedicated success", "SLA & audit logs", "Custom retention"],
    cta: "Contact sales",
    popular: false,
  },
];
