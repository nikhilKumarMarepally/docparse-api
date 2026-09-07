import { apiUrl, fetchApi } from "./apiBase";

export type SectionBounds = {
  min_x: number;
  min_y: number;
  max_x: number;
  max_y: number;
};

export type YoloDetection = {
  index: number;
  class: string;
  confidence: number;
  bounds: SectionBounds;
  label?: string;
  bbox_yolo?: number[];
};

export type YoloSectionsResponse = {
  job_id?: string;
  page_index?: number;
  image_width: number;
  image_height: number;
  bbox_format: string;
  yolo_format: string;
  source: string;
  model_path?: string | null;
  sections?: Array<Record<string, unknown>>;
  detections: YoloDetection[];
};

export type YoloSectionBounds = {
  class: string;
  bbox: [number, number, number, number];
  bbox_yolo: [number, number, number, number];
  confidence: number;
  index: number;
  label: string;
};

export type YoloBoundsResponse = {
  job_id: string;
  page_index: number;
  image_width: number;
  image_height: number;
  bbox_format: string;
  yolo_format: string;
  source: string;
  sections: YoloSectionBounds[];
};

export type SectionResult = {
  index: number;
  label: string;
  bounds?: SectionBounds;
  kept: boolean;
  preprocess_kept?: boolean;
  content_gate?: {
    extractable: boolean;
    confidence: number;
    provider?: string;
    model?: string;
  } | null;
  filter_reason: string | null;
  ocr_preview: string;
  fields: Record<string, unknown> | null;
};

export type PageResult = {
  page_index: number;
  sections: SectionResult[];
  merged_fields: Record<string, unknown>;
};

export type JobResponse = {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  step?: string;
  filename?: string;
  error?: string;
  source?: { filename: string; pages: number };
  pages?: PageResult[];
  merged_fields?: Record<string, unknown>;
  warnings?: string[];
};

export type UploadResult = {
  jobId: string;
  pageCount?: number;
  creditsCharged?: number;
  creditsRemaining?: number;
};

function parseErrorDetail(err: Record<string, unknown>, status: number): string {
  const detail = err.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0] && typeof detail[0] === "object" && "msg" in detail[0]) {
    return String((detail[0] as { msg: string }).msg);
  }
  return status === 402 ? "You ran out of credits." : "Upload failed. Please try again.";
}

function friendlyUploadError(
  raw: string,
  status?: number,
  creditsPerDocument = 2,
  initialCredits = 20,
): string {
  const msg = raw.toLowerCase();
  if (status === 402 || msg.includes("ran out of credits") || msg.includes("credits")) {
    return raw.includes("credits")
      ? raw
      : `You ran out of credits. Each page costs ${creditsPerDocument} credits; new accounts start with ${initialCredits} credits.`;
  }
  if (msg.includes("exceeds") || msg.includes("20 mb")) {
    return "This file is too large. Please use a file under 20 MB.";
  }
  if (msg.includes("unsupported") || msg.includes("file type")) {
    return "This file type isn't supported. Try PDF, PNG, or JPG.";
  }
  if (msg.includes("sign in") || msg.includes("401")) {
    return "Please sign in with email or Google before uploading.";
  }
  return "Upload failed. Please try again.";
}

export const PROCESSING_PHASES = [
  { label: "Preparing your document..." },
  { label: "Reading the information" },
  { label: "Analyzing the information" },
  { label: "Extracting the information" },
  { label: "Finishing up..." },
] as const;

const STEP_TO_PHASE: Record<string, number> = {
  queued: 0,
  upload: 0,
  rasterize: 0,
  pipeline: 0,
  ocr: 1,
  sections: 2,
  filter: 2,
  extract: 3,
  overlay: 4,
  done: 4,
};

export function parseBackendStep(step?: string): string {
  if (!step) return "queued";
  return step.split(":").pop() || step;
}

export function phaseIndexFromStep(step?: string): number {
  const base = parseBackendStep(step);
  if (base === "failed") return 0;
  return STEP_TO_PHASE[base] ?? 0;
}

export function stepLabel(step?: string): string {
  const idx = phaseIndexFromStep(step);
  return PROCESSING_PHASES[idx].label;
}

/** @deprecated Use phaseIndexFromStep — kept for any legacy callers */
export function stepIndex(step?: string): number {
  return phaseIndexFromStep(step);
}

export function sampleDocUrl(docType: string): string {
  return `/samples/${docType}.png`;
}

export function sampleOverlayUrl(docType: string): string {
  return `/samples/${docType}-overlay.png`;
}

export async function uploadFile(file: File, token?: string | null): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(apiUrl("/api/jobs"), { method: "POST", body: form, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(friendlyUploadError(parseErrorDetail(err, res.status), res.status));
  }
  const data = await res.json();
  return {
    jobId: data.job_id as string,
    pageCount: typeof data.page_count === "number" ? data.page_count : undefined,
    creditsCharged: typeof data.credits_charged === "number" ? data.credits_charged : undefined,
    creditsRemaining:
      typeof data.credits_remaining === "number" ? data.credits_remaining : undefined,
  };
}

export async function pollJob(jobId: string, attempts = 3): Promise<JobResponse> {
  let lastError: Error | null = null;
  for (let i = 0; i < attempts; i += 1) {
    try {
      const res = await fetch(apiUrl(`/api/jobs/${jobId}`));
      if (res.status === 404 && i < attempts - 1) {
        await new Promise((r) => window.setTimeout(r, 1500));
        continue;
      }
      if (!res.ok) throw new Error("Job not found");
      return res.json();
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e));
      if (i < attempts - 1) {
        await new Promise((r) => window.setTimeout(r, 1500));
      }
    }
  }
  throw lastError ?? new Error("Job not found");
}

export function pageSourceUrl(jobId: string, pageIndex: number): string {
  return apiUrl(`/api/jobs/${jobId}/pages/${pageIndex}/source.png`);
}

export function overlayUrl(jobId: string, pageIndex: number): string {
  return apiUrl(`/api/jobs/${jobId}/pages/${pageIndex}/overlay.png`);
}

export function cropUrl(jobId: string, pageIndex: number, sectionIndex: number): string {
  return apiUrl(`/api/jobs/${jobId}/pages/${pageIndex}/sections/${sectionIndex}/crop.png`);
}

export function yoloBoundsUrl(jobId: string, pageIndex: number): string {
  return apiUrl(`/api/jobs/${jobId}/pages/${pageIndex}/yolo_bounds`);
}

export async function fetchYoloSections(
  input: { jobId: string; pageIndex: number } | { file: File },
): Promise<YoloSectionsResponse> {
  if ("file" in input) {
    const form = new FormData();
    form.append("file", input.file);
    const res = await fetch(apiUrl("/api/yolo/sections"), { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(parseErrorDetail(err, res.status));
    }
    return res.json();
  }

  const form = new FormData();
  form.append("job_id", input.jobId);
  form.append("page_index", String(input.pageIndex));
  const res = await fetch(apiUrl("/api/yolo/sections"), { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(parseErrorDetail(err, res.status));
  }
  return res.json();
}

/** GET alias for job page bounds (same payload as POST /api/yolo/sections). */
export async function fetchYoloBounds(jobId: string, pageIndex: number): Promise<YoloSectionsResponse> {
  const res = await fetchApi(`/api/jobs/${jobId}/pages/${pageIndex}/yolo_bounds`);
  if (!res.ok) throw new Error("YOLO bounds not found");
  return res.json();
}
