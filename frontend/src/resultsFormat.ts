import { JobResponse, SectionResult } from "./api";

export type DisplayTable = {
  id: string;
  label: string;
  columns: string[];
  rows: Record<string, unknown>[];
};

const TABLE_KEYS = new Set(["line_items", "table_columns"]);

function isScalar(value: unknown): boolean {
  return (
    value === null ||
    value === undefined ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  );
}

function columnsFrom(rows: Record<string, unknown>[], columns: unknown): string[] {
  if (Array.isArray(columns) && columns.length > 0) {
    return columns.map(String);
  }
  const keys = new Set<string>();
  for (const row of rows) {
    Object.keys(row).forEach((k) => keys.add(k));
  }
  return [...keys];
}

function flattenSingleRowLineItems(
  fields: Record<string, unknown>,
): [string, string][] {
  const rawRows = fields.line_items;
  if (!Array.isArray(rawRows) || rawRows.length !== 1) return [];
  const row = rawRows[0];
  if (!row || typeof row !== "object") return [];
  return Object.entries(row as Record<string, unknown>)
    .filter(([, value]) => isScalar(value))
    .map(([key, value]) => [key, value == null ? "" : String(value)]);
}

function parseTable(
  fields: Record<string, unknown> | null | undefined,
  label: string,
  id: string,
): DisplayTable | null {
  if (!fields) return null;
  const rawRows = fields.line_items;
  if (!Array.isArray(rawRows) || rawRows.length < 2) return null;
  const rows = rawRows.filter((r) => r && typeof r === "object") as Record<string, unknown>[];
  if (rows.length < 2) return null;
  return {
    id,
    label,
    columns: columnsFrom(rows, fields.table_columns),
    rows,
  };
}

export function collectScalarFields(fields: Record<string, unknown> | undefined): [string, string][] {
  if (!fields) return [];
  const scalars = Object.entries(fields)
    .filter(([key, value]) => !TABLE_KEYS.has(key) && isScalar(value))
    .map(([key, value]) => [key, value == null ? "" : String(value)] as [string, string]);
  const fromSingleRow = flattenSingleRowLineItems(fields);
  const seen = new Set(scalars.map(([k]) => k));
  for (const [k, v] of fromSingleRow) {
    if (!seen.has(k)) scalars.push([k, v]);
  }
  return scalars;
}

export function collectTables(job: JobResponse): DisplayTable[] {
  const seen = new Set<string>();
  const tables: DisplayTable[] = [];

  const push = (table: DisplayTable | null) => {
    if (!table) return;
    const sig = JSON.stringify({ columns: table.columns, rows: table.rows });
    if (seen.has(sig)) return;
    seen.add(sig);
    tables.push(table);
  };

  push(parseTable(job.merged_fields, "Table", "merged"));

  for (const page of job.pages || []) {
    for (const section of page.sections) {
      // Include preprocess-red sections — extraction is not filtered by kept.
      if (!section.fields) continue;
      const label = section.label?.replace(/_/g, " ") || `Region ${section.index + 1}`;
      push(
        parseTable(
          section.fields as Record<string, unknown>,
          label,
          `p${page.page_index}-s${section.index}`,
        ),
      );
    }
  }

  return tables;
}

export function tableFromFields(
  fields: Record<string, unknown> | null | undefined,
  label: string,
  id: string,
): DisplayTable | null {
  return parseTable(fields, label, id);
}

export function sectionDisplayTitle(section: SectionResult): string {
  const preview = section.ocr_preview?.trim();
  if (preview) {
    const first = preview.split("\n").find((line) => line.trim())?.trim();
    if (first) {
      return first.length > 72 ? `${first.slice(0, 69)}…` : first;
    }
  }
  return section.label?.replace(/_/g, " ") || `Region ${section.index + 1}`;
}

export function formatFieldLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
