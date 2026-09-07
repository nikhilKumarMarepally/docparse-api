/** Rough PDF page count from file bytes (no extra deps). */
export async function countPdfPages(file: File): Promise<number> {
  const buf = await file.arrayBuffer();
  const text = new TextDecoder("latin1").decode(new Uint8Array(buf));
  const matches = text.match(/\/Type\s*\/Page\b(?!s)/g);
  return Math.max(1, matches?.length ?? 1);
}

export async function countUploadPages(file: File): Promise<number> {
  const name = file.name.toLowerCase();
  if (name.endsWith(".pdf")) {
    return countPdfPages(file);
  }
  return 1;
}
