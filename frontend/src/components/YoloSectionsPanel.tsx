import { useCallback, useEffect, useRef, useState } from "react";
import { fetchYoloSections, pageSourceUrl, YoloSectionsResponse } from "../api";

type Props = {
  jobId: string;
  pageIndex: number;
  onResponse: (payload: { response: YoloSectionsResponse | null; imageUrl: string }) => void;
};

export default function YoloSectionsPanel({ jobId, pageIndex, onResponse }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [response, setResponse] = useState<YoloSectionsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadName, setUploadName] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const publish = useCallback(
    (next: YoloSectionsResponse | null, imageUrl: string) => {
      setResponse(next);
      onResponse({ response: next, imageUrl });
    },
    [onResponse],
  );

  const loadJobPage = useCallback(async () => {
    setLoading(true);
    setError(null);
    setUploadName(null);
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    try {
      const data = await fetchYoloSections({ jobId, pageIndex });
      publish(data, pageSourceUrl(jobId, pageIndex));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      publish(null, pageSourceUrl(jobId, pageIndex));
    } finally {
      setLoading(false);
    }
  }, [jobId, pageIndex, publish]);

  useEffect(() => {
    void loadJobPage();
  }, [loadJobPage]);

  useEffect(
    () => () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    [],
  );

  const onUpload = async (file: File) => {
    setLoading(true);
    setError(null);
    setUploadName(file.name);
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const objectUrl = URL.createObjectURL(file);
    objectUrlRef.current = objectUrl;
    try {
      const data = await fetchYoloSections({ file });
      publish(data, objectUrl);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      publish(null, objectUrl);
    } finally {
      setLoading(false);
    }
  };

  const copyJson = async () => {
    if (!response) return;
    await navigator.clipboard.writeText(JSON.stringify(response, null, 2));
  };

  return (
    <div className="json-panel yolo-sections-panel">
      <div className="json-toolbar">
        <span className="json-filename mono">
          {uploadName || `page_${pageIndex}_yolo_sections.json`}
        </span>
        <div className="json-actions">
          <button type="button" className="btn-compact" onClick={() => void loadJobPage()} disabled={loading}>
            Refresh
          </button>
          <button
            type="button"
            className="btn-compact"
            onClick={() => inputRef.current?.click()}
            disabled={loading}
          >
            Upload image
          </button>
          <button type="button" className="btn-compact" onClick={() => void copyJson()} disabled={!response}>
            Copy
          </button>
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".png,.jpg,.jpeg,.tif,.tiff,.webp"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void onUpload(file);
          e.target.value = "";
        }}
      />
      {loading && <p className="empty">Loading YOLO section bounds…</p>}
      {error && <p className="empty">{error}</p>}
      {response && (
        <>
          <p className="yolo-meta">
            {response.detections.length} detection{response.detections.length === 1 ? "" : "s"} ·{" "}
            {response.image_width}×{response.image_height}px · {response.source}
            {response.model_path ? ` · ${response.model_path}` : ""}
          </p>
          <pre className="json-pre">{JSON.stringify(response, null, 2)}</pre>
        </>
      )}
    </div>
  );
}
