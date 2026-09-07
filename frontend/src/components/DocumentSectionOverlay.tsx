import { useCallback, useEffect, useRef, useState } from "react";
import { SectionResult } from "../api";

type Props = {
  imageUrl: string;
  sections: SectionResult[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
  boxClassName?: string;
};

export default function DocumentSectionOverlay({
  imageUrl,
  sections,
  selectedIndex,
  onSelect,
  boxClassName = "doc-section-box",
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [natural, setNatural] = useState({ w: 0, h: 0 });
  const [displayW, setDisplayW] = useState(0);

  const measure = useCallback(() => {
    const el = wrapRef.current;
    if (!el) return;
    setDisplayW(el.clientWidth);
  }, []);

  useEffect(() => {
    setNatural({ w: 0, h: 0 });
  }, [imageUrl]);

  useEffect(() => {
    measure();
    const el = wrapRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [measure, imageUrl]);

  const scale = natural.w > 0 ? displayW / natural.w : 1;
  const displayH = natural.h > 0 ? natural.h * scale : undefined;

  return (
    <div ref={wrapRef} className="doc-section-overlay">
      <img
        src={imageUrl}
        alt="Document page"
        className="doc-section-overlay__img"
        onLoad={(e) => {
          const img = e.currentTarget;
          setNatural({ w: img.naturalWidth, h: img.naturalHeight });
          measure();
        }}
      />
      {natural.w > 0 && (
        <div className="doc-section-overlay__layer" style={{ width: displayW, height: displayH }}>
          {sections.map((section) => {
            const b = section.bounds;
            if (!b) return null;
            const w = Math.max(1, (b.max_x - b.min_x) * scale);
            const h = Math.max(1, (b.max_y - b.min_y) * scale);
            const selected = selectedIndex === section.index;
            return (
              <button
                key={section.index}
                type="button"
                className={`${boxClassName}${selected ? " doc-section-box--selected" : ""}`}
                style={{
                  left: b.min_x * scale,
                  top: b.min_y * scale,
                  width: w,
                  height: h,
                }}
                aria-label={`Section ${section.index + 1}`}
                aria-pressed={selected}
                onClick={() => onSelect(section.index)}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
