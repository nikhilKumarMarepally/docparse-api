import { useEffect, useState } from "react";
import { PROCESSING_PHASES, phaseIndexFromStep, stepLabel } from "../api";

type Props = {
  step?: string;
  light?: boolean;
};

function useAnimatedProgress(step?: string) {
  const phaseIdx = phaseIndexFromStep(step);
  const label = stepLabel(step);
  const phaseCount = PROCESSING_PHASES.length;
  const minPercent = (phaseIdx / phaseCount) * 100;
  const maxPercent = Math.min(((phaseIdx + 1) / phaseCount) * 100 - 1, 97);

  const [percent, setPercent] = useState(minPercent);

  useEffect(() => {
    setPercent((current) => Math.max(current, minPercent));
  }, [minPercent]);

  useEffect(() => {
    const id = window.setInterval(() => {
      setPercent((current) => {
        if (current >= maxPercent) return current;
        return Math.min(current + 0.35, maxPercent);
      });
    }, 180);
    return () => window.clearInterval(id);
  }, [maxPercent]);

  return { percent, phaseIdx, label };
}

function StepList({ phaseIdx, compact }: { phaseIdx: number; compact?: boolean }) {
  return (
    <ol className={compact ? "mt-5 space-y-2" : "mt-5 space-y-2.5"}>
      {PROCESSING_PHASES.map((phase, i) => {
        const done = i < phaseIdx;
        const active = i === phaseIdx;
        return (
          <li
            key={phase.label}
            className={`flex items-center gap-2.5 text-[13px] leading-snug transition-colors duration-300 ${
              done
                ? "text-emerald-600"
                : active
                  ? "text-violet-700 font-medium"
                  : "text-zinc-400"
            }`}
          >
            <span
              className={`flex-shrink-0 w-5 h-5 rounded-full grid place-items-center text-[11px] font-semibold border transition-all duration-300 ${
                done
                  ? "bg-emerald-500 border-emerald-500 text-white"
                  : active
                    ? "bg-violet-600 border-violet-600 text-white animate-pulse"
                    : "bg-white border-zinc-200 text-zinc-400"
              }`}
              aria-hidden
            >
              {done ? "✓" : i + 1}
            </span>
            <span className={active ? "text-[14px]" : ""}>{phase.label}</span>
          </li>
        );
      })}
    </ol>
  );
}

export default function ProcessingBar({ step, light }: Props) {
  const { percent, phaseIdx, label } = useAnimatedProgress(step);

  if (light) {
    return (
      <div className="max-w-[640px] mx-auto px-5 pt-8">
        <div className="rounded-[16px] border border-black/[0.06] bg-white p-6 shadow-sm">
          <p className="text-[22px] sm:text-[24px] font-semibold tracking-[-0.02em] text-zinc-900 leading-tight">
            {label}
          </p>
          <div className="mt-5 h-2 rounded-full bg-zinc-100 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-violet-500 to-indigo-500 transition-[width] duration-500 ease-out"
              style={{ width: `${percent}%` }}
              role="progressbar"
              aria-valuenow={Math.round(percent)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={label}
            />
          </div>
          <StepList phaseIdx={phaseIdx} compact />
        </div>
      </div>
    );
  }

  return (
    <div className="processing-bar">
      <div className="processing-inner">
        <p className="processing-title">{label}</p>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${percent}%` }} />
        </div>
        <StepList phaseIdx={phaseIdx} />
      </div>
    </div>
  );
}
