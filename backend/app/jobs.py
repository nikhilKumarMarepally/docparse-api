from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.ingest import ingest_upload
from app.paths import JOB_ROOT, configure_web_env, ensure_script_path
from app.pipeline import run_pipeline

ensure_script_path()
configure_web_env()


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    step: str = "queued"
    filename: str = ""
    error: str | None = None
    result: dict[str, Any] | None = None
    job_dir: Path = field(default_factory=Path)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status.value,
            "step": self.step,
            "filename": self.filename,
        }
        if self.error:
            payload["error"] = self.error
        if self.result is not None:
            payload.update(self.result)
        return payload


_lock = threading.Lock()
_jobs: dict[str, Job] = {}


def create_job(upload_path: Path, filename: str, *, skip_llm: bool = False) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / f"upload{upload_path.suffix.lower()}"
    shutil.copy2(upload_path, dest)

    job = Job(job_id=job_id, filename=filename, job_dir=job_dir)
    with _lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_run_job, args=(job, dest, skip_llm), daemon=True)
    thread.start()
    return job


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def _set_step(job: Job, step: str) -> None:
    job.step = step
    job.status = JobStatus.PROCESSING


def _run_job(job: Job, upload_path: Path, skip_llm: bool) -> None:
    try:
        _set_step(job, "rasterize")
        pages = ingest_upload(upload_path, job.job_dir)

        def on_step(step: str) -> None:
            job.step = step

        _set_step(job, "pipeline")
        result = run_pipeline(
            job.job_id,
            job.job_dir,
            pages,
            filename=job.filename,
            on_step=on_step,
            skip_llm=skip_llm,
        )
        result_path = job.job_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2))
        job.result = result
        job.status = JobStatus.COMPLETED
        job.step = "done"
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.step = "failed"


def job_file_path(job_id: str, rel_path: str) -> Path | None:
    job = get_job(job_id)
    if job is None:
        return None
    path = (job.job_dir / rel_path).resolve()
    if not str(path).startswith(str(job.job_dir.resolve())):
        return None
    if not path.exists():
        return None
    return path
