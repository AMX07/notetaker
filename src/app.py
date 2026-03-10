"""FastAPI web application for notetaker."""

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from .audio import extract_audio
from .frames import extract_frames_hybrid, frames_from_json, frames_to_json
from .llm import run_agent_loop
from .segmenter import align_frames_to_segments, segment_by_time_windows
from .transcribe import segments_from_json, segments_to_json, transcribe_audio

load_dotenv()

# Logging setup — writes to notetaker.log and stdout
LOG_PATH = Path(__file__).parent.parent / "notetaker.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=3),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("notetaker")

# Job TTL and pipeline timeout (configurable via environment)
JOB_TTL_HOURS = int(os.environ.get("JOB_TTL_HOURS", "24"))
PIPELINE_TIMEOUT_SECONDS = int(os.environ.get("PIPELINE_TIMEOUT_SECONDS", "1800"))  # 30 min
FFMPEG_TIMEOUT_SECONDS = 300  # 5 min per ffmpeg call

# Rate limiting — simple in-memory tracker (per-IP, per-hour)
UPLOAD_RATE_LIMIT = int(os.environ.get("UPLOAD_RATE_LIMIT", "5"))  # requests per hour
_rate_limit_store: dict[str, list[float]] = defaultdict(list)

# Authentication — optional HTTP Basic Auth (set NOTETAKER_PASSWORD in .env to enable)
security = HTTPBasic(auto_error=False)
AUTH_PASSWORD = os.environ.get("NOTETAKER_PASSWORD", "")


def verify_auth(credentials: HTTPBasicCredentials | None = Depends(security)):
    """Verify HTTP Basic Auth credentials if NOTETAKER_PASSWORD is set."""
    if not AUTH_PASSWORD:
        return
    if not credentials or not secrets.compare_digest(credentials.password, AUTH_PASSWORD):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


def _check_rate_limit(client_ip: str) -> int | None:
    """Check if client_ip has exceeded the upload rate limit.

    Returns seconds until next allowed request, or None if allowed.
    """
    now = time.time()
    window = 3600  # 1 hour
    timestamps = _rate_limit_store[client_ip]
    # Prune old entries
    _rate_limit_store[client_ip] = [t for t in timestamps if now - t < window]
    timestamps = _rate_limit_store[client_ip]

    if len(timestamps) >= UPLOAD_RATE_LIMIT:
        retry_after = int(timestamps[0] + window - now) + 1
        return retry_after
    _rate_limit_store[client_ip].append(now)
    return None


async def _cleanup_expired_jobs() -> None:
    """Periodically remove job directories older than JOB_TTL_HOURS."""
    while True:
        await asyncio.sleep(600)  # check every 10 minutes
        now = time.time()
        ttl_seconds = JOB_TTL_HOURS * 3600
        removed = 0
        for job_dir in JOBS_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            job_id = job_dir.name
            # Don't remove running jobs
            if job_id in jobs and jobs[job_id]["status"] == "processing":
                continue
            try:
                mtime = job_dir.stat().st_mtime
                if now - mtime > ttl_seconds:
                    shutil.rmtree(job_dir)
                    jobs.pop(job_id, None)
                    removed += 1
            except OSError:
                pass
        if removed:
            logger.info(f"Cleaned up {removed} expired job(s)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup validation, job restore, cleanup task."""
    _validate_environment()
    _restore_jobs_from_disk()
    cleanup_task = asyncio.create_task(_cleanup_expired_jobs())
    yield
    cleanup_task.cancel()


app = FastAPI(
    title="Notetaker",
    description="Convert video lectures into well-structured markdown notes. "
    "Upload a video, get back a document preserving the speaker's voice with minimal edits.",
    version="0.2.0",
    lifespan=lifespan,
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
JOBS_DIR = PROJECT_ROOT / "output" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Serve static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory job store — rebuilt from disk on startup
jobs: dict[str, dict] = {}

PIPELINE_STEPS = ["extract", "transcribe", "segment", "agent"]

# Upload limits
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}

# Concurrency guard — limits parallel pipeline jobs
MAX_CONCURRENT_JOBS = 2
_job_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)


def _sanitize_filename(filename: str) -> str:
    """Strip path components and dangerous characters from an upload filename."""
    name = Path(filename).name
    name = re.sub(r"[^\w\-.]", "_", name)
    return name or "upload.mp4"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _checkpoint_path(job_dir: Path) -> Path:
    return job_dir / "checkpoint.json"


def _save_checkpoint(job_dir: Path, checkpoint: dict) -> None:
    _checkpoint_path(job_dir).write_text(json.dumps(checkpoint, indent=2))


def _load_checkpoint(job_dir: Path) -> dict | None:
    cp = _checkpoint_path(job_dir)
    if cp.exists():
        return json.loads(cp.read_text())
    return None


def _new_checkpoint(job_id: str, video_filename: str, title: str, language: str | None) -> dict:
    return {
        "job_id": job_id,
        "video_filename": video_filename,
        "title": title,
        "language": language,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": {step: {"status": "pending"} for step in PIPELINE_STEPS},
    }


def _mark_step(checkpoint: dict, step: str, status: str, error: str | None = None) -> None:
    checkpoint["steps"][step] = {
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        checkpoint["steps"][step]["error"] = error


def _step_completed(checkpoint: dict, step: str) -> bool:
    return checkpoint["steps"].get(step, {}).get("status") == "completed"


# ---------------------------------------------------------------------------
# Restore jobs from disk on startup
# ---------------------------------------------------------------------------

# Final step name — check both old and new checkpoint formats
_FINAL_STEPS = {"agent", "assemble"}


def _restore_jobs_from_disk() -> None:
    """Scan output/jobs/ and rebuild the in-memory jobs dict."""
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        cp = _load_checkpoint(job_dir)
        if not cp:
            continue

        job_id = cp["job_id"]
        steps = cp["steps"]

        # Check completion — handle both old format (assemble) and new (agent)
        final_completed = any(steps.get(s, {}).get("status") == "completed" for s in _FINAL_STEPS)

        if final_completed:
            status = "completed"
            stage = "Done"
        elif any(s.get("status") == "failed" for s in steps.values()):
            failed_step = next(k for k, v in steps.items() if v.get("status") == "failed")
            status = "error"
            stage = f"Failed at: {failed_step}"
        else:
            status = "incomplete"
            stage = "Incomplete"

        result_md = job_dir / "result.md"
        output_path = str(result_md) if result_md.exists() else None

        completed_count = sum(1 for s in steps.values() if s.get("status") == "completed")
        total = len(steps)

        jobs[job_id] = {
            "status": status,
            "stage": stage,
            "substage": "",
            "progress": int(completed_count / total * 100) if total else 0,
            "total": 100,
            "error": None,
            "output_path": output_path,
            "work_dir": str(job_dir),
            "video_filename": cp["video_filename"],
        }

    logger.info(f"Restored {len(jobs)} jobs from disk")


def _validate_environment() -> None:
    """Fail fast if required tools or API keys are missing."""
    errors = []
    if not shutil.which("ffmpeg"):
        errors.append("ffmpeg not found in PATH")
    if not shutil.which("ffprobe"):
        errors.append("ffprobe not found in PATH")
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is not set")
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_aws = bool(os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    if not has_anthropic and not has_aws:
        errors.append("Neither ANTHROPIC_API_KEY nor AWS_REGION is set")
    if errors:
        for e in errors:
            logger.error(f"STARTUP CHECK FAILED: {e}")
        raise RuntimeError(f"Missing requirements: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False, dependencies=[Depends(verify_auth)])
async def index():
    """Serve the frontend."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.post("/api/convert", summary="Upload video and start conversion", tags=["Jobs"], dependencies=[Depends(verify_auth)])
async def start_conversion(
    request: Request,
    video: UploadFile = File(...),
    title: str = Form(default=""),
    language: str = Form(default=""),
):
    """Upload a video file and start the conversion pipeline.

    Accepts MP4, MKV, MOV, WebM, and AVI files up to 2 GB.
    Returns a job ID for polling status.
    """
    # Rate limit check
    client_ip = request.client.host if request.client else "unknown"
    retry_after = _check_rate_limit(client_ip)
    if retry_after is not None:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded. Try again in {retry_after} seconds."},
            headers={"Retry-After": str(retry_after)},
        )

    # Validate file extension
    safe_name = _sanitize_filename(video.filename or "upload.mp4")
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Validate content type
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Check concurrency limit
    if _job_semaphore._value == 0:
        raise HTTPException(status_code=429, detail="Too many jobs running, try again later")

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Stream uploaded file to disk with size limit
    video_path = job_dir / safe_name
    total_bytes = 0
    try:
        with open(video_path, "wb") as f:
            while chunk := await video.read(8192):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    video_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024**3)}GB limit",
                    )
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    # Create checkpoint
    checkpoint = _new_checkpoint(job_id, safe_name, title or safe_name, language or None)
    _save_checkpoint(job_dir, checkpoint)

    jobs[job_id] = {
        "status": "queued",
        "stage": "",
        "substage": "",
        "progress": 0,
        "total": 100,
        "error": None,
        "output_path": None,
        "work_dir": str(job_dir),
        "video_filename": safe_name,
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id,),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.post("/api/resume/{job_id}", summary="Resume a failed job", tags=["Jobs"], dependencies=[Depends(verify_auth)])
async def resume_job(job_id: str):
    """Resume a failed or incomplete job from its last checkpoint."""
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    cp = _load_checkpoint(job_dir)
    if not cp:
        raise HTTPException(status_code=404, detail="No checkpoint found")

    # Check if already completed (handle both old and new formats)
    if any(cp["steps"].get(s, {}).get("status") == "completed" for s in _FINAL_STEPS):
        raise HTTPException(status_code=400, detail="Job already completed")

    completed_count = sum(1 for s in cp["steps"].values() if s.get("status") == "completed")
    total = len(cp["steps"])

    jobs[job_id] = {
        "status": "queued",
        "stage": "Resuming...",
        "substage": "",
        "progress": int(completed_count / total * 100) if total else 0,
        "total": 100,
        "error": None,
        "output_path": None,
        "work_dir": str(job_dir),
        "video_filename": cp["video_filename"],
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id,),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "message": "Resuming from last checkpoint"}


@app.get("/api/status/{job_id}", summary="Get job status", tags=["Jobs"], dependencies=[Depends(verify_auth)])
async def get_status(job_id: str):
    """Get the processing status of a job including progress percentage and current stage."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "status": job["status"],
        "stage": job["stage"],
        "substage": job.get("substage", ""),
        "progress": job["progress"],
        "total": job["total"],
        "error": job["error"],
    }


@app.get("/api/jobs", summary="List all jobs", tags=["Jobs"], dependencies=[Depends(verify_auth)])
async def list_jobs():
    """List all jobs with their current status."""
    return [
        {
            "job_id": jid,
            "status": j["status"],
            "stage": j["stage"],
            "video_filename": j["video_filename"],
        }
        for jid, j in jobs.items()
    ]


@app.get("/health", summary="Health check", tags=["System"])
async def health():
    """Health check endpoint for orchestrators and monitoring."""
    active = sum(1 for j in jobs.values() if j["status"] == "processing")
    return {"status": "ok", "version": "0.2.0", "active_jobs": active}


@app.delete("/api/jobs/{job_id}", summary="Delete a job", tags=["Jobs"], dependencies=[Depends(verify_auth)])
async def delete_job(job_id: str):
    """Delete a job and its files. Cannot delete a job that is currently processing."""
    if not re.fullmatch(r"[a-f0-9\-]{8}", job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    if job_id in jobs and jobs[job_id]["status"] == "processing":
        raise HTTPException(status_code=409, detail="Cannot delete a running job")
    shutil.rmtree(job_dir)
    jobs.pop(job_id, None)
    return {"status": "deleted"}


@app.get("/api/download/{job_id}", summary="Download result", tags=["Jobs"], dependencies=[Depends(verify_auth)])
async def download_result(job_id: str):
    """Download the generated markdown file for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    if not job["output_path"]:
        raise HTTPException(status_code=500, detail="No output file")

    return FileResponse(
        job["output_path"],
        media_type="text/markdown",
        filename=Path(job["output_path"]).name,
    )


# ---------------------------------------------------------------------------
# Pipeline: parallel workflow (steps 1-3) + agent loop (step 4)
# ---------------------------------------------------------------------------


def _run_pipeline(job_id: str):
    """Run the conversion pipeline with parallel extraction and agent-based LLM processing."""
    job = jobs[job_id]
    job_dir = Path(job["work_dir"])

    cp = _load_checkpoint(job_dir)
    if not cp:
        logger.error(f"[{job_id}] No checkpoint found")
        job.update(status="error", error="No checkpoint found")
        return

    video_path = job_dir / cp["video_filename"]
    title = cp["title"]
    language = cp["language"]

    _job_semaphore.acquire()
    pipeline_start = time.time()
    try:
        job.update(status="processing")

        audio_path = job_dir / "audio.mp3"
        frames_json_path = job_dir / "frames.json"
        frames_dir = job_dir / "frames"

        # === Step 1: Extract audio + frames concurrently ===
        if _step_completed(cp, "extract"):
            logger.info(f"[{job_id}] Skipping extract (checkpoint)")
            frames = frames_from_json(frames_json_path)
        else:
            logger.info(f"[{job_id}] Extracting audio & frames (parallel)")
            job.update(stage="Extracting audio & frames", substage="", progress=5)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                audio_future = executor.submit(extract_audio, video_path, audio_path)
                frames_future = executor.submit(extract_frames_hybrid, video_path, frames_dir)

                audio_path = audio_future.result()
                frames = frames_future.result()

            audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
            logger.info(f"[{job_id}] Audio: {audio_size_mb:.1f}MB, Frames: {len(frames)}")
            frames_to_json(frames, frames_json_path)
            _mark_step(cp, "extract", "completed")
            _save_checkpoint(job_dir, cp)

        # Check pipeline timeout
        if time.time() - pipeline_start > PIPELINE_TIMEOUT_SECONDS:
            raise TimeoutError("Pipeline exceeded maximum time limit")

        # === Step 2: Transcribe (parallel Whisper chunks internally) ===
        transcript_path = job_dir / "transcript.json"
        if _step_completed(cp, "transcribe"):
            logger.info(f"[{job_id}] Skipping transcription (checkpoint)")
            transcript_segments = segments_from_json(transcript_path)
        else:
            logger.info(f"[{job_id}] Transcribing audio")
            job.update(stage="Transcribing audio", substage="", progress=20)
            transcript_segments = transcribe_audio(audio_path, language=language)
            segments_to_json(transcript_segments, transcript_path)
            logger.info(f"[{job_id}] Transcription done: {len(transcript_segments)} segments")
            _mark_step(cp, "transcribe", "completed")
            _save_checkpoint(job_dir, cp)

        # Check pipeline timeout
        if time.time() - pipeline_start > PIPELINE_TIMEOUT_SECONDS:
            raise TimeoutError("Pipeline exceeded maximum time limit")

        # === Step 3: Segment transcript + align frames ===
        segments_json_path = job_dir / "segments.json"
        if _step_completed(cp, "segment"):
            logger.info(f"[{job_id}] Skipping segmentation (checkpoint)")
            seg_data = json.loads(segments_json_path.read_text())
            windows = [(s["start"], s["end"]) for s in seg_data]
        else:
            logger.info(f"[{job_id}] Segmenting video")
            job.update(stage="Segmenting video", substage="", progress=40)
            windows = segment_by_time_windows(transcript_segments)
            seg_data = [{"start": s, "end": e} for s, e in windows]
            segments_json_path.write_text(json.dumps(seg_data, indent=2))
            logger.info(f"[{job_id}] Created {len(windows)} segments")
            _mark_step(cp, "segment", "completed")
            _save_checkpoint(job_dir, cp)

        video_segments = align_frames_to_segments(windows, frames, transcript_segments)

        # Check pipeline timeout
        if time.time() - pipeline_start > PIPELINE_TIMEOUT_SECONDS:
            raise TimeoutError("Pipeline exceeded maximum time limit")

        # === Step 4: LLM Agent (Opus 4.6 orchestrator + reviewer) ===
        if _step_completed(cp, "agent"):
            logger.info(f"[{job_id}] Skipping agent (checkpoint)")
            result_path = job_dir / "result.md"
        else:
            logger.info(f"[{job_id}] Starting LLM agent loop")
            job.update(stage="LLM agent processing", substage="", progress=45)

            def on_agent_progress(stage: str, detail: str, pct: int):
                logger.info(f"[{job_id}] Agent: {stage} — {detail}")
                job["stage"] = f"LLM: {stage}"
                job["substage"] = detail
                # Map agent's 0-100% to pipeline's 45-95% range
                job["progress"] = 45 + int(pct * 0.50)

            markdown_content = run_agent_loop(
                video_segments=video_segments,
                job_dir=job_dir,
                title=title,
                source_video=cp["video_filename"],
                on_progress=on_agent_progress,
            )

            # Ensure result.md exists (agent tools should have written it)
            result_path = job_dir / "result.md"
            if not result_path.exists():
                result_path.write_text(markdown_content)

            _mark_step(cp, "agent", "completed")
            _save_checkpoint(job_dir, cp)

        logger.info(f"[{job_id}] Completed: {result_path}")
        job.update(
            status="completed",
            stage="Done",
            substage="",
            progress=100,
            output_path=str(result_path),
        )

    except TimeoutError as e:
        logger.error(f"[{job_id}] Pipeline timeout: {e}")
        for step in PIPELINE_STEPS:
            if cp["steps"][step]["status"] == "pending":
                _mark_step(cp, step, "failed", error=str(e))
                _save_checkpoint(job_dir, cp)
                break
        job.update(
            status="error",
            stage="Timed out",
            error="Processing timed out. The video may be too long. You can try resuming.",
        )
    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}", exc_info=True)
        for step in PIPELINE_STEPS:
            if cp["steps"][step]["status"] == "pending":
                _mark_step(cp, step, "failed", error=str(e))
                _save_checkpoint(job_dir, cp)
                break
        job.update(status="error", error="Processing failed. Check server logs for details.")
    finally:
        _job_semaphore.release()


def main():
    """Entry point for the notetaker command."""
    print("Starting Notetaker server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
