"""
YouTube multimodal analyzer.

Downloads a YouTube video, extracts slide changes with SSIM, transcribes audio
with Faster-Whisper, analyzes slide screenshots with Qwen2-VL, and writes
records compatible with qa_Module.graphrag.document_loader.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from queue import Empty, Queue
from typing import Callable

logger = logging.getLogger(__name__)

WHISPER_PROMPT = (
    "IETF meeting, RFC, Internet-Draft, QUIC, TCP, UDP, congestion control, "
    "protocol, latency, bandwidth."
)

ProgressCallback = Callable[[str, int, str], None]

_MODELS_LOCK = threading.Lock()
_WHISPER_MODEL = None
_QWEN_MODEL = None
_QWEN_PROCESSOR = None
_PROCESS_VISION_INFO = None


@dataclass
class AnalyzerResult:
    meeting_name: str
    video_name: str
    video_url: str
    json_path: Path
    transcript_path: Path
    meeting_dir: Path
    total_slides: int


def analyze_youtube_to_meetgrag_json(
    video_url: str,
    database_dir: str | Path,
    meeting_name: str | None = None,
    work_root: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> AnalyzerResult:
    """Run the full multimodal pipeline and write meetGRAG-compatible JSON."""
    _require_url(video_url)
    database_dir = Path(database_dir)
    database_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(work_root) if work_root else database_dir / "_work"
    work_root.mkdir(parents=True, exist_ok=True)

    task_id = uuid.uuid4().hex[:8]
    work_dir = work_root / f"task_{task_id}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    slides_dir = work_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    _set(progress, "downloading", 2, "Downloading YouTube video...")
    video_path, inferred_title = _download_video(video_url, work_dir)
    meeting_name = _safe_meeting_name(meeting_name or inferred_title or Path(video_path).stem)
    video_name = f"{meeting_name}.mp4"

    _set(progress, "transcribing", 12, "Transcribing audio with Faster-Whisper...")
    whisper_model = _get_whisper_model()
    segments, _ = whisper_model.transcribe(
        str(video_path),
        initial_prompt=WHISPER_PROMPT,
        language="en",
    )
    transcript = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
    ]
    logger.info("Transcription complete: %d segments", len(transcript))

    _set(progress, "processing", 28, "Detecting slide changes and analyzing visuals...")
    extracted_slides = _extract_and_analyze_slides(video_path, slides_dir, progress)
    if not extracted_slides:
        raise RuntimeError("No slide changes were detected in this video.")

    _set(progress, "writing", 88, "Writing meetGRAG JSON...")
    records = _build_records(extracted_slides, transcript, video_name)
    meeting_dir = database_dir / meeting_name
    final_slides_dir = meeting_dir / "slides"
    final_slides_dir.mkdir(parents=True, exist_ok=True)

    for slide in extracted_slides:
        src = Path(slide["path"])
        dst = final_slides_dir / src.name
        if src.exists():
            shutil.copy2(src, dst)
        slide["final_image_path"] = str(dst.resolve())

    for record, slide in zip(records, extracted_slides):
        record["slide_image"] = slide.get("final_image_path", "")

    flat_json_path = database_dir / f"{_safe_file_stem(meeting_name)}.json"
    flat_json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    transcript_payload = {
        "status": "done",
        "meeting_metadata": {
            "meeting_name": meeting_name,
            "video_name": video_name,
            "video_url": video_url,
            "total_slides": len(records),
        },
        "slides": records,
    }
    transcript_path = meeting_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(transcript_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _upsert_link_json(database_dir / "link.json", meeting_name, video_url)
    _set(progress, "done", 100, f"Analysis complete: {len(records)} slide segments.")

    return AnalyzerResult(
        meeting_name=meeting_name,
        video_name=video_name,
        video_url=video_url,
        json_path=flat_json_path,
        transcript_path=transcript_path,
        meeting_dir=meeting_dir,
        total_slides=len(records),
    )


def _download_video(video_url: str, work_dir: Path) -> tuple[Path, str]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError("Missing dependency: pip install yt-dlp") from exc

    outtmpl = str(work_dir / "input_video.%(ext)s")
    info_title = ""
    ydl_opts = {
        "format": "bestvideo[height<=720][vcodec^=avc]+bestaudio/best[height<=720]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        info_title = info.get("title") or ""

    downloaded = sorted(work_dir.glob("input_video.*"))
    if not downloaded:
        raise RuntimeError("Video download failed: no output file was created")

    video_path = work_dir / "input_video.mp4"
    if downloaded[0] != video_path:
        if video_path.exists():
            video_path.unlink()
        downloaded[0].rename(video_path)
    return video_path, info_title


def _extract_and_analyze_slides(
    video_path: Path,
    slides_dir: Path,
    progress: ProgressCallback | None,
) -> list[dict]:
    try:
        import cv2
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies: pip install opencv-python scikit-image numpy"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1

    anchor_frame = None
    prev_saved_time = -2.0
    frame_count = 0
    saved_count = 0
    process_width = 500
    mask_x_start, mask_width = 0.73, 0.27

    extracted_slides: list[dict] = []
    slide_queue: Queue = Queue()
    slicing_done = threading.Event()
    worker_failed = threading.Event()
    worker_errors: list[str] = []

    def vlm_worker() -> None:
        while True:
            try:
                slide_info = slide_queue.get(timeout=1)
            except Empty:
                if slicing_done.is_set():
                    break
                continue

            try:
                data = _analyze_slide(slide_info["path"])
                slide_info["vlm_data"] = data
                _set(
                    progress,
                    "processing",
                    min(86, 30 + slide_info["slide_id"]),
                    f"Analyzed {slide_info['slide_id']} slide screenshots...",
                )
            except Exception as exc:
                logger.exception("VLM analysis failed for %s", slide_info.get("path"))
                worker_errors.append(str(exc))
                worker_failed.set()
                break
            finally:
                slide_queue.task_done()

    consumer_thread = threading.Thread(target=vlm_worker, daemon=True)
    consumer_thread.start()

    while True:
        if worker_failed.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % max(1, int(fps / 2)) == 0:
            current_time = frame_count / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            small_gray = cv2.resize(gray, (0, 0), fx=process_width / w, fy=process_width / w)
            _, sw = small_gray.shape
            masked = small_gray.copy()
            masked[:, int(sw * mask_x_start):int(sw * (mask_x_start + mask_width))] = 0

            is_new_slide = False
            if anchor_frame is None:
                is_new_slide = True
            else:
                score, _ = ssim(anchor_frame, masked, full=True)
                if score < 0.96 and (current_time - prev_saved_time) > 2:
                    is_new_slide = True

            if is_new_slide:
                ts_display = str(timedelta(seconds=int(current_time))).split(".")[0]
                if len(ts_display) < 8:
                    ts_display = f"0{ts_display}"

                saved_count += 1
                img_path = slides_dir / f"slide_{saved_count:03d}_{ts_display.replace(':', '-')}.jpg"
                cv2.imwrite(str(img_path), frame)

                anchor_frame = masked
                prev_saved_time = current_time
                slide_info = {
                    "slide_id": saved_count,
                    "filename": img_path.name,
                    "timestamp": ts_display,
                    "slide_start_sec": current_time,
                    "path": str(img_path),
                }
                extracted_slides.append(slide_info)
                slide_queue.put(slide_info)

        if frame_count % max(1, int(fps * 10)) == 0:
            pct = 30 + int((frame_count / total_frames) * 50)
            _set(progress, "processing", min(80, pct), "Scanning video for slide changes...")

        frame_count += 1

    cap.release()
    slicing_done.set()
    consumer_thread.join()
    if worker_errors:
        raise RuntimeError(f"VLM analysis failed: {worker_errors[0]}")
    return extracted_slides


def _analyze_slide(image_path: str) -> dict:
    import torch

    qwen_model, qwen_processor, process_vision_info = _get_qwen_model()
    prompt_text = """
You are an expert technical recorder for IETF meetings. Analyze this slide image.
Output ONLY a valid JSON object without any additional text or markdown formatting.
{
  "title": "String",
  "content": ["String"],
  "type": "String - e.g. Title Page/Diagram/Chart/Text",
  "chart_data": "String or null"
}
"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ],
    }]

    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
    )[0]

    try:
        clean_text = output_text.replace("```json", "").replace("```", "").strip()
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            clean_text = clean_text[start_idx:end_idx + 1]
        data = json.loads(clean_text)
        return {
            "title": str(data.get("title") or "Untitled Slide"),
            "content": data.get("content") if isinstance(data.get("content"), list) else [],
            "type": str(data.get("type") or "Unknown"),
            "chart_data": data.get("chart_data"),
        }
    except Exception:
        logger.warning("Qwen JSON parse failed for %s: %r", image_path, output_text[:300])
        return {
            "title": "Parse failed",
            "content": [output_text[:300]],
            "type": "Error",
            "chart_data": None,
        }


def _build_records(extracted_slides: list[dict], transcript: list[dict], video_name: str) -> list[dict]:
    records: list[dict] = []
    total_slides = len(extracted_slides)

    for i, slide_info in enumerate(extracted_slides):
        start = float(slide_info["slide_start_sec"])
        end = (
            float(extracted_slides[i + 1]["slide_start_sec"])
            if i + 1 < total_slides
            else max([float(s["end"]) for s in transcript], default=start)
        )
        matched_transcript = " ".join(
            s["text"] for s in transcript
            if float(s["end"]) > start and float(s["start"]) < end
        )
        vlm = slide_info.get("vlm_data") or {}
        content = vlm.get("content") if isinstance(vlm.get("content"), list) else []
        ocr_text = "\n".join([str(vlm.get("title") or "Untitled Slide"), *map(str, content)]).strip()
        chart_data = vlm.get("chart_data")

        combined_text = f"{matched_transcript} {ocr_text} {chart_data or ''}"
        rfc_matches = sorted(set(re.findall(r"RFC\s?\d+", combined_text, re.IGNORECASE)))
        protocol_matches = sorted(set(re.findall(r"\b(QUIC|TCP|UDP|HTTP|TLS)\b", combined_text, re.IGNORECASE)))

        records.append({
            "id": f"{i + 1:03d}",
            "video_name": video_name,
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "ocr_text": ocr_text,
            "chart_description": "" if chart_data in (None, "null", "String or null") else str(chart_data),
            "transcript": matched_transcript,
            "slide_image": slide_info.get("final_image_path", ""),
            "extracted_entities": {
                "documents_mentioned": rfc_matches,
                "protocols_mentioned": protocol_matches,
            },
        })
    return records


def _get_whisper_model():
    global _WHISPER_MODEL
    with _MODELS_LOCK:
        if _WHISPER_MODEL is not None:
            return _WHISPER_MODEL
        try:
            import torch
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError("Missing dependency: pip install faster-whisper torch") from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        _WHISPER_MODEL = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
        return _WHISPER_MODEL


def _get_qwen_model():
    global _QWEN_MODEL, _QWEN_PROCESSOR, _PROCESS_VISION_INFO
    with _MODELS_LOCK:
        if _QWEN_MODEL is not None:
            return _QWEN_MODEL, _QWEN_PROCESSOR, _PROCESS_VISION_INFO
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Missing dependencies: pip install transformers qwen-vl-utils accelerate torch"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _QWEN_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            _QWEN_MODEL.to("cpu")
        _QWEN_PROCESSOR = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        _PROCESS_VISION_INFO = process_vision_info
        return _QWEN_MODEL, _QWEN_PROCESSOR, _PROCESS_VISION_INFO


def _upsert_link_json(link_path: Path, meeting_name: str, video_url: str) -> None:
    entries = []
    if link_path.exists():
        try:
            data = json.loads(link_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                entries = [e for e in data if isinstance(e, dict)]
        except json.JSONDecodeError:
            entries = []

    updated = False
    for entry in entries:
        if entry.get("meet") == meeting_name:
            entry["link"] = video_url
            updated = True
            break
    if not updated:
        entries.append({"meet": meeting_name, "link": video_url})

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_meeting_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    return value[:120] or "YouTube Meeting"


def _safe_file_stem(value: str) -> str:
    return re.sub(r"[^\w\-]+", "_", value).strip("_") or "meeting"


def _require_url(video_url: str) -> None:
    if not _youtube_video_id(video_url):
        raise ValueError("Only public YouTube URLs are supported")


def _youtube_video_id(video_url: str) -> str | None:
    from urllib.parse import parse_qs, urlparse

    try:
        parsed = urlparse(video_url)
    except Exception:
        return None

    if parsed.scheme != "https":
        return None

    host = parsed.hostname or ""
    if host == "youtu.be":
        return parsed.path.strip("/") or None

    if host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed.path.startswith("/live/"):
            parts = [p for p in parsed.path.split("/") if p]
            return parts[1] if len(parts) > 1 else None
        return parse_qs(parsed.query).get("v", [None])[0]

    return None


def _set(progress: ProgressCallback | None, stage: str, pct: int, message: str) -> None:
    if progress:
        progress(stage, pct, message)
