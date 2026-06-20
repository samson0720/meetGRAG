"""
CLI entry point for YouTube multimodal ingestion.

Example:
    python -m qa_Module.multimodal.run_youtube_analysis \
        "https://www.youtube.com/watch?v=..." \
        --meeting-name "IETF 125 IAB Open" \
        --auto-index
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _progress(stage: str, pct: int, message: str) -> None:
    print(f"[{pct:3d}%] {stage}: {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a YouTube video and write meetGRAG-compatible JSON."
    )
    parser.add_argument("url", help="Public YouTube URL to analyze.")
    parser.add_argument(
        "--meeting-name",
        default=None,
        help="Optional meeting name. Defaults to the YouTube title.",
    )
    parser.add_argument(
        "--database-dir",
        default=str(_ROOT / "database" / "meet_origin_data"),
        help="Directory where meeting JSON/transcript files are written.",
    )
    parser.add_argument(
        "--auto-index",
        action="store_true",
        help="Run document loading and GraphRAG indexing after analysis.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from qa_Module.multimodal.youtube_analyzer import analyze_youtube_to_meetgrag_json

    database_dir = Path(args.database_dir)
    result = analyze_youtube_to_meetgrag_json(
        video_url=args.url,
        meeting_name=args.meeting_name,
        database_dir=database_dir,
        progress=_progress,
    )

    print(f"Analysis complete: {result.total_slides} slide segments")
    print(f"JSON: {result.json_path}")
    print(f"Transcript: {result.transcript_path}")

    if args.auto_index:
        from qa_Module.graphrag.document_loader import run as run_document_loader
        from qa_Module.graphrag.indexer import run_indexing

        input_dir = _ROOT / "qa_Module" / "graphrag" / "input"
        output_dir = _ROOT / "qa_Module" / "graphrag" / "output"
        settings_path = _ROOT / "qa_Module" / "graphrag" / "settings.yaml"

        run_document_loader(database_dir, input_dir, verbose=args.verbose)
        stats = run_indexing(
            input_dir=input_dir,
            output_dir=output_dir,
            settings_path=settings_path,
            verbose=args.verbose,
        )
        print(f"Indexing complete: {stats}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
