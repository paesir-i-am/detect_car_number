from __future__ import annotations

import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .detector import DetectionResult, DetectedCrop, YoloDetector
from .ocr_service import tesseract_ocr
from .settings import Settings


@dataclass
class PipelineResult:
    status: str
    car_number: Optional[str]
    message: Optional[str]
    result_image_path: Optional[Path]
    processing_time_ms: float
    detections_count: int
    timestamp: str
    frame_index: Optional[int] = None
    confidence: Optional[float] = None
    save_dir: Optional[Path] = None


def analyze_media(detector: YoloDetector, media_path: Path, settings: Settings) -> PipelineResult:
    """YOLO 탐지 → OCR → 최빈값 선택 → 결과 이미지 복사."""
    settings.ensure_dirs()

    start = time.perf_counter()
    detect_result: DetectionResult = detector.detect(media_path)

    if not detect_result.detections:
        return PipelineResult(
            status="no_detections",
            car_number=None,
            message="번호판 영역을 찾지 못했습니다.",
            result_image_path=None,
            processing_time_ms=(time.perf_counter() - start) * 1000,
            detections_count=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            save_dir=detect_result.save_dir,
        )

    plate_counter: Counter[str] = Counter()
    best_detection_map: dict[str, DetectedCrop] = {}

    for det in detect_result.detections:
        plate_text = tesseract_ocr(det.crop_path, settings.tesseract_cmd, settings.tessdata_prefix)
        if len(plate_text) < 7:
            continue
        plate_counter[plate_text] += 1
        current_best = best_detection_map.get(plate_text)
        if current_best is None or det.confidence > current_best.confidence:
            best_detection_map[plate_text] = det

    if not plate_counter:
        return PipelineResult(
            status="no_ocr",
            car_number=None,
            message="OCR 결과가 없습니다.",
            result_image_path=None,
            processing_time_ms=(time.perf_counter() - start) * 1000,
            detections_count=len(detect_result.detections),
            timestamp=datetime.now(timezone.utc).isoformat(),
            save_dir=detect_result.save_dir,
        )

    winner, _ = plate_counter.most_common(1)[0]
    best_det = best_detection_map[winner]

    result_img_path = settings.result_img_dir / f"{winner}.jpg"
    result_img_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_det.image_path, result_img_path)

    if settings.cleanup_intermediate and detect_result.save_dir.exists():
        shutil.rmtree(detect_result.save_dir, ignore_errors=True)

    return PipelineResult(
        status="ok",
        car_number=winner,
        message=None,
        result_image_path=result_img_path,
        processing_time_ms=(time.perf_counter() - start) * 1000,
        detections_count=len(detect_result.detections),
        timestamp=datetime.now(timezone.utc).isoformat(),
        frame_index=best_det.frame_index,
        confidence=best_det.confidence,
        save_dir=detect_result.save_dir,
    )
