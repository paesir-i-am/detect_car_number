from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .pipeline import analyze_media
from .settings import Settings
from .detector import YoloDetector


settings = Settings()
detector = YoloDetector(settings)
app = FastAPI(title="detect_car_number v2", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


async def _prepare_source_from_url(url: str) -> tuple[Path, bool]:
    """URL 또는 로컬 경로를 받아 임시 파일로 저장하고 경로 반환."""
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            resp = await client.get(url)
            if resp.status_code >= 400:
                raise HTTPException(status_code=400, detail=f"URL 다운로드 실패: {resp.status_code}")
            suffix = Path(parsed.path).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(resp.content)
                return Path(tmp.name), True

    local_path = Path(url)
    if not local_path.exists():
        raise HTTPException(status_code=400, detail="파일 경로 또는 URL을 확인하세요.")
    return local_path, False


@app.post("/v1/live/customer-car")
async def detect_customer_car(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
):
    # file(업로드) 또는 url(내부 경로/외부 URL) 중 하나는 필수
    if not file and not url:
        raise HTTPException(status_code=400, detail="file 또는 url 중 하나는 필요합니다.")

    cleanup_paths: list[Path] = []

    if file:
        if not file.filename:
            raise HTTPException(status_code=400, detail="파일 이름이 없습니다.")
        suffix = Path(file.filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            src_path = Path(tmp.name)
            cleanup_paths.append(src_path)
    else:
        src_path, should_cleanup = await _prepare_source_from_url(url)  # type: ignore[arg-type]
        if should_cleanup:
            cleanup_paths.append(src_path)

    try:
        result = analyze_media(detector, src_path, settings)
    finally:
        for p in cleanup_paths:
            p.unlink(missing_ok=True)

    response = {
        "status": result.status,
        "car_number": result.car_number,
        "timestamp": result.timestamp,
        "frame_index": result.frame_index,
        "confidence": result.confidence,
        "processing_time_ms": round(result.processing_time_ms, 2),
        "result_image_path": str(result.result_image_path) if result.result_image_path else None,
        "detections_count": result.detections_count,
        "save_dir": str(result.save_dir) if result.save_dir else None,
        "message": result.message,
    }
    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("detect_car_number.main:app", host="0.0.0.0", port=8000, reload=True)
