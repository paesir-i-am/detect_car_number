from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from shutil import which
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent  # v2 루트 경로


def _find_default_tesseract_cmd() -> Path:
    """OS에 설치된 tesseract 실행 파일을 우선 검색한다. 환경변수 → 플랫폼 기본 경로 → which 순서."""
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd:
        return Path(env_cmd)

    system = platform.system()
    candidates = []
    if system == "Darwin":
        candidates.extend(["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"])
    elif system == "Linux":
        candidates.extend(["/usr/bin/tesseract", "/usr/local/bin/tesseract"])

    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)

    which_cmd = which("tesseract")
    if which_cmd:
        return Path(which_cmd)

    return Path("/usr/bin/tesseract")


def _find_default_tessdata_prefix() -> Path:
    """tessdata(언어 데이터) 위치를 찾는다. 로컬 kor/osd 우선, 없으면 시스템 경로를 순회."""
    env_prefix = os.getenv("TESSDATA_PREFIX")
    if env_prefix:
        return Path(env_prefix)

    local_tessdata = BASE_DIR / "model" / "tessdata"
    if (local_tessdata / "kor.traineddata").exists():
        return local_tessdata

    for candidate in (
        "/opt/homebrew/share/tessdata",
        "/usr/local/share/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tessdata",
    ):
        if Path(candidate).exists():
            return Path(candidate)

    return local_tessdata


@dataclass
class Settings:
    base_dir: Path = BASE_DIR  # v2 루트
    model_dir: Path = field(default_factory=lambda: BASE_DIR / "model")  # 모델/설정/언어 데이터 위치
    weights_path: Path = field(default_factory=lambda: (BASE_DIR / "model" / "yo5s_b32_e10.pt"))
    data_yaml: Path = field(default_factory=lambda: (BASE_DIR / "model" / "custom.yaml"))
    result_root: Path = field(default_factory=lambda: BASE_DIR / "outputs")
    result_save_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs" / "result_save")
    result_img_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs" / "result_img")
    tesseract_cmd: Path = field(default_factory=_find_default_tesseract_cmd)
    tessdata_prefix: Path = field(default_factory=_find_default_tessdata_prefix)
    max_frames: int = 30  # 처리할 최대 프레임 수
    frame_stride: int = 1  # 프레임 샘플링 간격
    conf_threshold: float = 0.25  # YOLO confidence 임계값
    iou_threshold: float = 0.45  # IOU 임계값
    device: str = ""  # GPU 선택 시 "0" 등
    cleanup_intermediate: bool = True  # True면 exp(크롭/프레임) 폴더를 최종 이미지 복사 후 삭제

    def ensure_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.result_save_dir.mkdir(parents=True, exist_ok=True)
        self.result_img_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = Settings()
