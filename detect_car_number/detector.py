from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import (
    LOGGER,
    check_img_size,
    increment_path,
    non_max_suppression,
    scale_boxes,
)
from yolov5.utils.plots import save_one_box
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import cv2  # type: ignore

from .settings import Settings


@dataclass
class DetectedCrop:
    crop_path: Path
    image_path: Path
    frame_index: int
    confidence: float


@dataclass
class DetectionResult:
    save_dir: Path
    detections: List[DetectedCrop]
    total_frames: int
    processed_frames: int


class YoloDetector:
    def __init__(self, settings: Settings):
        # 모델 로딩은 앱 시작 시 1회 수행하여 추론 호출 시 지연을 최소화
        self.settings = settings
        self.device = select_device(settings.device)
        self.model = DetectMultiBackend(
            settings.weights_path,
            device=self.device,
            dnn=False,
            data=settings.data_yaml,
            fp16=False,
        )
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((640, 640), s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def detect(self, source: Path) -> DetectionResult:
        # 매 실행마다 exp 디렉터리를 새로 생성해 결과를 분리
        save_dir = increment_path(self.settings.result_save_dir / "exp", exist_ok=True)
        images_dir = save_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        detections: List[DetectedCrop] = []
        processed_frames = 0

        dataset = LoadImages(
            str(source),
            img_size=self.imgsz,
            stride=self.stride,
            auto=self.pt,
            vid_stride=max(1, self.settings.frame_stride),
        )

        for idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            if idx >= self.settings.max_frames:
                break

            # 입력을 float(0~1)로 변환하여 모델에 전달
            im_tensor = torch.from_numpy(im).to(self.model.device)
            im_tensor = im_tensor.half() if self.model.fp16 else im_tensor.float()
            im_tensor /= 255
            if len(im_tensor.shape) == 3:
                im_tensor = im_tensor[None]

            pred = self.model(im_tensor)
            pred = non_max_suppression(
                pred,
                self.settings.conf_threshold,
                self.settings.iou_threshold,
                classes=None,
                agnostic=False,
                max_det=1000,
            )

            frame_index = getattr(dataset, "frame", 0)
            im0 = im0s.copy()

            for det in pred:
                if len(det) == 0:
                    continue

                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    class_id = int(cls)
                    class_name = str(self.names.get(class_id, class_id))
                    crop_dir = save_dir / "crops" / class_name
                    crop_dir.mkdir(parents=True, exist_ok=True)

                    file_name = f"{Path(path).stem}_{frame_index:04}_{conf:.2f}.jpg"
                    crop_path = crop_dir / file_name
                    save_one_box(xyxy, im0, file=crop_path, BGR=True)  # 번호판 크롭 저장

                    image_path = images_dir / file_name
                    cv2.imwrite(str(image_path), im0)  # 전체 프레임 저장(디버그/로그용)

                    detections.append(
                        DetectedCrop(
                            crop_path=crop_path,
                            image_path=image_path,
                            frame_index=int(frame_index),
                            confidence=float(conf),
                        )
                    )

            processed_frames += 1
            if processed_frames >= self.settings.max_frames:
                break

        LOGGER.info(f"Detections: {len(detections)} saved to {save_dir}")
        return DetectionResult(
            save_dir=save_dir,
            detections=detections,
            total_frames=getattr(dataset, "frames", 0) if hasattr(dataset, "frames") else processed_frames,
            processed_frames=processed_frames,
        )
