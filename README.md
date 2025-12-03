# detect_car_number (FastAPI, YOLOv5 + Tesseract)

## 요구 사항
- Python 3.12 이상
- Tesseract 바이너리(OS 설치 필요)  
  - macOS(Homebrew): `/opt/homebrew/bin/tesseract` 자동 검색
  - 환경변수로 지정 가능: `TESSERACT_CMD`, `TESSDATA_PREFIX`
- 모델/데이터: `model/yo5s_b32_e10.pt`, `model/custom.yaml`, `model/tessdata/kor.traineddata`

## 설치 (uv 사용 예시)
```bash
# 가상환경 생성
uv venv .venv
source .venv/bin/activate  # zsh/bash 기준

# 의존성 설치 (루트에서 실행)
uv pip install -r requirements.txt
```

## 실행
```bash
uv run uvicorn detect_car_number.main:app --reload
```
- 헬스 체크: `GET /health`

## 로컬 스모크 테스트 (샘플 영상)
- 사전 준비: macOS의 경우 `brew install tesseract`로 바이너리 설치(이미 있으면 생략)
- 의존성 설치: 위 "설치" 절차로 가상환경 생성 후 `uv pip install -r requirements.txt`
- 서버 실행: 위 "실행" 절차로 FastAPI 서버 기동
- 새 터미널에서 샘플 영상 호출:
  ```bash
  curl -F "url=/Users/paesir/Desktop/git/detect_car_number/gate_60f_01_day.MOV" \
       http://localhost:8000/v1/live/customer-car
  ```
  또는 업로드 방식:
  ```bash
  curl -F "file=@gate_60f_01_day.MOV" http://localhost:8000/v1/live/customer-car
  ```
- 결과 확인: 응답 JSON의 `status`/`car_number` 확인, 최종 이미지는 `outputs/result_img/`에 저장됨(`cleanup_intermediate=False`로 변경 시 `outputs/result_save/exp`에 중간 산출물 보존)

## 프로젝트 구조 (최상위)
- `detect_car_number/` 패키지: settings/detector/ocr/pipeline/main
- `model/`: `yo5s_b32_e10.pt`, `custom.yaml`, `tessdata/kor/osd.traineddata`
- `yolov5/`: YOLOv5 소스(로컬 editable 설치)
- `outputs/`: `result_img/`(최종), `result_save/`(중간 산출물, 기본 삭제)
- 샘플 영상: `gate_60f_01_day.MOV`
- `requirements.txt`: FastAPI/Tesseract/YOLO 의존성

## API 요청 (프론트 참고)
- 엔드포인트: `POST /v1/live/customer-car`
- Content-Type: `multipart/form-data`
- 입력 파라미터(둘 중 하나 필수)
  - `file`: 업로드할 이미지/영상
  - `url`: VPC 내부 경로 또는 http/https 링크
- cURL 예시
  - 업로드: `curl -F "file=@gate_60f_01_day.MOV" http://localhost:8000/v1/live/customer-car`
  - 경로/URL: `curl -F "url=/absolute/path/to/file.mp4" http://localhost:8000/v1/live/customer-car`
- Fetch 예시
  ```js
  // 파일 업로드
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  const res = await fetch('http://localhost:8000/v1/live/customer-car', { method: 'POST', body: fd });
  console.log(await res.json());

  // 로컬/VPC 경로 또는 http(s) 전달
  const fd2 = new FormData();
  fd2.append('url', '/path/in/vpc/video.mp4');
  const res2 = await fetch('http://localhost:8000/v1/live/customer-car', { method: 'POST', body: fd2 });
  console.log(await res2.json());
  ```
- 응답 JSON 필드
  - `status`: `ok` / `no_detections` / `no_ocr`
  - `car_number`: 인식된 번호판 문자열 또는 null
  - `timestamp`: 처리 시각(UTC ISO)
  - `frame_index`: 선택된 프레임 번호
  - `confidence`: 탐지 신뢰도
  - `processing_time_ms`: 처리 시간(ms)
  - `result_image_path`: 최종 이미지 파일 경로
  - `detections_count`: 탐지된 크롭 개수
  - `save_dir`: 중간 저장 디렉터리 경로(기본 `cleanup_intermediate=True`라 실행 후 비어 있을 수 있음)
  - `message`: 에러/보조 메시지

## 테스트 메모
- 로컬 샘플: `url=/absolute/path/to/gate_60f_01_day.MOV`로 호출해 스모크 테스트 가능
- 기본 옵션: `cleanup_intermediate=True`라 결과 이미지만 남고 `outputs/result_save/exp`는 삭제됨
- Tesseract: OS 설치 바이너리 자동 검색. 한국어 tessdata는 `model/tessdata`에 포함됨.

## 동작 파이프라인
1) 입력 확보: 업로드 `file` 또는 `url`(로컬/VPC/http) → 임시 파일로 저장
2) YOLOv5 추론 (`detect_car_number.detector.YoloDetector`)
   - 가중치 `model/yo5s_b32_e10.pt`, 데이터 `model/custom.yaml`
   - `max_frames`, `frame_stride` 옵션 적용
   - 크롭/전체 프레임 `outputs/result_save/exp/`에 저장(기본적으로 최종 후 삭제)
3) OCR (`detect_car_number.ocr_service.tesseract_ocr`)
   - Tesseract 경로 자동 검색 또는 `TESSERACT_CMD`
   - tessdata: 기본 `model/tessdata`(kor/osd 포함) → 시스템 경로 fallback
   - 길이 ≥ 7인 문자열만 카운트, 최빈값 선택
4) 결과 정리 (`detect_car_number.pipeline.analyze_media`)
   - 선택 프레임을 `outputs/result_img/<plate>.jpg`로 복사
   - `cleanup_intermediate=True` 기본값으로 중간 산출물(exp) 삭제

## 주요 파일
- `detect_car_number/settings.py`: 경로/모델/테서랙트/프레임 옵션 설정
- `detect_car_number/detector.py`: YOLOv5 로더 및 추론 래퍼
- `detect_car_number/ocr_service.py`: 번호판 크롭 OCR
- `detect_car_number/pipeline.py`: 탐지→OCR→최종 선택→이미지 저장
- `detect_car_number/main.py`: FastAPI 엔드포인트 (file 또는 url 입력)
- `requirements.txt`: FastAPI/Tesseract/YOLO 의존성 + 로컬 `./yolov5` 설치
- `model/`: `yo5s_b32_e10.pt`, `custom.yaml`, `tessdata/kor/osd.traineddata`
- `outputs/`: `result_img/` 최종 결과, `result_save/` 중간 산출물(기본 삭제)

## 설정 팁
- 중간 산출물 보존: `Settings(cleanup_intermediate=False)`
- GPU 지정: `Settings(device="0")` 등
- Tesseract 경로/데이터 커스터마이즈: `TESSERACT_CMD`, `TESSDATA_PREFIX` 환경변수 사용
