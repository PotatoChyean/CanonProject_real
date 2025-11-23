# Cannon Project Backend API

FastAPI 기반 백엔드 서버 - YOLO + OCR + CNN 모델 통합

## 설치 및 실행

### 1. 가상환경 생성 및 활성화

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 모델 파일 준비

다음 파일들을 `backend/models/` 폴더에 배치하세요:

- `yolov8m.pt`: 학습된 YOLO 모델 파일 (또는 기본 yolov8m.pt 사용)
- `cnn_4class_conditional.pt`: 학습된 CNN 모델 파일
- `OCR_lang.csv`: OCR 언어별 텍스트 매칭 테이블

**OCR_lang.csv 형식:**
```csv
lang,term,group
ko,텍스트1,0
ko,텍스트2,1
en,text1,0
en,text2,1
```

- `lang`: 언어 코드 (ko, en, ja, ch_sim, ch_tra)
- `term`: 검색할 텍스트
- `group`: 그룹 번호 (0: 필수, 1: XOR 조건)

### 4. 서버 실행

```bash
python main.py
```

또는

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. API 문서 확인

브라우저에서 `http://localhost:8000/docs` 접속

## 모델 동작 방식

### 1. OCR 언어 탐지
- EasyOCR을 사용하여 여러 언어(ko, en, ja, ch_sim, ch_tra)로 텍스트 인식
- OCR_lang.csv의 group 0 텍스트가 모두 있고, group 1 텍스트 중 하나만 있으면 Pass

### 2. YOLO 객체 검출
- 다음 조건을 모두 만족해야 Pass:
  - `Btn_Home`과 `Btn_Stat` 필수 검출
  - `Btn_Back`과 `Btn_ID` 중 하나만 검출 (XOR)
  - `Monitor_Small` 또는 `Monitor_Big` 중 하나 이상 검출

### 3. CNN ROI 검증
- YOLO로 검출된 버튼 영역(Btn_Back, Btn_Home, Btn_ID, Btn_Stat)을 CNN으로 검증
- 모든 ROI가 0.5 이상의 확률을 가지면 Pass

### 4. 최종 판정
- OCR, YOLO, CNN 모두 Pass여야 최종 Pass

## API 엔드포인트

- `POST /api/analyze-image`: 단일 이미지 분석
- `POST /api/analyze-batch`: 여러 이미지 일괄 분석
- `POST /api/analyze-frame`: 실시간 프레임 분석
- `GET /api/statistics`: 통계 조회
- `GET /api/results`: 결과 목록 조회
- `GET /health`: 서버 상태 확인

## 응답 형식

```json
{
  "id": 1,
  "filename": "image.jpg",
  "status": "PASS" | "FAIL",
  "reason": "Fail 사유",
  "confidence": 85.5,
  "details": {
    "ocr_status": "Pass" | "Fail",
    "ocr_lang": "ko",
    "yolo_status": "Pass" | "Fail",
    "cnn_status": "Pass" | "Fail",
    "yolo_detections": [...],
    "ocr_results": [...],
    "cnn_results": [...],
    "detected_classes": [...]
  },
  "timestamp": "2024-01-01T12:00:00"
}
```
