# 모델 파일 설정 가이드

## 필요한 모델 파일

Flask 코드에서 사용하던 모델 파일들을 `backend/models/` 폴더에 배치하세요.

### 1. YOLO 모델
- **파일명**: `yolov8m.pt` (또는 학습된 커스텀 모델)
- **위치**: `backend/models/yolov8m.pt`
- **클래스**: ['Btn_Home', 'Btn_Back', 'Btn_ID', 'Btn_Stat', 'Monitor_Small', 'Monitor_Big', 'sticker']

### 2. CNN 모델
- **파일명**: `cnn_4class_conditional.pt`
- **위치**: `backend/models/cnn_4class_conditional.pt`
- **구조**: ConditionalEfficientNet (4개 클래스 조건부)

### 3. OCR 언어 테이블
- **파일명**: `OCR_lang.csv`
- **위치**: `backend/models/OCR_lang.csv`
- **형식**:
```csv
lang,term,group
ko,홈,0
ko,통계,0
ko,뒤로가기,1
ko,아이디,1
en,Home,0
en,Statistics,0
en,Back,1
en,ID,1
```

**컬럼 설명:**
- `lang`: 언어 코드 (ko, en, ja, ch_sim, ch_tra)
- `term`: 검색할 텍스트
- `group`: 
  - `0`: 필수 텍스트 (모두 있어야 함)
  - `1`: XOR 조건 텍스트 (하나만 있어야 함)

## 모델 파일 구조

```
backend/
├── models/
│   ├── yolov8m.pt
│   ├── cnn_4class_conditional.pt
│   └── OCR_lang.csv
├── main.py
└── ...
```

## 모델 파일이 없는 경우

모델 파일이 없어도 서버는 실행되지만, 기본 모델을 사용하거나 오류가 발생할 수 있습니다.

- **YOLO**: 기본 `yolov8m.pt` 다운로드 시도
- **CNN**: 모델이 없으면 CNN 검증을 건너뜀
- **OCR**: EasyOCR이 자동으로 모델을 다운로드

## 모델 경로 변경

모델 경로를 변경하려면 `backend/main.py`의 `startup_event()` 함수를 수정하세요:

```python
yolo_path = "your/custom/path/yolo.pt"
cnn_path = "your/custom/path/cnn.pt"
ocr_csv_path = "your/custom/path/ocr.csv"
```

