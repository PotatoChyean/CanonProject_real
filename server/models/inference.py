"""
모델 추론 통합 로직 (수정 및 통합 버전)
YOLO, ConditionalViT (CNN) 모델을 연결하여 7단계 규칙 기반 Pass/Fail 및 제품 모델 분류
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from PIL import Image
from collections import Counter #! ===== 수정/추가 =====
from ultralytics import YOLO
from torchvision import transforms
from transformers import ViTModel
import io
import os
import traceback

# 외부 모듈 임포트 
from models.yolo_model import YOLOModel
from models.cnn_model import CNNModel

from .cnn_model import ConditionalViT


# ============================================================
# 제품 스펙테이블 및 레이블 #! ===== 수정/추가 =====
# ============================================================
PRODUCT_SPEC = {
    "FM2-V160-000": {"button": "ID",   "lang": "CN"},
    "FM2-V161-000": {"button": "STAT", "lang": None},
    "FM2-V162-000": {"button": "STAT", "lang": "EN"},
    "FM2-V163-000": {"button": "STAT", "lang": "CN"},
    "FM2-V164-000": {"button": "STAT", "lang": "KR"},
    "FM2-V165-000": {"button": "STAT", "lang": "TW"},
    "FM2-V166-000": {"button": "ID",   "lang": "EN"},
    "FM2-V167-000": {"button": "STAT", "lang": "JP"},
}

LANG_LABEL = ["CN", "EN", "JP", "KR", "TW"]

# YOLO 클래스 이름 (이전 규칙에 맞게 재정의)
# 0: Home, 1: Back, 2: ID, 3: Stat, 4: Monitor, 5: Text
CLASS_NAMES = ['Home', 'Back', 'ID', 'Stat', 'Monitor', 'Text', 'Monitor_Small', 'Monitor_Big', 'sticker']
CLASS_MAP = { 
    0: 'Home', 1: 'Back', 2: 'ID', 3: 'Stat', 4: 'Monitor', 5: 'Text', 
    6: 'Monitor_Small', 7: 'Monitor_Big', 8: 'sticker'
}

# 전역 모델 인스턴스
yolo_model = None
cnn_model = None
transform = None
DEVICE = "cpu"

# ============================================================
# 제품 모델 자동 분류 함수 #! ===== 수정/추가 =====
# ============================================================
def classify_model(found_back, found_id, text_langs):

    # (1) 텍스트 언어 결정 (N=0 또는 N>=3일 때만 호출된다고 가정)
    if len(text_langs) == 0:
        lang = None
    else:
        # N >= 3인 경우 다수결 (majority)
        lang = Counter(text_langs).most_common(1)[0][0] 

    # (2) Back/ID 결정
    if found_back and (not found_id):
        btn_type = "STAT"  # Back → STAT 모델군
    elif found_id and (not found_back):
        btn_type = "ID"
    else:
        # 이 에러는 analyze_image의 yolo_xor_ok에서 이미 걸러지지만, 명시적으로 반환
        return None, "Back/ID Mismatch (XOR Fail)" 

    # (3) 후보 제품 찾기
    candidates = []
    for name, spec in PRODUCT_SPEC.items():
        if spec["lang"] == lang and spec["button"] == btn_type:
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0], None # 성공
    elif len(candidates) > 1:
        return None, "AmbiguousModel" # 실패
    else:
        return None, "UnknownModel" # 실패

# ============================================================
# NumPy 타입 변환 함수 (기존 유지)
# ============================================================
def convert_numpy_types(data):
    """
    분석 결과에 포함된 NumPy 타입을 Python 기본 타입으로 재귀적으로 변환
    """
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [convert_numpy_types(i) for i in data]
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data

# ============================================================
# 모델 초기화 함수 (수정) #! ===== 수정 =====
# ============================================================
def initialize_models(
    yolo_path: str = "models/YOLO.pt",
    cnn_path: str = "models/CNN_classifier.pt",
):
    """모델 초기화 (서버 시작 시 호출)"""
    global yolo_model, cnn_model, transform, DEVICE
    
    # YOLO 모델 초기화 (YOLOModel은 외부 모듈 가정)
    if yolo_model is None:
        yolo_model = YOLOModel(model_path=yolo_path)

    # CNN/Text 모델 초기화 (추가)
    if cnn_model is None:
        try:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {DEVICE}")
            
            cnn_model = ConditionalViT() 
            cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
            cnn_model.eval()
            cnn_model = cnn_model.to(DEVICE)

            # 이미지 변환 정의
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            print("CNN/Text 모델 로드 완료.")
        except Exception as e:
            print(f"CNN/Text 모델 로드 실패: {e}")
            cnn_model = None
            
    return yolo_model, cnn_model

# ============================================================
# 이미지 분석 메인 함수 (전면 수정) #! ===== 전면 수정 =====
# ============================================================
def analyze_image(image: np.ndarray) -> Dict:
    """
    이미지 분석 메인 함수: 7단계 복합 검사 파이프라인 수행
    """
    # 모델 초기화 확인
    if yolo_model is None or cnn_model is None or transform is None:
        initialize_models()
        if cnn_model is None:
             raise RuntimeError("CNN/Text 모델이 로드되지 않았습니다. 초기화 오류를 확인하세요.")
    
    try:
        # 이미지를 PIL Image로 변환 (RGB로 변환)
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = Image.fromarray(image).convert("RGB") 

        # 1. YOLO 객체 검출
        yolo_results = yolo_model.detect(image) 
        detected_classes_raw = [d["class"] for d in yolo_results.get("detections", [])]
        
        # --- 2. YOLO 결과 플래그 초기화 및 CNN 데이터 수집 ---
        found_home = False
        found_stat = False
        found_monitor = False
        found_back = False
        found_id = False

        cnn_results = []
        roi_pass_list = [] # 버튼 CNN PASS/FAIL 결과만 저장
        text_langs = []
        
        button_classes = ['Home', 'Back', 'ID', 'Stat', 'Btn_Home', 'Btn_Back', 'Btn_ID', 'Btn_Stat']
        
        for detection in yolo_results.get("detections", []):
            cls_name = detection["class"]
            bbox = detection["bbox"]
            crop_pil = pil_img.crop((bbox[0], bbox[1], bbox[2], bbox[3])) 
            
            # 플래그 설정
            if cls_name in ['Home', 'Btn_Home']: found_home = True    # 🚨 수정됨
            elif cls_name in ['Back', 'Btn_Back']: found_back = True  # 🚨 수정됨
            elif cls_name in ['ID', 'Btn_ID']: found_id = True        # 🚨 수정됨 (ID도 Btn_ID로 탐지될 수 있으므로 포함)
            elif cls_name in ['Stat', 'Btn_Stat']: found_stat = True  # 🚨 수정됨
            elif cls_name == 'Monitor': found_monitor = True # Monitor 클래스 가정
            elif cls_name in ['Monitor_Small', 'Monitor_Big']: found_monitor = True # 확장 Monitor 클래스 대응
            
            # --- 3. CNN 수행 (버튼 & 텍스트) ---
            
            if cls_name in button_classes:
                # CNN head_btn (cond = 0): Pass/Fail 분류
                cond = torch.tensor([0]).to(DEVICE)
                with torch.no_grad():
                    t = transform(crop_pil).unsqueeze(0).to(DEVICE)
                    out = cnn_model(t, cond)[0]
                
                prob_pass = torch.softmax(out[:2], dim=0)[0].item() # Pass 확률
                is_pass = (torch.argmax(out[:2]).item() == 0) # 0이 Pass
                
                roi_pass_list.append(is_pass) # 모든 탐지된 버튼에 대해 품질 체크
                
                cnn_results.append({
                    "class": cls_name,
                    "bbox": bbox,
                    "probability": round(prob_pass, 4), 
                    "status": "Pass" if is_pass else "Fail"
                })

            elif cls_name == 'Text':
                # CNN head_txt (cond = 1): 언어 분류
                cond = torch.tensor([1]).to(DEVICE)
                with torch.no_grad():
                    t = transform(crop_pil).unsqueeze(0).to(DEVICE)
                    out = cnn_model(t, cond)[0]
                
                lang_idx = torch.argmax(out).item()
                lang = LANG_LABEL[lang_idx]
                prob_lang = torch.softmax(out, dim=0)[lang_idx].item()
                
                text_langs.append(lang)
                
                cnn_results.append({
                    "class": cls_name,
                    "bbox": bbox,
                    "probability": round(prob_lang, 4), 
                    "status": "OK", 
                    "lang": lang
                })

        # --- 4. 7가지 규칙 기반 판정 시작 ---
        
        # Rule A: YOLO 존재 조건 (Home, Stat, Monitor)
        yolo_presence_ok = found_home and found_stat and found_monitor
        
        # Rule B: Back XOR ID 조건
        yolo_xor_ok = found_back ^ found_id
        
        # Rule C: 버튼 CNN PASS 조건 (모든 탐지된 버튼)
        cnn_ok = all(roi_pass_list) if roi_pass_list else False

        # Rule D: 텍스트 개수 조건 (N=0 또는 N>=3)
        text_count = len(text_langs)
        text_ok = (text_count == 0) or (text_count >= 3)

        # Rule E: 제품 모델 분류 성공
        detected_prod, model_err = classify_model(found_back, found_id, text_langs)
        model_ok = (detected_prod is not None)
        
        # --- 5. 최종 판정 ---
        final_pass = yolo_presence_ok and yolo_xor_ok and cnn_ok and text_ok and model_ok
        final_status = "PASS" if final_pass else "FAIL" 
        
        # --- 6. Fail 사유 수집 ---
        reasons = []
        if not yolo_presence_ok: reasons.append("필수 객체 미검출 (Home/Stat/Monitor)")
        if not yolo_xor_ok: reasons.append("Back/ID 조건 불만족 (XOR 실패)")
        if not cnn_ok: reasons.append("버튼 CNN 검증 실패")
        if not text_ok: reasons.append(f"텍스트 개수 불만족 (N={text_count})")
        
        if not model_ok: reasons.append(model_err) 

        reason = "; ".join(reasons) if reasons else None
        
        # --- 7. 신뢰도 계산 및 결과 구성 (기존 로직 활용) ---
        confidence_scores = []
        for detection in yolo_results.get("detections", []):
            confidence_scores.append(detection.get("confidence", 0) * 100)
        for cnn_result in cnn_results:
            confidence_scores.append(cnn_result.get("probability", 0) * 100)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        final_result = {
            "status": final_status,
            "reason": reason,
            "confidence": round(avg_confidence, 2),
            "details": {
                "product_model": detected_prod,
                "language": Counter(text_langs).most_common(1)[0][0] if text_langs else None,
                "yolo_status": "Pass" if (yolo_presence_ok and yolo_xor_ok) else "Fail",
                "cnn_status": "Pass" if cnn_ok else "Fail",
                "model_status": "Pass" if model_ok else "Fail",
                "text_count": text_count,
                "yolo_detections": yolo_results.get("detections", []),
                "cnn_results": cnn_results, 
                "detected_classes": detected_classes_raw
            }
        }
        return convert_numpy_types(final_result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_result = {
            "status": "FAIL",
            "reason": f"분석 중 오류 발생: {type(e).__name__} - {str(e)}",
            "confidence": 0,
            "details": {}
        }
        return convert_numpy_types(error_result)


def analyze_frame(image: np.ndarray) -> Dict:
    """
    실시간 프레임 분석 (analyze_image와 동일)
    """
    # analyze_image와 동일 로직을 사용하되, 추후 성능 최적화를 위해 분리됨
    return analyze_image(image)