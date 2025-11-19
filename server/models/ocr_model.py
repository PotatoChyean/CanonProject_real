"""
OCR 모델 로드 및 추론 (EasyOCR 사용)
"""

import numpy as np
from typing import Dict
import easyocr
import torch


class OCRModel:
    """OCR 모델 래퍼 클래스 (EasyOCR 사용)"""
    
    def __init__(self):
        """
        OCR 모델 초기화 (Flask 코드와 동일하게 여러 언어 지원)
        """
        self.readers = {}
        self.load_model()
    
    def load_model(self):
        """OCR 모델 로드 (각 언어별로 Reader 생성)"""
        try:
            gpu_available = torch.cuda.is_available()
            print(f"OCR 모델 로드 중... (GPU: {gpu_available})")
            
            # Flask 코드와 동일한 언어 설정
            self.readers = {
                "ko": easyocr.Reader(['ko'], gpu=gpu_available),
                "en": easyocr.Reader(['en'], gpu=gpu_available),
                "ja": easyocr.Reader(['ja'], gpu=gpu_available),
                "ch_sim": easyocr.Reader(['ch_sim', 'en'], gpu=gpu_available),
                "ch_tra": easyocr.Reader(['ch_tra', 'en'], gpu=gpu_available)
            }
            
            print("OCR 모델 로드 완료")
        
        except Exception as e:
            print(f"OCR 모델 로드 실패: {e}")
            self.readers = {}
    
    def read_text(self, image: np.ndarray, lang: str = "ko") -> Dict:
        """
        이미지에서 텍스트 추출
        
        Args:
            image: numpy array 형태의 이미지 (H, W, C 또는 H, W)
            lang: 언어 코드 ("ko", "en", "ja", "ch_sim", "ch_tra")
            
        Returns:
            {
                "text": "추출된 텍스트",
                "confidence": 0.0-1.0,
                "bboxes": [[x1, y1, x2, y2], ...]  # 각 단어/문자의 위치
            }
        """
        if lang not in self.readers or self.readers[lang] is None:
            return {
                "text": "",
                "confidence": 0.0,
                "bboxes": []
            }
        
        try:
            reader = self.readers[lang]
            results = reader.readtext(image)
            
            if not results:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "bboxes": []
                }
            
            text = " ".join([result[1] for result in results])
            confidence = sum([result[2] for result in results]) / len(results) if results else 0
            bboxes = [result[0] for result in results]
            
            return {
                "text": text,
                "confidence": confidence,
                "bboxes": bboxes
            }
        
        except Exception as e:
            print(f"OCR 추론 오류 ({lang}): {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "bboxes": []
            }

