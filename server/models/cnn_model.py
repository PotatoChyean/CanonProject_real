"""
CNN 모델 (ConditionalEfficientNet) 로드 및 추론
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from typing import Dict, Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ConditionalEfficientNet(nn.Module):
    """조건부 EfficientNet 모델"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(1, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        self.backbone.features[0][0] = new_conv
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.embed = nn.Linear(num_classes, in_features)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))

    def forward(self, x, cond):
        feat = self.backbone(x)
        cond_embed = self.embed(cond)
        fused = feat + cond_embed
        out = self.head(fused)
        return torch.sigmoid(out), feat


class CNNModel:
    """CNN 모델 래퍼 클래스"""
    
    def __init__(self, model_path: str = "models/cnn_4class_conditional.pt", num_classes: int = 4):
        """
        CNN 모델 초기화
        
        Args:
            model_path: 학습된 CNN 모델 경로
            num_classes: 조건 클래스 수
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = None
        # Pillow 10.0.0 이상에서 Image.ANTIALIAS는 Image.Resampling.LANCZOS로 변경됨
        try:
            # 최신 Pillow 버전
            resampling = Image.Resampling.LANCZOS
        except AttributeError:
            # 구버전 Pillow 호환 (Pillow < 10.0.0)
            resampling = Image.LANCZOS
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=resampling),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.conditions = ['Btn_Back', 'Btn_Home', 'Btn_ID', 'Btn_Stat']
        self.load_model()
    
    def load_model(self):
        """모델 로드"""
        try:
            self.model = ConditionalEfficientNet(self.num_classes).to(DEVICE)
            self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
            self.model.eval()
            print(f"CNN 모델 로드 완료: {self.model_path}")
        except Exception as e:
            print(f"CNN 모델 로드 실패: {e}")
            print("경로를 확인하거나 모델 파일이 존재하는지 확인하세요.")
            self.model = None
    
    def predict_roi(self, image: Image.Image, condition: str) -> Tuple[float, bool]:
        """
        ROI 이미지에 대한 예측 수행
        
        Args:
            image: PIL Image (그레이스케일)
            condition: 조건 클래스 이름 ('Btn_Back', 'Btn_Home', 'Btn_ID', 'Btn_Stat')
            
        Returns:
            (probability, is_pass): 확률값과 Pass 여부 (0.5 이상이면 Pass)
        """
        if self.model is None:
            return 0.0, False
        
        if condition not in self.conditions:
            return 0.0, False
        
        try:
            # 이미지 전처리
            x = self.transform(image).unsqueeze(0).to(DEVICE)
            
            # 조건 one-hot 인코딩
            cond_onehot = torch.zeros(len(self.conditions)).to(DEVICE)
            cond_onehot[self.conditions.index(condition)] = 1
            
            # 추론
            with torch.no_grad():
                pred, _ = self.model(x, cond_onehot.unsqueeze(0))
                prob = pred.item()
            
            is_pass = prob >= 0.5
            return prob, is_pass
        
        except Exception as e:
            print(f"CNN 예측 오류: {e}")
            return 0.0, False

