import os, time, cv2, torch, easyocr
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request, send_from_directory
from torchvision import transforms
from ultralytics import YOLO
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------
# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
ALL_DIR = os.path.join(STATIC_DIR, "all")
RESULT_DIR = os.path.join(STATIC_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

OCR_CSV = os.path.join(BASE_DIR, "OCR_lang.csv")
MODEL_PATH = os.path.join(BASE_DIR, "./models/cnn_4class_conditional.pt")
YOLO_PATH = os.path.join(BASE_DIR, "./models.yolov8m.pt")

DEVICE = "cpu"

# -----------------------------------------------------
# OCR reader (단일 인스턴스, 모든 언어 지원)
readers = {
    "ko": easyocr.Reader(['ko'], gpu=False),
    "en": easyocr.Reader(['en'], gpu=False),
    "ja": easyocr.Reader(['ja'], gpu=False),
    "ch_sim": easyocr.Reader(['ch_sim', 'en'], gpu=False),
    "ch_tra": easyocr.Reader(['ch_tra', 'en'], gpu=False)
}

OCR_TABLE = pd.read_csv(OCR_CSV)

# -----------------------------------------------------
# YOLO 모델
yolo_model = YOLO(YOLO_PATH)
CLASS_NAMES = ['Btn_Home', 'Btn_Back', 'Btn_ID', 'Btn_Stat', 'Monitor_Small', 'Monitor_Big', 'sticker']

# -----------------------------------------------------
# CNN 모델 정의
class ConditionalEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from torchvision import models
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

cnn_model = ConditionalEfficientNet(num_classes=4).to(DEVICE)
cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -----------------------------------------------------
# OCR 언어 탐지
def detect_language(img):
    img_np = np.array(img)
    for lang, reader in readers.items():
        results = reader.readtext(img_np, detail=1, text_threshold=0.4, low_text=0.3, contrast_ths=0.05)
        if not results:
            continue
        recognized = " ".join([r[1] for r in results])
        subset = OCR_TABLE[OCR_TABLE['lang'] == lang]
        matched = subset[subset['term'].apply(lambda t: t in recognized)]
        if matched.empty:
            continue

        # group 판정
        has_group0 = all(subset[subset['group'] == 0]['term'].apply(lambda t: t in recognized))
        xor_terms = subset[subset['group'] == 1]['term'].tolist()
        has_xor = any(t in recognized for t in xor_terms)
        xor_multi = sum(t in recognized for t in xor_terms) > 1
        if has_group0 and has_xor and not xor_multi:
            return lang, "Pass", results
        else:
            return lang, "Fail", results
    return "Nonlingual", "Pass", []

# -----------------------------------------------------
# Flask app
app = Flask(__name__, static_folder=STATIC_DIR)

@app.route('/')
def index():
    imgs = sorted([f for f in os.listdir(ALL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    idx = int(request.args.get('idx', 0) or 0)
    if not imgs:
        return "no images found"

    img_name = imgs[idx % len(imgs)]
    img_path = os.path.join(ALL_DIR, img_name)
    pil_img = Image.open(img_path).convert("L")
    start = time.time()

    # 1. OCR
    lang, ocr_status, ocr_boxes = detect_language(pil_img)

    # 2. YOLO
    results = yolo_model.predict(source=img_path, conf=0.5, imgsz=800, device=DEVICE, verbose=False)
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    detected = [CLASS_NAMES[c] for c in cls_ids]

    # 3. CNN (ROI 예측 + 시각화)
    conds = ['Btn_Back', 'Btn_Home', 'Btn_ID', 'Btn_Stat']
    roi_pass = []
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_GRAY2BGR)

    for (x1, y1, x2, y2), cls_id in zip(boxes, cls_ids):
        cls_name = CLASS_NAMES[cls_id]
        if cls_name not in conds:
            continue
        crop = pil_img.crop((x1, y1, x2, y2))
        x = transform(crop).unsqueeze(0).to(DEVICE)
        cond_onehot = torch.zeros(len(conds)).to(DEVICE)
        cond_onehot[conds.index(cls_name)] = 1
        pred, _ = cnn_model(x, cond_onehot.unsqueeze(0))
        prob = pred.item()
        roi_pass.append(prob >= 0.5)
        color = (0, 255, 0) if prob >= 0.5 else (0, 0, 255)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_cv, f"{cls_name}:{prob:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 4. OCR 박스 시각화
    for (bbox, text, conf) in ocr_boxes:
        pts = np.array(bbox).astype(int)
        cv2.polylines(img_cv, [pts], True, (255, 255, 0), 1)
        cv2.putText(img_cv, text, (pts[0][0], pts[0][1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # 5. 개별 판정
    yolo_ok = ('Btn_Home' in detected and 'Btn_Stat' in detected) and \
              (('Btn_Back' in detected) ^ ('Btn_ID' in detected)) and \
              (('Monitor_Small' in detected) or ('Monitor_Big' in detected))
    cnn_ok = all(roi_pass) if roi_pass else False
    overall = "Pass" if (ocr_status == "Pass" and yolo_ok and cnn_ok) else "Fail"

    elapsed = (time.time() - start) * 1000
    fps = 1000.0 / elapsed if elapsed > 0 else 0

    # 6. 결과 이미지에 텍스트 요약 추가
    cv2.putText(img_cv, f"OCR:{lang}({ocr_status}) YOLO:{'Pass' if yolo_ok else 'Fail'} CNN:{'Pass' if cnn_ok else 'Fail'}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_cv, f"FINAL: {overall} | Time: {elapsed:.1f} ms ({fps:.1f} FPS)",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 7. 저장
    result_path = os.path.join(RESULT_DIR, img_name)
    cv2.imwrite(result_path, img_cv)

    # 8. 템플릿 반환
    result = {
        "ocr_status": ocr_status,
        "ocr_lang": lang,
        "yolo_status": "Pass" if yolo_ok else "Fail",
        "cnn_status": "Pass" if cnn_ok else "Fail",
        "final_status": overall,
        "time_ms": f"{elapsed:.1f}",
        "fps": f"{fps:.1f}"
    }

    prev_idx = (idx - 1) % len(imgs)
    next_idx = (idx + 1) % len(imgs)

    return render_template(
        "index.html",
        img_name=img_name,
        idx=idx,
        total=len(imgs),
        result=result
    )


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(STATIC_DIR, path)

if __name__ == '__main__':
    app.run(debug=True)
