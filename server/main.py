"""
FastAPI 백엔드 서버
YOLO + OCR 모델을 사용한 이미지 분석 API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional
import uvicorn
from datetime import datetime
import numpy as np
from PIL import Image
import io
import csv

from models.inference import analyze_image, analyze_frame, initialize_models
import os
from database.db import save_result, get_statistics, get_results

import os
# PyTorch DLL 경로 (사용자님 경로에 맞게 수정 필요)
torch_dll_path = r"C:\Users\user\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\lib"

app = FastAPI(title="Cannon Project API", version="1.0.0")

# 서버 시작 시 모델 초기화
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    print("모델 초기화 중...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 모델 경로 설정 (Flask 코드와 동일한 구조)
    yolo_path = os.path.join(BASE_DIR, "models", "yolov8m.pt")
    cnn_path = os.path.join(BASE_DIR, "models", "cnn_4class_conditional.pt")
    ocr_csv_path = os.path.join(BASE_DIR, "models", "OCR_lang.csv")
    
    # 경로가 없으면 상대 경로로 시도
    if not os.path.exists(yolo_path):
        yolo_path = "models/yolov8m.pt"
    if not os.path.exists(cnn_path):
        cnn_path = "models/cnn_4class_conditional.pt"
    if not os.path.exists(ocr_csv_path):
        ocr_csv_path = "models/OCR_lang.csv"
    
    initialize_models(
        yolo_path=yolo_path,
        cnn_path=cnn_path,
        ocr_csv_path=ocr_csv_path
    )
    print("모델 초기화 완료")

# CORS 설정 (Next.js 프론트엔드와 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    이미지 파일을 분석하여 Pass/Fail 결과 반환
    """
    try:
        # 이미지 파일 읽기
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        
        try:
            image = Image.open(io.BytesIO(contents))
            # 이미지를 RGB로 변환 (RGBA나 다른 형식 대응)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 파일 형식 오류: {str(e)}")
        
        image_array = np.array(image)
        
        # 모델 추론 실행
        result = analyze_image(image_array)
        
        # 결과 저장
        saved_result = save_result(
            filename=file.filename,
            status=result["status"],
            reason=result.get("reason"),
            confidence=result.get("confidence", 0),
            details=result.get("details", {})
        )
        
        return JSONResponse(content={
            "id": saved_result["id"],
            "filename": file.filename,
            "status": result["status"],
            "reason": result.get("reason"),
            "confidence": result.get("confidence", 0),
            "details": result.get("details", {}),
            "timestamp": saved_result["timestamp"]
        })
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"분석 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # 서버 로그에 출력
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")


@app.post("/api/analyze-batch")
async def analyze_batch_endpoint(files: List[UploadFile] = File(...)):
    """
    여러 이미지 파일을 일괄 분석
    """
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            if not contents:
                results.append({
                    "filename": file.filename,
                    "status": "ERROR",
                    "reason": "빈 파일입니다.",
                    "confidence": 0
                })
                continue
            
            try:
                image = Image.open(io.BytesIO(contents))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "ERROR",
                    "reason": f"이미지 파일 형식 오류: {str(e)}",
                    "confidence": 0
                })
                continue
            
            image_array = np.array(image)
            result = analyze_image(image_array)
            
            saved_result = save_result(
                filename=file.filename,
                status=result["status"],
                reason=result.get("reason"),
                confidence=result.get("confidence", 0),
                details=result.get("details", {})
            )
            
            results.append({
                "id": saved_result["id"],
                "filename": file.filename,
                "status": result["status"],
                "reason": result.get("reason"),
                "confidence": result.get("confidence", 0),
                "details": result.get("details", {}),
                "timestamp": saved_result["timestamp"]
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "ERROR",
                "reason": f"처리 실패: {str(e)}",
                "confidence": 0
            })
    
    return JSONResponse(content={"results": results})


@app.post("/api/analyze-frame")
async def analyze_frame_endpoint(file: UploadFile = File(...)):
    """
    실시간 카메라 프레임 분석
    """
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        
        try:
            image = Image.open(io.BytesIO(contents))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 파일 형식 오류: {str(e)}")
        
        image_array = np.array(image)
        
        result = analyze_frame(image_array)
        
        # 실시간 분석은 저장하지 않거나 별도 처리
        return JSONResponse(content={
            "status": result["status"],
            "reason": result.get("reason"),
            "confidence": result.get("confidence", 0),
            "details": result.get("details", {})
        })
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"프레임 분석 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # 서버 로그에 출력
        raise HTTPException(status_code=500, detail=f"프레임 분석 중 오류 발생: {str(e)}")


@app.get("/api/report")
async def get_report_endpoint(
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    분석 결과를 CSV 리포트로 다운로드
    """
    results = get_results(
        status=status, 
        start_date=start_date, 
        end_date=end_date,
        limit=10000  # 리포트는 많은 데이터를 포함할 수 있도록 limit를 크게 설정
    )

    output = io.StringIO()
    writer = csv.writer(output)

    # CSV 헤더 작성
    header = [
        "ID", "Filename", "Timestamp", "Status", "Reason", "Confidence", 
        "OCR Language", "OCR Status", "YOLO Status", "CNN Status"
    ]
    writer.writerow(header)

    # CSV 데이터 행 작성
    for result in results:
        details = result.get("details", {})
        row = [
            result.get("id"),
            result.get("filename"),
            result.get("timestamp"),
            result.get("status"),
            result.get("reason"),
            result.get("confidence"),
            details.get("ocr_lang"),
            details.get("ocr_status"),
            details.get("yolo_status"),
            details.get("cnn_status")
        ]
        writer.writerow(row)

    output.seek(0)
    
    report_filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={report_filename}"}
    )


@app.get("/api/statistics")
async def get_statistics_endpoint(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    분석 결과 통계 조회
    """
    try:
        stats = get_statistics(start_date, end_date)
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류 발생: {str(e)}")


@app.get("/api/results")
async def get_results_endpoint(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    분석 결과 목록 조회
    """
    try:
        results = get_results(status=status, limit=limit, offset=offset)
        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 조회 중 오류 발생: {str(e)}")


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

