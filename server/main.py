"""
FastAPI 백엔드 서버
YOLO + OCR 모델을 사용한 이미지 분석 API (완성본)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
from datetime import datetime
import numpy as np
from PIL import Image
import io
import os
import traceback # 디버깅을 위해 추가

# --- 1. 통합된 모델 및 DB 모듈 임포트 ---
# 모델 초기화 및 분석 로직 (models/inference.py 파일)
from models.inference import analyze_image, analyze_frame, initialize_models
# DB 저장 및 조회 로직 (database/db.py 파일 - MySQL 버전으로 대체될 예정)
from database.db import save_result, get_statistics, get_results 


# --- 2. FastAPI 앱 초기화 및 설정 ---
app = FastAPI(title="Cannon Project API", version="1.0.0")

# CORS 설정 (Next.js 프론트엔드와 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. 모델 초기화 이벤트 (서버 시작 시 1회 실행) ---
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드 및 DB 연결 준비"""
    print("모델 초기화 및 DB 연결 준비 중...")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 모델 경로 설정
    yolo_path = os.path.join(BASE_DIR, "models", "yolov8m.pt")
    cnn_path = os.path.join(BASE_DIR, "models", "cnn_4class_conditional.pt")
    ocr_csv_path = os.path.join(BASE_DIR, "models", "OCR_lang.csv")
    
    # 모델 초기화 (models/inference.py의 함수 호출)
    initialize_models(
        yolo_path=yolo_path,
        cnn_path=cnn_path,
        ocr_csv_path=ocr_csv_path
    )
    print("모델 초기화 완료")
    
    # DB 초기화 (MySQL 테이블 생성 등)
    try:
        from database.db import init_db 
        init_db() 
        print("DB 초기화 완료")
    except ImportError:
        # init_db 함수가 정의되지 않았을 경우 (DB 모듈이 아직 불완전할 경우)
        print("경고: database/db.py에 init_db 함수를 찾을 수 없습니다. 수동으로 DB 초기화 필요.")
    except Exception as e:
        print(f"DB 초기화 중 오류 발생: {e}")


# --- 4. API 엔드포인트 정의 ---

@app.post("/api/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """단일 이미지 파일을 분석하여 Pass/Fail 결과 반환"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        result = analyze_image(image_array)
        
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
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")


@app.post("/api/analyze-batch")
async def analyze_batch_endpoint(files: List[UploadFile] = File(...)):
    """여러 이미지 파일을 일괄 분석"""
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
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
    """실시간 카메라 프레임 분석"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # analyze_frame은 모델 추론 로직을 사용하지만 DB 저장은 하지 않음 (실시간 처리)
        result = analyze_frame(image_array)
        
        return JSONResponse(content={
            "status": result["status"],
            "reason": result.get("reason"),
            "confidence": result.get("confidence", 0),
            "details": result.get("details", {})
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"프레임 분석 중 오류 발생: {str(e)}")


@app.get("/api/statistics")
async def get_statistics_endpoint(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """분석 결과 통계 조회 (DB read)"""
    try:
        # database/db.py의 get_statistics 호출
        stats = get_statistics(start_date, end_date)
        return JSONResponse(content=stats)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류 발생: {str(e)}")


@app.get("/api/results")
async def get_results_endpoint(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """분석 결과 목록 조회 (DB read)"""
    try:
        # database/db.py의 get_results 호출
        results = get_results(status=status, limit=limit, offset=offset)
        return JSONResponse(content={"results": results})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"결과 조회 중 오류 발생: {str(e)}")


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# --- 5. 서버 실행 ---
if __name__ == "__main__":
    # 포트를 5000번으로 변경
    print("FastAPI 서버 시작: http://localhost:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)