"""
데이터베이스 연동
결과 저장 및 조회
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import json

DB_PATH = "results.db"


def get_connection():
    """데이터베이스 연결"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """데이터베이스 초기화 (테이블 생성)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            confidence REAL,
            details TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()


def save_result(
    filename: str,
    status: str,
    reason: Optional[str] = None,
    confidence: float = 0.0,
    details: Optional[Dict] = None
) -> Dict:
    """
    분석 결과 저장
    
    Returns:
        저장된 결과 딕셔너리
    """
    init_db()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    details_json = json.dumps(details) if details else None
    
    cursor.execute("""
        INSERT INTO analysis_results (filename, status, reason, confidence, details, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, status, reason, confidence, details_json, timestamp))
    
    result_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "id": result_id,
        "filename": filename,
        "status": status,
        "reason": reason,
        "confidence": confidence,
        "details": details,
        "timestamp": timestamp
    }


def get_results(
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict]:
    """
    분석 결과 조회
    
    Args:
        status: 필터링할 상태 ("PASS", "FAIL")
        start_date: 시작일 (ISO 형식)
        end_date: 종료일 (ISO 형식)
        limit: 최대 조회 개수
        offset: 시작 위치
    """
    init_db()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM analysis_results WHERE 1=1"
    params = []
    
    if status:
        query += " AND status = ?"
        params.append(status)
    
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
        
    if end_date:
        # 종료일은 해당 날짜의 끝까지 포함하도록 처리
        query += " AND timestamp <= ?"
        params.append(end_date + "T23:59:59")

    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, tuple(params))
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "filename": row["filename"],
            "status": row["status"],
            "reason": row["reason"],
            "confidence": row["confidence"],
            "details": json.loads(row["details"]) if row["details"] else {},
            "timestamp": row["timestamp"]
        })
    
    return results


def get_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """
    통계 조회
    
    Returns:
        {
            "total": 총 개수,
            "pass": Pass 개수,
            "fail": Fail 개수,
            "pass_rate": Pass 비율,
            "fail_reasons": {reason: count, ...}
        }
    """
    init_db()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # 날짜 필터 조건
    date_filter = ""
    params = []
    if start_date:
        date_filter += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        date_filter += " AND timestamp <= ?"
        params.append(end_date)
    
    # 전체 통계
    cursor.execute(f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END) as pass_count,
            SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END) as fail_count
        FROM analysis_results
        WHERE 1=1 {date_filter}
    """, params)
    
    stats_row = cursor.fetchone()
    total = stats_row["total"] or 0
    pass_count = stats_row["pass_count"] or 0
    fail_count = stats_row["fail_count"] or 0
    pass_rate = (pass_count / total * 100) if total > 0 else 0
    
    # Fail 사유별 통계
    cursor.execute(f"""
        SELECT reason, COUNT(*) as count
        FROM analysis_results
        WHERE status = 'FAIL' AND reason IS NOT NULL {date_filter}
        GROUP BY reason
        ORDER BY count DESC
    """, params)
    
    fail_reasons = {}
    for row in cursor.fetchall():
        fail_reasons[row["reason"]] = row["count"]
    
    conn.close()
    
    return {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "pass_rate": round(pass_rate, 2),
        "fail_reasons": fail_reasons
    }

