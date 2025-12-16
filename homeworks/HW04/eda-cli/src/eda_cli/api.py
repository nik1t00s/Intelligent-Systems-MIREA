import time
from typing import Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from .core import (
    summarize_dataset,
    missing_table,
    compute_quality_flags,
    DatasetSummary,
)

app = FastAPI(
    title="EDA Quality API",
    description="HTTP API for dataset quality analysis based on eda-cli",
    version="0.1.0",
)

class QualityRequest(BaseModel):
    Dict[str, Any]
    n_rows: int
    n_cols: int

class QualityResponse(BaseModel):
    quality_score: float
    ok_for_model: bool
    latency_ms: float
    flags: Dict[str, Any]

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "ok", "service": "eda-quality-api", "version": "0.1.0"}

@app.post("/quality", response_model=QualityResponse)
async def quality_from_json(request: QualityRequest):
    """Анализ качества данных из JSON-данных"""
    start_time = time.time()
    
    # Для упрощения: в реальном проекте здесь нужно преобразовать JSON в DataFrame
    # Для демо-версии имитируем данные
    mock_summary = DatasetSummary(
        n_rows=request.n_rows,
        n_cols=request.n_cols,
        columns=[]
    )
    mock_missing = pd.DataFrame({"missing_share": [0.1, 0.2, 0.05]})
    
    # Вычисляем флаги качества
    flags = compute_quality_flags(mock_summary, mock_missing)
    quality_score = flags["quality_score"]
    ok_for_model = quality_score > 0.5
    
    latency_ms = (time.time() - start_time) * 1000
    
    return QualityResponse(
        quality_score=quality_score,
        ok_for_model=ok_for_model,
        latency_ms=round(latency_ms, 2),
        flags=flags
    )

@app.post("/quality-from-csv")
async def quality_from_csv(file: UploadFile = File(...)):
    """Анализ качества данных из CSV-файла"""
    start_time = time.time()
    
    # Проверка типа файла
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Чтение CSV файла
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Анализ данных
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "quality_score": flags["quality_score"],
            "ok_for_model": flags["quality_score"] > 0.5,
            "latency_ms": round(latency_ms, 2),
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "max_missing_share": flags["max_missing_share"],
            "flags": flags
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")

@app.post("/quality-flags-from-csv")
async def quality_flags_from_csv(file: UploadFile = File(...)):
    """Полный анализ флагов качества из CSV-файла (HW03)"""
    start_time = time.time()
    
    # Проверка типа файла
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Чтение CSV файла
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Анализ данных с использованием эвристик из HW03
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Формируем ответ с флагами качества из HW03
        result = {
            "latency_ms": round(latency_ms, 2),
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "flags": {
                "too_few_rows": flags["too_few_rows"],
                "too_many_columns": flags["too_many_columns"],
                "too_many_missing": flags["too_many_missing"],
                "has_constant_columns": flags["has_constant_columns"],
                "has_many_zero_values": flags["has_many_zero_values"],
                "max_missing_share": flags["max_missing_share"],
                "quality_score": flags["quality_score"]
            }
        }
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")