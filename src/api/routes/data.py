"""
Data-related API endpoints for OpenPerturbation.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from ..models import DataUploadResponse

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIRECTORY = "uploads"

@router.post("/upload", response_model=DataUploadResponse)
async def upload_data_file(file: UploadFile = File(...)):
    """Upload data file for analysis."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided.")

        upload_dir = Path(UPLOAD_DIRECTORY)
        upload_dir.mkdir(exist_ok=True)

        safe_filename = Path(file.filename).name
        file_path = upload_dir / safe_filename
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        file_size = file_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            file_path.unlink()
            raise HTTPException(status_code=413, detail="File too large")
            
        file_extension = file_path.suffix.lower()
        supported_formats = ['.csv', '.json', '.xlsx', '.h5', '.tsv']
        
        if file_extension not in supported_formats:
            file_path.unlink()
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        return DataUploadResponse(
            filename=safe_filename,
            file_path=str(file_path),
            file_size=file_size,
            format=file_extension,
            status="uploaded",
            message=f"File {safe_filename} uploaded successfully"
        )
            
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download a file"""
    try:
        file_path = Path(UPLOAD_DIRECTORY) / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=str(filename),
            media_type='application/octet-stream'
        )
            
    except Exception as e:
        logger.error(f"File download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{filename}")
async def delete_data_file(filename: str):
    """Delete an uploaded file"""
    try:
        file_path = Path(UPLOAD_DIRECTORY) / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path.unlink()
        return {"message": f"File {filename} deleted successfully"}
        
    except Exception as e:
        logger.error(f"File deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 