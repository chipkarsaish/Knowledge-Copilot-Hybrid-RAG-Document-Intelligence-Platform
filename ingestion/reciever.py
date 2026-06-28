from pathlib import Path
from fastapi import UploadFile, HTTPException
import shutil
from dataclasses import dataclass


UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md"
}

MAX_FILE_SIZE = 25 * 1024 * 1024      # 25 MB upload max
class FileReceiver:

    def __init__(self):
        self.upload_dir = UPLOAD_DIR
    
    async def receive(self, file: UploadFile) -> Path:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is missing."
            )

        extension = Path(file.filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {extension}"
            )

        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="File exceeds maximum size."
            )

        await file.seek(0)
        save_path = self.upload_dir / file.filename
        
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return save_path
    
@dataclass
class UploadedDocument:
    path: Path
    filename: str
    extension: str
    size: int