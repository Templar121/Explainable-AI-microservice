from fastapi import FastAPI, UploadFile, File
from explain import explain_image
from fastapi.responses import JSONResponse
import shutil
import uuid
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "API is healthy ✅"}

@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    # Save to temp
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(TEMP_DIR, filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run explanation
    result = explain_image(file_path)

    # Clean up file
    os.remove(file_path)

    return JSONResponse(content=result)
