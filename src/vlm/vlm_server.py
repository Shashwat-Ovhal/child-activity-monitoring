import os
import sys
import cv2
import numpy as np
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.vlm.qwen_engine import QwenVLEngine
from src.utils.logger import setup_logger

logger = setup_logger("vlm_server")

app = FastAPI(title="Child Monitoring VLM Microservice", version="1.0.0")

# Global engine instance loaded once at startup
engine: QwenVLEngine = None

class DescribeRequest(BaseModel):
    frames_hex: List[str]
    prompt: str

class DescribeResponse(BaseModel):
    description: str

@app.on_event("startup")
def startup_event():
    global engine
    model_id = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
    device = os.environ.get("VLM_DEVICE", "cuda")
    precision = os.environ.get("VLM_PRECISION", "float16")

    logger.info(f"Starting VLM Server with model: {model_id} on {device}...")
    engine = QwenVLEngine(model_id=model_id, device=device, precision=precision)

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": engine.is_available() if engine else False}

@app.post("/describe", response_model=DescribeResponse)
def describe(req: DescribeRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="VLM engine not initialized")

    try:
        frames = []
        for hx in req.frames_hex:
            nparr = np.frombuffer(bytes.fromhex(hx), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)

        if not frames:
            return DescribeResponse(description="Child not in view.")

        sentence = engine.describe_activity(frames, req.prompt)
        return DescribeResponse(description=sentence)
    except Exception as e:
        logger.error(f"Error handling /describe request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vlm_server:app", host="0.0.0.0", port=8000, reload=False)
