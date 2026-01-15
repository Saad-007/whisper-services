# server.py - FULLY OPTIMIZED FOR RENDER FREE TIER
import os
import sys
import gc
import uuid
import shutil
from contextlib import asynccontextmanager

# 🚀 CRITICAL: Memory optimization for 512MB
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper

print("=" * 50)
print("🚀 Whisper Service - Render Free Tier Optimized")
print(f"🐍 Python: {sys.version}")
print(f"💾 Model: tiny (39MB instead of 142MB)")
print("=" * 50)

# =============================
# GLOBAL MODEL
# =============================
MODEL_SIZE = "tiny"  # ⚠️ MUST BE "tiny" for 512MB
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    print("📥 Loading Whisper model...")
    
    try:
        # Force garbage collection before loading
        gc.collect()
        
        # Load TINY model with CPU only
        model = whisper.load_model(
            MODEL_SIZE,
            device="cpu",
            download_root="/tmp/models"  # Use /tmp for Render
        )
        
        print(f"✅ {MODEL_SIZE.upper()} model loaded successfully!")
        print("⚡ Service ready for transcription requests")
        
    except Exception as e:
        print(f"❌ FATAL: Failed to load model: {e}")
        print("💡 Try: MODEL_SIZE='tiny' and device='cpu'")
        model = None
        raise
    
    yield
    
    # Cleanup
    if model:
        del model
    gc.collect()
    print("👋 Service shutdown complete")

# =============================
# APP SETUP
# =============================
app = FastAPI(
    title="Whisper Transcription",
    description="Optimized for Render Free Tier (512MB)",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# HEALTH ENDPOINTS
# =============================
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Whisper API",
        "model": MODEL_SIZE,
        "memory": "optimized",
        "endpoints": ["GET /health", "POST /transcribe"]
    }

@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "healthy", "model": MODEL_SIZE}

# =============================
# TRANSCRIPTION ENDPOINT
# =============================
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    language: str = Form("auto")
):
    if model is None:
        raise HTTPException(503, "Service starting, please wait...")
    
    # Create temp file in /tmp (Render's writable directory)
    import tempfile
    temp_dir = tempfile.mkdtemp()
    file_path = f"{temp_dir}/{uuid.uuid4()}.wav"
    
    try:
        # Read file with size limit (5MB for free tier)
        content = await file.read(5 * 1024 * 1024)  # 5MB max
        
        if not content:
            raise HTTPException(400, "Empty file")
        
        # Save to temp file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Simple transcription
        result = model.transcribe(
            file_path,
            language=None if language == "auto" else language,
            fp16=False,
            verbose=None  # Disable progress bar
        )
        
        return {
            "success": True,
            "transcript": result["text"].strip(),
            "language": result.get("language", "en"),
            "model": MODEL_SIZE
        }
        
    except Exception as e:
        raise HTTPException(500, f"Transcription error: {str(e)}")
    finally:
        # Cleanup temp files
        try:
            import os
            if os.path.exists(file_path):
                os.remove(file_path)
            os.rmdir(temp_dir)
        except:
            pass

# =============================
# START SERVER
# =============================
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,  # Single worker for free tier
        log_level="info"
    )