# server.py - Optimized version
import os
import re
import shutil
import uuid
from contextlib import asynccontextmanager

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import faster_whisper

# =============================
# LIFECYCLE MANAGEMENT
# =============================
model = None
model_size = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model, model_size
    model_size = os.environ.get("WHISPER_MODEL", "base")  # base is lighter than small
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    
    print(f"🚀 Loading Whisper Model: {model_size} on {device}")
    
    try:
        model = faster_whisper.WhisperModel(
            model_size,
            device=device,
            compute_type="int8",  # int8 for CPU, "float16" for GPU
            cpu_threads=4,  # Optimize for Render
            num_workers=1,
            download_root="./models"
        )
        print("✅ Whisper model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
    
    yield
    
    # Shutdown: Cleanup
    print("👋 Shutting down Whisper service")
    if model:
        del model

# =============================
# APP SETUP WITH LIFESPAN
# =============================
app = FastAPI(title="Whisper Transcription API", lifespan=lifespan)

# More permissive CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# =============================
# CONFIGURATION
# =============================
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# File size limit (25MB for free tier)
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac'}

# Entity correction for your project
ENTITY_MAP = {
    # Names
    "huzefa": "Huzaifa",
    "huzaifah": "Huzaifa",
    "amars": "AmrZ",
    "amarz": "AmrZ",
    "ali": "Ali",
    "fatima": "Fatima",
    
    # Tech stack
    "remex framework": "Remix Framework",
    "remix framwork": "Remix Framework",
    "remix": "Remix Framework",
    "react": "React",
    "node": "Node.js",
    "postgres": "Postgres",
    "mongodb": "MongoDB",
    "tailwind": "Tailwind CSS",
    
    # Common mishearings
    "whiteboard": "Whiteboard",
    "whisper": "Whisper",
    "ollama": "Ollama",
    "agora": "Agora",
    "groq": "Groq",
    "socket": "Socket.io"
}

def correct_entities(text: str) -> str:
    """Correct common misheard words in transcription"""
    if not text:
        return text
    
    # First, normalize the text
    normalized = text
    
    # Apply corrections
    for wrong, correct in ENTITY_MAP.items():
        pattern = re.compile(rf'(?<!\w){re.escape(wrong)}(?!\w)', re.IGNORECASE)
        normalized = pattern.sub(correct, normalized)
    
    return normalized

def validate_file(file: UploadFile):
    """Validate uploaded file"""
    # Check extension
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size by reading first MAX_FILE_SIZE bytes
    content = file.file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Reset file pointer
    file.file.seek(0)
    return content

# =============================
# HEALTH CHECK
# =============================
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Whisper Transcription API",
        "model": model_size if model else "not_loaded",
        "device": os.environ.get("WHISPER_DEVICE", "cpu"),
        "endpoints": {
            "transcribe": "POST /transcribe",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model": model_size,
        "memory": "ok",
        "timestamp": "2024-01-15T10:30:00Z"  # In production, use actual timestamp
    }

# =============================
# TRANSCRIPTION ENDPOINT
# =============================
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    language: str = Form(""),
    keywords: str = Form(""),
    beam_size: int = Form(5),
    temperature: float = Form(0.0)
):
    # Validate model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    
    # Validate file
    try:
        file_content = validate_file(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")
    
    # Generate unique file path
    file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Determine language
        language_param = language if language and language != "auto" else None
        
        # If no language specified but prompt contains Urdu hints
        if not language_param and re.search(r'[\u0600-\u06FF]', prompt):
            language_param = "ur"
        
        # Build enhanced prompt
        enhanced_prompt = ""
        if keywords:
            enhanced_prompt += f"Important keywords: {keywords}. "
        if prompt:
            enhanced_prompt += f"Context: {prompt}. "
        
        enhanced_prompt += """
        This is a software development meeting transcription.
        Preserve exact names, technical terms, and project titles.
        Maintain proper punctuation and capitalization.
        """
        
        # Transcribe with error handling
        try:
            segments, info = model.transcribe(
                file_path,
                beam_size=min(beam_size, 10),  # Limit beam size
                temperature=min(temperature, 1.0),
                initial_prompt=enhanced_prompt[:400],  # Whisper has prompt length limit
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "threshold": 0.5
                },
                language=language_param,
                condition_on_previous_text=True,
                best_of=5,
                patience=1.0
            )
            
            # Collect segments
            transcript_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    transcript_parts.append(text)
            
            # Combine and clean transcript
            raw_transcript = " ".join(transcript_parts)
            
            # Apply entity correction
            corrected_transcript = correct_entities(raw_transcript)
            
            # Additional cleaning
            cleaned_transcript = re.sub(r'\s+', ' ', corrected_transcript).strip()
            
            # Calculate confidence if available
            avg_prob = None
            if hasattr(segment, 'avg_logprob') and transcript_parts:
                # Simple average probability calculation
                probs = [s.avg_logprob for s in segments if hasattr(s, 'avg_logprob')]
                if probs:
                    avg_prob = sum(probs) / len(probs)
            
            return {
                "success": True,
                "transcript": cleaned_transcript,
                "language": info.language if hasattr(info, 'language') else "unknown",
                "confidence": avg_prob,
                "duration": info.duration if hasattr(info, 'duration') else 0,
                "model": model_size,
                "word_count": len(cleaned_transcript.split())
            }
            
        except Exception as transcribe_error:
            print(f"❌ Transcription error: {transcribe_error}")
            raise HTTPException(
                status_code=500, 
                detail=f"Transcription failed: {str(transcribe_error)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    
    finally:
        # Cleanup temp file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass

# =============================
# BATCH PROCESSING (OPTIONAL)
# =============================
@app.post("/transcribe/batch")
async def transcribe_batch(files: list[UploadFile] = File(...)):
    """Process multiple files (for meetings with multiple audio tracks)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 5:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 5 files per batch")
    
    results = []
    for file in files:
        try:
            # Create a temp endpoint-like call
            result = await transcribe(file)
            results.append({
                "filename": file.filename,
                "success": result.get("success", False),
                "transcript": result.get("transcript", ""),
                "language": result.get("language", "unknown")
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": any(r.get("success", False) for r in results),
        "results": results,
        "total_files": len(files),
        "successful": sum(1 for r in results if r.get("success", False))
    }

# =============================
# RUN SERVER
# =============================
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🌐 Starting server on {host}:{port}")
    print(f"📦 Model: {os.environ.get('WHISPER_MODEL', 'base')}")
    print(f"⚙️ Device: {os.environ.get('WHISPER_DEVICE', 'cpu')}")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        timeout_keep_alive=30,
        log_level="info"
    )