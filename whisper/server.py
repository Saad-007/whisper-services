import os
import re
import shutil
import uuid

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# =============================
# APP SETUP
# =============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# DIRECTORIES
# =============================
TEMP_DIR = "temp_audio"
MODEL_CACHE = "./models"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE, exist_ok=True)

# =============================
# ENTITY CORRECTION MAP
# =============================
ENTITY_MAP = {
    "huzefa": "Huzaifa",
    "huzaifah": "Huzaifa",
    "amars": "AmrZ",
    "amarz": "AmrZ",
    "remex framework": "Remix Framework",
    "remix framwork": "Remix Framework",
}

def correct_entities(text: str) -> str:
    for wrong, correct in ENTITY_MAP.items():
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
        text = pattern.sub(correct, text)
    return text

# =============================
# LOAD WHISPER MODEL
# =============================
device = os.environ.get("WHISPER_DEVICE", "cpu")

print("🚀 Loading Whisper Model...")
try:
    model = WhisperModel(
        "large-v3",
        device=device,
        compute_type="int8",
        download_root=MODEL_CACHE
    )
    print("✅ Loaded: faster-whisper large-v3")
except Exception as e:
    print(f"⚠️ Failed to load large-v3 ({e}), falling back to tiny...")
    model = WhisperModel("tiny", device=device, compute_type="int8")

# =============================
# TRANSCRIPTION ENDPOINT
# =============================
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    language: str = Form(""),
    keywords: str = Form("")
):
    file_path = f"{TEMP_DIR}/{uuid.uuid4()}.wav"

    try:
        # Save uploaded audio
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Language detection / forcing
        language_param = language or ("ur" if re.search(r"\b(urdu|اردو)\b", prompt, re.I) else None)

        # Keyword hint
        keyword_hint = f"Use exact names such as {keywords}. " if keywords else ""

        final_prompt = f"""
        {keyword_hint}
        {prompt}
        Context: Technical software development meeting.
        Mixed Urdu and English.
        Preserve exact names, tools, and project titles.
        """

        # Transcribe
        segments, info = model.transcribe(
            file_path,
            beam_size=8,
            best_of=5,
            temperature=0.0,
            initial_prompt=final_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
            language=language_param,
            condition_on_previous_text=False
        )

        # Assemble transcript
        transcript_parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                if not text.endswith(('.', '?', '!', '۔')):
                    text += '.'
                transcript_parts.append(text)

        transcript = " ".join(transcript_parts)
        transcript = correct_entities(transcript)

        return {"transcript": transcript, "language": info.language, "success": True}

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return {"transcript": "", "success": False, "error": str(e)}

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# =============================
# LOCAL / DEPLOY RUN
# =============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
