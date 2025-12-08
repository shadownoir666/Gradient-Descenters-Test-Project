from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

# --- FFmpeg Fix for Windows Winget Install ---
# Check if ffmpeg is in PATH
if not shutil.which("ffmpeg"):
    # Common Winget install path pattern
    user_home = os.path.expanduser("~")
    # Exact path found in verification
    possible_paths = [
        os.path.join(user_home, r"AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"),
        os.path.join(user_home, r"AppData\Local\Microsoft\WinGet\Links")
    ]
    
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "ffmpeg.exe")):
            print(f"Found FFmpeg at {p}. Adding to PATH...", flush=True)
            os.environ["PATH"] += os.pathsep + p
            break
# ---------------------------------------------

from app.routes import audio, health, media

app = FastAPI(title="SonicTrace")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
app.include_router(media.router, prefix="/api")