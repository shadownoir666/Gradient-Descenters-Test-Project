from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    hf_token: str | None = None
    groq_api_key: str | None = None
    whisper_model_size: str = "medium"   
    device: str = "cuda"
    audio_dir: str = "data/audio"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
settings = Settings()

import torch
if settings.device == "cuda" and not torch.cuda.is_available():
    print("Warning: Torch not compiled with CUDA enabled. Falling back to CPU.", flush=True)
    settings.device = "cpu"
