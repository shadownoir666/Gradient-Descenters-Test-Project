import torch
import librosa
import numpy as np
from transformers import pipeline
from app.config import settings

class EmotionAnalyzer:
    def __init__(self):
        self.device = settings.device
        # Use a specific model for emotion recognition
        # reliable choice: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
        self.model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        
        # Load pipeline on CPU or GPU based on config
        # pipeline takes 'device' argument: -1 for CPU, 0+ for GPU ID
        device_id = 0 if self.device == "cuda" else -1
        
        print(f"Loading Emotion model {self.model_name} on {self.device}...", flush=True)
        self.classifier = pipeline(
            "audio-classification", 
            model=self.model_name, 
            device=device_id
        )

    def detect_emotion(self, audio_path: str, start: float, end: float) -> dict:
        """
        Loads a specific segment of audio and returns the top predicted emotion.
        Returns: {"emotion": str, "score": float}
        """
        try:
            # Load only the relevant segment
            duration = end - start
            if duration < 0.1:
                return {"emotion": "neutral", "score": 0.0}

            # Librosa load (resample to 16kHz as typically required by wav2vec2 models)
            y, sr = librosa.load(audio_path, sr=16000, offset=start, duration=duration)
            
            if len(y) == 0:
                return {"emotion": "neutral", "score": 0.0}

            # Run inference
            # The pipeline accepts numpy array directly
            results = self.classifier(y, top_k=1)
            
            if results:
                top_result = results[0]
                return {
                    "emotion": top_result["label"],
                    "score": float(top_result["score"])
                }
            
            return {"emotion": "neutral", "score": 0.0}

        except Exception as e:
            print(f"[ERROR] Emotion detection failed for {start}-{end}: {e}")
            return {"emotion": "error", "score": 0.0}
