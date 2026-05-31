#audio_processor.py
import librosa
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)
def load_and_preprocess(file_path, target_sr = 22050, apply_hpss = True):
    try:
        y, sr = librosa.load(file_path, sr = target_sr, mono = True)
        print("Preprocessing...")
        if apply_hpss:
            print("HPSS Processing...")
            
            y_harmonic, y_percussive = librosa.effects.hpss(y, margin=1.2)
            y = y_harmonic
            
            print("HPSS Process Completed")
            
        print("Preprocess Completed")
        return y,sr
    
    except Exception as e:
        print(f"Loading Failed: {e}")
        return None, None
    
        