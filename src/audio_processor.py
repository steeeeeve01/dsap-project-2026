#audio_processor.py

import librosa
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)
def load_and_preprocess(file_path, target_sr = 22050):
    try:
        y, sr = librosa.load(file_path, sr = target_sr, mono = True)
        print("Preprocess Completed")
        return y,sr
    except Exception as e:
        print("Loading Failed")
        return None, None
    
# Test Part
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    test_file = os.path.join(script_dir, "TestAudio.mp3") 
    audio_array, sample_rate = load_and_preprocess(test_file)
    
    if audio_array is not None:
        second_10_index = 10 * sample_rate
        
        print(audio_array[second_10_index:second_10_index+10])
        print("-" * 30)
        print("整首歌的最大振幅( 0<...<1):", np.max(np.abs(audio_array)))
        