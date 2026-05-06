# feature_extracter.py
import librosa
import numpy as np

def extract_chroma_features(y, sr, n_fft=2048, hop_length=512):
    #1 STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D)
    print(f"STFT 頻譜圖矩陣維度: {S.shape}")
    print(f"有 {S.shape[0]} 個頻率區間 ，切了 {S.shape[1]} 個時間幀")
    
    #2 Chroma Extraction
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)
    print("Extraction Completed")
    print(f"Chroma 矩陣維度: {chroma.shape}")
    print(f"Y 軸固定為 12 (對應 C 到 B)，X 軸為 {chroma.shape[1]} 個時間幀")
    
    return chroma
    
    
# test region
if __name__ == "__main__":
    import os

    from audio_processor import load_and_preprocess
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, "TestAudio.mp3") 
    
    audio_array, sample_rate = load_and_preprocess(test_file)
    
    if audio_array is not None:
        chroma_matrix = extract_chroma_features(audio_array, sample_rate)
        
        print("第一幀的 12 半音能量分佈 (由 C 到 B):")
        print(np.round(chroma_matrix[:, 0], 3))