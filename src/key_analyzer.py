# key_analyzer.py
import os
import numpy as np
import scipy.linalg
from scipy.stats import zscore
from dataclasses import dataclass, field
from typing import List

# from previous module
from audio_processor import load_and_preprocess
from feature_extracter import extract_chroma_features

import time


@dataclass
class KeyEstimator:
#Krumhansl-Schmuckler key-finding algorithm
    major : np.ndarray = field(default_factory = lambda: np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]))
    minor : np.ndarray = field(default_factory = lambda: np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]))
    def __post_init__(self):
        self.major = zscore(self.major)
        self.major_norm = scipy.linalg.norm(self.major)
        self.major = scipy.linalg.circulant(self.major)
        
        self.minor = zscore(self.minor)
        self.minor_norm = scipy.linalg.norm(self.minor)
        self.minor = scipy.linalg.circulant(self.minor)
        
    def __call__(self, x: np.array) -> List[np.array]:
        x = zscore(x)
        x_norm = scipy.linalg.norm(x)
        
        coeffs_major = self.major.T.dot(x) / self.major_norm / x_norm
        coeffs_minor = self.minor.T.dot(x) / self.minor_norm / x_norm

        return coeffs_major, coeffs_minor    
    
    
 # 記得在檔案最上方 import time

def sliding_window_naive(chroma, window_size):
    """
    每次滑動都把視窗內的矩陣切片，並重新完整加總。
    O(N * W)
    """
    num_frames = chroma.shape[1]
    results = []
    
    for i in range(num_frames - window_size + 1):
        window_sum = np.sum(chroma[:, i : i + window_size], axis=1)
        results.append(window_sum)
        
    return results

def sliding_window_optimized(chroma, window_size):
    """
    利用前一次的結果，扣掉離開的幀，加上進入的幀。
    O(N)
    """
    num_frames = chroma.shape[1]
    results = []
    
    current_sum = np.sum(chroma[:, 0:window_size], axis=1)
    results.append(current_sum.copy())
    
    for i in range(1, num_frames - window_size + 1):
        # 新總和 = 舊總和 - 離開的最左側幀 + 進入的最右側幀
        frame_out = chroma[:, i - 1]
        frame_in = chroma[:, i + window_size - 1]
        
        current_sum = current_sum - frame_out + frame_in
        results.append(current_sum.copy())
        
    return results

if __name__ == "__main__":
    print("🚀 Music Key Estimator (Final Version) 🚀")
    user_input_path = input("\n請輸入音檔路徑: ").strip(" '\"")
    
    if not os.path.exists(user_input_path):
        print("❌ Error: Cannot find such file through the path")
    else:
        # 開啟 HPSS 前置處理
        audio_array, sample_rate = load_and_preprocess(user_input_path, apply_hpss=True)
        
        if audio_array is not None: 
            chroma_matrix = extract_chroma_features(audio_array, sample_rate)
            
            # --- 效能測試區塊開始 ---
            print("\n" + "="*40)
            print("執行 DSA 效能分析：滑動視窗演算法比較")
            print("="*40)
            
            # 設定視窗大小為 1000 幀 (大約是幾秒鐘的長度，視窗越大差異越明顯)
            WINDOW_SIZE = 1000 
            
            # 1. 測試暴力法 (Naive)
            start_time_naive = time.perf_counter()
            naive_results = sliding_window_naive(chroma_matrix, WINDOW_SIZE)
            end_time_naive = time.perf_counter()
            time_naive = end_time_naive - start_time_naive
            
            # 2. 測試優化法 (Optimized)
            start_time_opt = time.perf_counter()
            opt_results = sliding_window_optimized(chroma_matrix, WINDOW_SIZE)
            end_time_opt = time.perf_counter()
            time_opt = end_time_opt - start_time_opt
            
            print(f"音訊總幀數: {chroma_matrix.shape[1]}")
            print(f"視窗大小: {WINDOW_SIZE}")
            print(f"暴力法 (O(N*W)) 執行時間:\t {time_naive:.6f} 秒")
            print(f"優化法 (O(N))   執行時間:\t {time_opt:.6f} 秒")
            print(f"效能提升倍率:\t\t\t 約 {time_naive / time_opt:.2f} 倍")
            print("="*40 + "\n")
            # --- 效能測試區塊結束 ---

            # 後續原本的單一調性分析可以先保留，或是改用 naive_results[0] 來做測試
            pitch_class_distribution = chroma_matrix.sum(axis = 1)
            
            key_estimator = KeyEstimator()
            major_scores, minor_scores = key_estimator(pitch_class_distribution)
            
            mapping = {0: "C", 1: "C#/Db", 2: "D", 3: "D#/Eb", 4: "E", 5: "F", 6: "F#/Gb", 7: "G", 8: "G#/Ab", 9: "A", 10: "A#/Bb", 11: "B"}
            
            best_major_idx = np.argmax(major_scores)
            best_minor_idx = np.argmax(minor_scores)
            
            print("\n Major Key Coefficients:")
            for i, coeff in enumerate(major_scores):
                print(f"{mapping[i]}:\t{coeff:.2f}")
            
            print("\n Minor Key Coefficients:")
            for i, coeff in enumerate(minor_scores):
                print(f"{mapping[i]}:\t{coeff:.2f}")
                
            if major_scores[best_major_idx] > minor_scores[best_minor_idx]:
                final_key = f"{mapping[best_major_idx]} Major"
                confidence = major_scores[best_major_idx]
            else:
                final_key = f"{mapping[best_minor_idx]} Minor"
                confidence = minor_scores[best_minor_idx]
                
            print(f"Estimated Key: {final_key}")
            print(f"Score: {confidence:.4f}")

"""
if __name__ == "__main__":
    print("Music Key Estimator (Prototype)")
    user_input_path = input("\n請輸入音檔路徑: ").strip(" '\"")
    if not os.path.exists(user_input_path):
        print("Error: Cannot find such file through the path")
    else:
        audio_array, sample_rate = load_and_preprocess(user_input_path, apply_hpss = True)
        
        if audio_array is not None: 
            chroma_matrix = extract_chroma_features(audio_array, sample_rate)
            
            pitch_class_distribution = chroma_matrix.sum(axis = 1)
            
            key_estimator = KeyEstimator()
            major_scores, minor_scores = key_estimator(pitch_class_distribution)
            
            mapping = {0: "C", 1: "C#/Db", 2: "D", 3: "D#/Eb", 4: "E", 5: "F", 6: "F#/Gb", 7: "G", 8: "G#/Ab", 9: "A", 10: "A#/Bb", 11: "B"}
            
            best_major_idx = np.argmax(major_scores)
            best_minor_idx = np.argmax(minor_scores)
            
            print("\n Major Key Coefficients:")
            for i, coeff in enumerate(major_scores):
                print(f"{mapping[i]}:\t{coeff:.2f}")
            
            print("\n Minor Key Coefficients:")
            for i, coeff in enumerate(minor_scores):
                print(f"{mapping[i]}:\t{coeff:.2f}")
                
            if major_scores[best_major_idx] > minor_scores[best_minor_idx]:
                final_key = f"{mapping[best_major_idx]} Major"
                confidence = major_scores[best_major_idx]
            else:
                final_key = f"{mapping[best_minor_idx]} Minor"
                confidence = minor_scores[best_minor_idx]
                
            print(f"Estimated Key: {final_key}")
            print(f"Score: {confidence:.4f}")
 """
 
 
            
        
        
        
        