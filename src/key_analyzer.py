#key_analyzer.py
import os
import numpy as np
import scipy.linalg
from scipy.stats import zscore
from dataclasses import dataclass
from typing import List

# from previous module
from audio_processor import load_and_preprocess
from feature_extracter import extract_chroma_features

@dataclass
class KeyEstimator:
#Krumhansl-Schmuckler key-finding algorithm
    major : np.ndarray = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor : np.ndarray = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
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
    
    
if __name__ == "__main__":
    print("Music Key Estimator (Prototype)")
    user_input_path = input("\n請輸入音檔路徑: ").strip(" '\"")
    if not os.path.exists(user_input_path):
        print("Error: Cannot find such file through the path")
    else:
        audio_array, sample_rate = load_and_preprocess(user_input_path)
        
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
            
            
        
        
        
        