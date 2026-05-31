import os
import numpy as np
import sounddevice as sd
import librosa
import customtkinter as ctk
import tkinter.filedialog as fd
import threading

# 載入我們的心血結晶
from audio_processor import load_and_preprocess
from feature_extracter import extract_chroma_features
from key_analyzer import KeyEstimator

# 設定 UI 的主題與顏色
ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue") 

class MusicAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🎵 音樂調性分析器 (MVP)")
        self.geometry("500x450") # 稍微加高一點以容納多行文字
        
        # --- UI 元件佈局 ---
        self.title_label = ctk.CTkLabel(self, text="調性分析系統", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(30, 10))
        
        self.status_label = ctk.CTkLabel(self, text="請選擇分析方式...", text_color="gray")
        self.status_label.pack(pady=(0, 20))
        
        # 按鈕區
        self.btn_file = ctk.CTkButton(self, text="📂 選擇音檔 (MP3/WAV)", command=self.process_file, width=200, height=40)
        self.btn_file.pack(pady=10)
        
        self.btn_record = ctk.CTkButton(self, text="🎤 現場錄音 (10秒)", command=self.start_recording, width=200, height=40, fg_color="#C850C0", hover_color="#FFCC70")
        self.btn_record.pack(pady=10)
        
        # 結果顯示區
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.pack(pady=20, padx=40, fill="both", expand=True)
        
        # 顯示第一名的超大字體
        self.result_key = ctk.CTkLabel(self.result_frame, text="--", font=ctk.CTkFont(size=36, weight="bold"))
        self.result_key.pack(pady=(15, 5))
        
        # 顯示 Top 1~3 的詳細列表
        self.result_top3 = ctk.CTkLabel(self.result_frame, text="", font=ctk.CTkFont(size=15), justify="left")
        self.result_top3.pack(pady=(0, 15))

    def update_status(self, message, is_processing=False):
        """更新狀態列的文字，如果正在處理就鎖定按鈕"""
        self.status_label.configure(text=message)
        state = "disabled" if is_processing else "normal"
        self.btn_file.configure(state=state)
        self.btn_record.configure(state=state)
        self.update()

    def analyze_array(self, audio_array, sample_rate):
        """封裝我們之前寫好的核心分析流程，並實作 Top 3 排序"""
        try:
            self.update_status("正在進行 HPSS 濾波與特徵萃取...", True)
            chroma_matrix = extract_chroma_features(audio_array, sample_rate)
            
            self.update_status("正在比對調性樣板...", True)
            pitch_class_distribution = chroma_matrix.sum(axis=1)
            
            estimator = KeyEstimator()
            major_scores, minor_scores = estimator(pitch_class_distribution)
            
            mapping = {0: "C", 1: "C#/Db", 2: "D", 3: "D#/Eb", 4: "E", 5: "F", 6: "F#/Gb", 7: "G", 8: "G#/Ab", 9: "A", 10: "A#/Bb", 11: "B"}
            
            # --- 將大小調的 24 個分數合併並排序 ---
            all_results = []
            for i in range(12):
                all_results.append((f"{mapping[i]} Major", major_scores[i]))
                all_results.append((f"{mapping[i]} Minor", minor_scores[i]))
                
            # 根據信心指數 (Tuple 的第 1 項) 進行降冪排序
            all_results.sort(key=lambda x: x[1], reverse=True)
            
            # 取出前三名
            top_3 = all_results[:3]
            
            # 組合要顯示在 UI 上的字串
            best_key = top_3[0][0]
            top3_text = (
                f"🥇 第一可能: {top_3[0][0]:<10} (分數: {top_3[0][1]:.4f})\n\n"
                f"🥈 第二可能: {top_3[1][0]:<10} (分數: {top_3[1][1]:.4f})\n\n"
                f"🥉 第三可能: {top_3[2][0]:<10} (分數: {top_3[2][1]:.4f})"
            )
            
            # 將結果顯示在 UI 上
            self.result_key.configure(text=best_key)
            self.result_top3.configure(text=top3_text)
            self.update_status("分析完成！")
            
        except Exception as e:
            self.update_status(f"發生錯誤: {e}")

    def process_file(self):
        """處理按鈕：選擇檔案"""
        file_path = fd.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")])
        if not file_path:
            return
            
        def task():
            self.update_status("正在載入音檔...", True)
            audio_array, sr = load_and_preprocess(file_path, apply_hpss=True)
            if audio_array is not None:
                self.analyze_array(audio_array, sr)
            else:
                self.update_status("讀取音檔失敗。")
                
        threading.Thread(target=task).start()

    def start_recording(self):
        """處理按鈕：現場錄音"""
        def task():
            sr = 22050
            # 將錄音時長改為 10 秒
            duration = 10  
            self.update_status(f"🔴 錄音中...請發出聲音！({duration}秒)", True)
            
            recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
            sd.wait() 
            
            audio_array = recording.flatten()
            
            self.update_status("🔴 錄音結束，套用 HPSS...", True)
            y_harmonic, _ = librosa.effects.hpss(audio_array, margin=1.2)
            
            self.analyze_array(y_harmonic, sr)
            
        threading.Thread(target=task).start()

if __name__ == "__main__":
    app = MusicAnalyzerApp()
    app.mainloop()