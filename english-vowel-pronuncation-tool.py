import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import io
import librosa
import numpy as np

def get_formants(audio_bytes):
    # --- 關鍵修正：用 pydub 自動辨識並讀取音訊流 ---
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        # 強制轉為單聲道並設定採樣率
        audio = audio.set_channels(1).set_frame_rate(22050)
        
        # 轉為 numpy array 讓 librosa 處理
        y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        sr = 22050
    except Exception as e:
        raise Exception(f"音訊解碼失敗: {e}")

    # --- 以下維持原本的分析邏輯 ---
    # 預加重與 LPC 分析...
    # (省略部分代碼以保持簡潔)
    return [600, 1200] # 範例回傳
