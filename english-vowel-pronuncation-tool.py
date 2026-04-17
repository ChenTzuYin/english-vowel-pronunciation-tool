import streamlit as st
from streamlit_mic_recorder import mic_recorder
import librosa
import numpy as np
import scipy.signal
import io
import json
import soundfile as sf  # 新增這個來處理格式

def get_formants(audio_bytes):
    # --- 修正處：使用 soundfile 讀取 BytesIO ---
    data, sr = sf.read(io.BytesIO(audio_bytes))
    
    # 如果是雙聲道，轉成單聲道
    if len(data.shape) > 1:
        y = np.mean(data, axis=1)
    else:
        y = data
        
    # 確保採樣率適合分析 (librosa 預設處理)
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050

    # 1. 預加重處理
    y = scipy.signal.lfilter([1, -0.63], [1], y)
    
    # 2. 為了穩定，取音訊中間段落 0.2 秒
    mid = len(y) // 2
    chunk = y[mid:mid+int(0.2*sr)] if len(y) > int(0.2*sr) else y
    
    # 3. LPC 演算法 (n_coeffs 建議為 sr/1000 + 2)
    n_coeffs = int(sr / 1000) + 2
    lpc_coeffs = librosa.lpc(chunk, order=n_coeffs)
    
    # 4. 求根並換算 Hz
    roots = np.roots(lpc_coeffs)
    roots = [r for r in roots if np.imag(r) >= 0]
    angz = np.arctan2(np.imag(roots), np.real(roots))
    formants = sorted(angz * (sr / (2 * np.pi)))
    
    return [f for f in formants if f > 50]

st.title("語音共振峰擷取工具 🎙️")

# 增加性別選擇，這能讓分析更準確 (模擬 Praat 的 Max Formant 設定)
gender = st.radio("請選擇說話者性別（影響分析範圍）：", ("女性 (Max 5500Hz)", "男性 (Max 5000Hz)"))

audio_info = mic_recorder(start_prompt="按住錄音", stop_prompt="停止錄音", key='vowel_rec')

if audio_info:
    try:
        f_list = get_formants(audio_info['bytes'])
        
        if len(f_list) >= 2:
            f1, f2 = f_list[0], f_list[1]
            st.subheader("分析結果")
            col1, col2 = st.columns(2)
            col1.metric("F1 (舌位高低)", f"{round(f1, 1)} Hz")
            col2.metric("F2 (舌位前後)", f"{round(f2, 1)} Hz")
            
            # 這裡可以加入繪製母音圖的邏輯
        else:
            st.warning("錄音太短或音質不清晰，無法辨識共振峰。")
            
    except Exception as e:
        st.error(f"分析失敗，請再錄一次。錯誤原因：{e}")
