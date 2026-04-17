import streamlit as st
from streamlit_mic_recorder import mic_recorder
import librosa
import numpy as np
import scipy.signal
import io
import json

def get_formants(audio_bytes):
    # 1. 將二進位音訊轉為 librosa 可以讀取的格式
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    
    # 2. 預加重處理 (Pre-emphasis) 模擬 Praat 的做法
    y = scipy.signal.lfilter([1, -0.63], [1], y)
    
    # 3. 取得中間一段穩定的音訊 (例如 0.5 秒處)
    # 這裡簡化為取整段的平均 LPC
    win = np.hamming(len(y))
    lpc_coeffs = librosa.lpc(y * win, order=2 + sr // 1000)
    
    # 4. 解方程式求根，找出頻率
    roots = np.roots(lpc_coeffs)
    roots = [r for r in roots if np.imag(r) >= 0]
    angz = np.arctan2(np.imag(roots), np.real(roots))
    
    # 5. 轉換為 Hz 並排序
    formants = sorted(angz * (sr / (2 * np.pi)))
    return [f for f in formants if f > 50] # 過濾極低頻

st.title("語音共振峰擷取工具")

audio_info = mic_recorder(start_prompt="開始錄音", stop_prompt="停止錄音", key='vowel_rec')

if audio_info:
    # 執行分析
    try:
        f_list = get_formants(audio_info['bytes'])
        
        # 整理成我們要的資料
        result = {
            "F1": round(f_list[0], 2) if len(f_list) > 0 else 0,
            "F2": round(f_list[1], 2) if len(f_list) > 1 else 0,
            "F3": round(f_list[2], 2) if len(f_list) > 2 else 0
        }
        
        # 顯示結果
        st.subheader("分析結果 (Hz)")
        col1, col2, col3 = st.columns(3)
        col1.metric("F1 (舌位高度)", f"{result['F1']} Hz")
        col2.metric("F2 (舌位前後)", f"{result['F2']} Hz")
        col3.metric("F3", f"{result['F3']} Hz")
        
        # 轉成 JSON 字串供後續儲存
        result_json = json.dumps(result)
        
        # 妳可以只下載這個 JSON 檔，它非常小！
        st.download_button("💾 下載頻率分析資料", result_json, "formants.json", "application/json")
        
    except Exception as e:
        st.error(f"分析失敗，請再錄一次。錯誤原因：{e}")
