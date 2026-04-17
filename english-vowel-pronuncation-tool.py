import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import io
import librosa
import numpy as np
import scipy.signal
import json

# 1. 注入自定義 CSS
st.markdown("""
<style>
    /* 強制定位所有在 mic_recorder 內的按鈕 */
    div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] button,
    .stButton > button, 
    button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 auto !important;
        padding: 0 !important;
        width: 150px !important;
        height: 150px !important;
        border-radius: 50% !important;
        font-size: 20px !important;
        font-weight: bold !important;
        border: 4px solid white !important;
        transition: all 0.2s ease !important;
    }

    /* 藍色狀態：匹配「開始」字眼 */
    button:has(div p:contains("開始")), 
    button:contains("開始") {
        background-color: #007bff !important;
        color: white !important;
        box-shadow: 0 0 15px rgba(0,123,255,0.4) !important;
    }

    /* 紅色閃爍狀態：匹配「停止」字眼 */
    button:has(div p:contains("停止")), 
    button:contains("停止") {
        background-color: #ff4b4b !important;
        color: white !important;
        animation: pulse-red 1.2s infinite !important;
    }

    @keyframes pulse-red {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
        50% { transform: scale(1.05); }
        70% { box-shadow: 0 0 0 25px rgba(255, 75, 75, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
</style>
""", unsafe_allow_html=True)

# 設定頁面
st.set_page_config(page_title="英語母音發音輔助工具", layout="centered")

def get_formants(audio_bytes, gender_max_formant):
    """
    從音訊位元流中擷取共振峰 (F1, F2)
    """
    try:
        # 1. 使用 pydub 讀取並標準化音訊 (解決格式不相容問題)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(22050)
        
        # 2. 轉換為 numpy array 供 librosa 使用
        y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        sr = 22050

        # 3. 預加重處理 (Pre-emphasis)
        y = scipy.signal.lfilter([1, -0.63], [1], y)
        
        # 4. 取得音訊中間 0.2 秒的穩定片段
        mid = len(y) // 2
        duration_samples = int(0.2 * sr)
        if len(y) > duration_samples:
            chunk = y[mid : mid + duration_samples]
        else:
            chunk = y
            
        # 5. LPC 演算法分析 (模擬 Praat)
        # 係數數量根據性別調整：sr/1000 + 2 (男) 或 + 4 (女)
        n_coeffs = int(sr / 1000) + (4 if gender_max_formant > 5000 else 2)
        lpc_coeffs = librosa.lpc(chunk, order=n_coeffs)
        
        # 6. 求根並換算為頻率 (Hz)
        roots = np.roots(lpc_coeffs)
        roots = [r for r in roots if np.imag(r) > 0]
        angz = np.arctan2(np.imag(roots), np.real(roots))
        formants = sorted(angz * (sr / (2 * np.pi)))
        
        # 過濾掉雜訊，只取大於 250Hz 的有效頻率
        valid_formants = [f for f in formants if f > 250]
        return valid_formants

    except Exception as e:
        raise Exception(f"分析過程出錯: {e}")

# UI 介面
st.title("英語母音發音輔助工具")
st.write("本工具會擷取您發音中的 F1 (舌位高低) 與 F2 (舌位前後) 頻率。")

# 側邊欄設定
with st.sidebar:
    st.header("分析設定")
    gender = st.radio("說話者類型：", ("女性 / 小孩", "男性"))
    max_f = 5500 if "女性" in gender else 5000
    st.info(f"當前分析範圍上限: {max_f} Hz")

st.divider()

# 錄音組件
st.subheader("1. 錄製母音")
st.write("請發出一個長母音（例如 [i], [a], [u]），持續約 1-2 秒。")

audio_info = mic_recorder(
    start_prompt="開始錄音 🎤", # 這裡的文字會觸發藍色 CSS
    stop_prompt="停止錄音 ⏹️", # 這裡的文字會觸發紅色 CSS 與閃爍
    key='vowel_recorder'
)

# 處理錄音結果
if audio_info:
    # 播放錄好的聲音 (改用 mp3 格式增加相容性)
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_info['bytes']))
        mp3_fp = io.BytesIO()
        audio.export(mp3_fp, format="mp3") # 強制轉為 mp3
        st.audio(mp3_fp, format="audio/mp3")
    except:
        st.audio(audio_info['bytes']) # 失敗則回退到原始格式
    
    with st.spinner("正在分析共振峰，請稍候..."):
        try:
            f_list = get_formants(audio_info['bytes'], max_f)
            
            if len(f_list) >= 2:
                f1 = f_list[0]
                f2 = f_list[1]
                
                # 顯示數據
                st.subheader("2. 分析數據 (Formants)")
                c1, c2 = st.columns(2)
                c1.metric("F1 (舌位高低)", f"{round(f1, 1)} Hz")
                c2.metric("F2 (舌位前後)", f"{round(f2, 1)} Hz")
                
                # 準備 JSON 下載
                result_data = {
                    "gender": gender,
                    "F1": round(f1, 2),
                    "F2": round(f2, 2),
                    "timestamp": st.session_state.get('last_rec_time', 'N/A')
                }
                
                st.download_button(
                    label="💾 下載分析結果 (JSON)",
                    data=json.dumps(result_data),
                    file_name="vowel_analysis.json",
                    mime="application/json"
                )
            else:
                st.warning("偵測到的共振峰不足，請試著發音更清楚或錄久一點。")
                
        except Exception as e:
            st.error(f"分析失敗。錯誤原因：{e}")

st.divider()
st.caption("提示：F1 較高代表舌位較低（如 [a]）；F2 較高代表舌位較前（如 [i]）。")
