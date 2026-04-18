import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import io
import librosa
import numpy as np
import scipy.signal
import json

# 設定頁面
st.set_page_config(page_title="英語母音發音輔助工具", layout="centered")

# --- 1. 資料對應表 (對齊您的檔名結構) ---
# 格式為 "顯示名稱": ("序號", "關鍵字", "代表單字")
VOWEL_MAP = {
    "i (eat/see)": ("01", "high_i", "eat"),
    "eɪ (ate/say)": ("02", "ei", "ate"),
    "ɛ (bed/egg)": ("03", "epsilon", "bed"),
    "æ (bad/cat)": ("04", "ash", "bad"),
    "u (too/zoo)": ("05", "high_u", "too"),
    "oʊ (go/no)": ("06", "ou", "go"),
    "ɔ (dog/law)": ("07", "open_o", "dog"),
    "ɑ (box/hot)": ("08", "script_a", "box")
}

# --- 2. 工具函數 ---
def get_formants(audio_bytes, gender_max_formant):
    """
    從音訊擷取共振峰 (F1, F2)
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(22050)
        y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        sr = 22050
        y = scipy.signal.lfilter([1, -0.63], [1], y)
        mid = len(y) // 2
        duration_samples = int(0.2 * sr)
        chunk = y[mid : mid + duration_samples] if len(y) > duration_samples else y
        n_coeffs = int(sr / 1000) + (4 if gender_max_formant > 5000 else 2)
        lpc_coeffs = librosa.lpc(chunk, order=n_coeffs)
        roots = np.roots(lpc_coeffs)
        roots = [r for r in roots if np.imag(r) > 0]
        angz = np.arctan2(np.imag(roots), np.real(roots))
        formants = sorted(angz * (sr / (2 * np.pi)))
        return [f for f in formants if f > 250]
    except Exception as e:
        raise Exception(f"分析出錯: {e}")

# --- 3. UI 介面佈局 ---
st.title("🔊 英語母音發音輔助工具")

# 側邊欄設定
with st.sidebar:
    st.header("設定")
    selected_label = st.selectbox("選擇練習母音：", list(VOWEL_MAP.keys()))
    prefix, v_key, word = VOWEL_MAP[selected_label]
    
    gender = st.radio("您的性別（影響分析頻率）：", ("女性 / 小孩", "男性"))
    max_f = 5500 if "女性" in gender else 5000

st.divider()

# --- 第一步：範例展示 ---
st.header(f"第一步：聽與看 - /{v_key}/")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("範例音檔")
    # 動態組合路徑：assets/01_high_i_eat.mp3
    audio_path = f"assets/{prefix}_{v_key}_{word}.mp3"
    try:
        st.audio(audio_path)
        st.caption(f"示範單字: {word.capitalize()}")
    except:
        st.error(f"找不到音檔: {audio_path}")

with col2:
    st.subheader("標準舌位圖")
    # 動態組合路徑：assets/01_high_i_full.png
    img_path = f"assets/{prefix}_{v_key}_full.png"
    try:
        st.image(img_path, use_container_width=True)
    except:
        st.info(f"等待上傳圖檔: {img_path}")

st.divider()

# --- 第二步：錄音與回饋 ---
st.header("第二步：您的發音挑戰")
st.write(f"請點擊下方麥克風，並發出「{word}」中的母音長音。")

audio_info = mic_recorder(
    start_prompt="開始錄音 🎤",
    stop_prompt="停止錄音 ⏹️",
    key='vowel_recorder'
)

if audio_info:
    # 播放自己的錄音
    st.audio(audio_info['bytes'])
    
    with st.spinner("分析中..."):
        try:
            f_list = get_formants(audio_info['bytes'], max_f)
            
            if len(f_list) >= 2:
                f1, f2 = f_list[0], f_list[1]
                
                st.subheader("📊 您的聲學數據")
                c1, c2 = st.columns(2)
                c1.metric("F1 (舌位高低)", f"{round(f1, 1)} Hz")
                c2.metric("F2 (舌位前後)", f"{round(f2, 1)} Hz")
                
                # 下載按鈕
                st.download_button(
                    label="💾 下載我的音檔",
                    data=audio_info['bytes'],
                    file_name=f"my_{v_key}.wav",
                    mime="audio/wav"
                )
            else:
                st.warning("錄音長度不足或環境太吵，請再試一次。")
        except Exception as e:
            st.error(f"分析失敗：{e}")

st.caption("提示：完成後可從側邊欄切換下一個母音進行練習。")


