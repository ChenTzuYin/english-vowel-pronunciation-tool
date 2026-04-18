import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
from PIL import Image, ImageDraw
import io
import librosa
import numpy as np
import scipy.signal
import json

# --- 1. 設定與資料結構 ---
st.set_page_config(page_title="英語母音發音輔助工具", layout="wide")

# 整合學術常模、判定範圍與像素座標
# 範圍設定：F1/F2 (min, max)
VOWEL_MAP = {
    "i (eat/see)": {
        "prefix": "01", "v_key": "high_i", "word": "eat",
        "target_px": (196, 115),
        "ref": {
            "female": {"f1": 310, "f2": 2790, "range_f1": (250, 400), "range_f2": (2300, 3200)},
            "male": {"f1": 270, "f2": 2290, "range_f1": (220, 350), "range_f2": (1900, 2600)}
        }
    },
    "eɪ (ate/say)": {
        "prefix": "02", "v_key": "ei", "word": "ate",
        "target_px": (196, 122),
        "ref": {
            "female": {"f1": 480, "f2": 2400, "range_f1": (400, 550), "range_f2": (2000, 2600)},
            "male": {"f1": 400, "f2": 2100, "range_f1": (350, 450), "range_f2": (1800, 2300)}
        }
    },
    "ɛ (bed/egg)": {
        "prefix": "03", "v_key": "epsilon", "word": "bed",
        "target_px": (204, 136),
        "ref": {
            "female": {"f1": 610, "f2": 2330, "range_f1": (550, 700), "range_f2": (1800, 2400)},
            "male": {"f1": 530, "f2": 1840, "range_f1": (450, 600), "range_f2": (1500, 2000)}
        }
    },
    "æ (bad/cat)": {
        "prefix": "04", "v_key": "ash", "word": "bad",
        "target_px": (214, 153),
        "ref": {
            "female": {"f1": 860, "f2": 2050, "range_f1": (750, 1000), "range_f2": (1700, 2200)},
            "male": {"f1": 660, "f2": 1720, "range_f1": (600, 800), "range_f2": (1400, 1900)}
        }
    },
    "u (too/zoo)": {
        "prefix": "05", "v_key": "high_u", "word": "too",
        "target_px": (247, 109),
        "ref": {
            "female": {"f1": 370, "f2": 950, "range_f1": (300, 450), "range_f2": (700, 1200)},
            "male": {"f1": 300, "f2": 870, "range_f1": (250, 380), "range_f2": (700, 1100)}
        }
    },
    "oʊ (go/no)": {
        "prefix": "06", "v_key": "ou", "word": "go",
        "target_px": (259, 134),
        "ref": {
            "female": {"f1": 500, "f2": 1000, "range_f1": (450, 600), "range_f2": (800, 1300)},
            "male": {"f1": 450, "f2": 900, "range_f1": (380, 520), "range_f2": (750, 1150)}
        }
    },
    "ɔ (dog/law)": {
        "prefix": "07", "v_key": "open_o", "word": "dog",
        "target_px": (247, 146),
        "ref": {
            "female": {"f1": 700, "f2": 1100, "range_f1": (650, 800), "range_f2": (900, 1400)},
            "male": {"f1": 570, "f2": 840, "range_f1": (520, 650), "range_f2": (750, 1000)}
        }
    },
    "ɑ (box/hot)": {
        "prefix": "08", "v_key": "script_a", "word": "box",
        "target_px": (241, 157),
        "ref": {
            "female": {"f1": 850, "f2": 1220, "range_f1": (750, 1000), "range_f2": (1000, 1500)},
            "male": {"f1": 730, "f2": 1090, "range_f1": (650, 850), "range_f2": (900, 1300)}
        }
    }
}

# --- 2. 工具函數 ---

def get_formants(audio_bytes, gender_max_formant):
    """擷取 F1, F2"""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes)).set_channels(1).set_frame_rate(22050)
    y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    y = scipy.signal.lfilter([1, -0.63], [1], y)
    mid = len(y) // 2
    chunk = y[mid : mid + int(0.2 * 22050)] if len(y) > int(0.2 * 22050) else y
    n_coeffs = int(22050 / 1000) + (4 if gender_max_formant > 5000 else 2)
    lpc_coeffs = librosa.lpc(chunk, order=n_coeffs)
    roots = np.roots(lpc_coeffs)
    roots = [r for r in roots if np.imag(r) > 0]
    angz = np.arctan2(np.imag(roots), np.real(roots))
    formants = sorted(angz * (22050 / (2 * np.pi)))
    return [f for f in formants if f > 250]

def draw_result(base_img_path, st_f1, st_f2, target_px, ref_f1, ref_f2):
    """繪製疊加紅點的圖片"""
    img = Image.open(base_img_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    
    # 計算位移映射 (簡單線性預估)
    f1_diff = st_f1 - ref_f1
    f2_diff = st_f2 - ref_f2
    st_x = target_px[0] - (f2_diff * 0.08)
    st_y = target_px[1] + (f1_diff * 0.15)
    
    # 畫目標藍圈
    r1 = 8
    draw.ellipse([target_px[0]-r1, target_px[1]-r1, target_px[0]+r1, target_px[1]+r1], outline="blue", width=3)
    # 畫學生紅點
    r2 = 10
    draw.ellipse([st_x-r2, st_y-r2, st_x+r2, st_y+r2], fill=(255, 0, 0, 180))
    return img

# --- 3. UI 介面 ---
st.title("🎙️ 英語母音發音即時回饋系統")

with st.sidebar:
    st.header("1. 練習設定")
    selected_label = st.selectbox("選擇目標母音：", list(VOWEL_MAP.keys()))
    gender = st.radio("說話者類型：", ("女性 / 小孩", "男性"))
    
    v_data = VOWEL_MAP[selected_label]
    g_key = "female" if "女性" in gender else "male"
    ref_f1 = v_data["ref"][g_key]["f1"]
    ref_f2 = v_data["ref"][g_key]["f2"]
    max_f = 5500 if g_key == "female" else 5000

# 第一步：示範
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### 目標：/{v_data['v_key']}/")
    audio_path = f"assets/{v_data['prefix']}_{v_data['v_key']}_{v_data['word']}.mp3"
    try:
        st.audio(audio_path)
    except:
        st.info("等待範例音檔...")

with col2:
    img_path = f"assets/{v_data['prefix']}_{v_data['v_key']}_full.png"
    try:
        st.image(img_path, width=300, caption="標準舌位圖")
    except:
        st.info("等待舌位圖...")

st.divider()

# 第二步：錄音分析
st.header("2. 開始練習")
audio_info = mic_recorder(start_prompt="按住說話 🎤", stop_prompt="放開停止 ⏹️", key='vowel_rec')

if audio_info:
    f_list = get_formants(audio_info['bytes'], max_f)
    if len(f_list) >= 2:
        f1, f2 = f_list[0], f_list[1]
        
        # 判定發音落在哪個母音區塊
        detected_key = "未知 (請調整發音)"
        detected_v_data = None
        for k, info in VOWEL_MAP.items():
            r = info["ref"][g_key]
            if r["range_f1"][0] <= f1 <= r["range_f1"][1] and r["range_f2"][0] <= f2 <= r["range_f2"][1]:
                detected_key = k
                detected_v_data = info
                break
        
        # 顯示結果
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.success(f"辨識結果：{detected_key}")
            st.metric("您的 F1 (高低)", f"{round(f1, 1)} Hz", f"{round(f1-ref_f1, 1)} Hz", delta_color="inverse")
            st.metric("您的 F2 (前後)", f"{round(f2, 1)} Hz", f"{round(f2-ref_f2, 1)} Hz")
        
        with res_col2:
            # 如果辨識出特定母音，顯示該母音的疊加圖；否則用目前練習目標的圖
            display_info = detected_v_data if detected_v_data else v_data
            display_img_path = f"assets/{display_info['prefix']}_{display_info['v_key']}_full.png"
            
            try:
                res_img = draw_result(display_img_path, f1, f2, display_info["target_px"], ref_f1, ref_f2)
                st.image(res_img, width=400, caption="紅點：您的位置 | 藍圈：目標位置")
            except:
                st.error("影像處理失敗，請確認 assets 資料夾。")
    else:
        st.warning("音訊不夠清晰，請發長一點的母音。")

