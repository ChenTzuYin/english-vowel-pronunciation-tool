import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
from PIL import Image, ImageDraw
import io
import librosa
import numpy as np
import scipy.signal
from pathlib import Path

# --- 1. 初始化設定與資料結構 ---
st.set_page_config(page_title="英語母音發音視覺回饋系統", layout="wide")

VOWEL_MAP = {
    "i (eat/see)": {
        "prefix": "01", "v_key": "high_i", "word": "eat", "t_suffix": "tougue",
        "target_px": (196, 115),
        "ref": {
            "female": {"f1": 310, "f2": 2790, "range_f1": (250, 400), "range_f2": (2300, 3200)},
            "male": {"f1": 270, "f2": 2290, "range_f1": (220, 350), "range_f2": (1900, 2600)}
        }
    },
    "eɪ (ate/say)": {
        "prefix": "02", "v_key": "ei", "word": "ate", "t_suffix": "tongue",
        "target_px": (196, 122),
        "ref": {
            "female": {"f1": 480, "f2": 2400, "range_f1": (400, 550), "range_f2": (2000, 2600)},
            "male": {"f1": 400, "f2": 2100, "range_f1": (350, 450), "range_f2": (1800, 2300)}
        }
    },
    "ɛ (bed/egg)": {
        "prefix": "03", "v_key": "epsilon", "word": "bed", "t_suffix": "tongue",
        "target_px": (204, 136),
        "ref": {
            "female": {"f1": 610, "f2": 2330, "range_f1": (550, 700), "range_f2": (1800, 2400)},
            "male": {"f1": 530, "f2": 1840, "range_f1": (450, 600), "range_f2": (1500, 2000)}
        }
    },
    "æ (bad/cat)": {
        "prefix": "04", "v_key": "ash", "word": "bad", "t_suffix": "tongue",
        "target_px": (214, 153),
        "ref": {
            "female": {"f1": 860, "f2": 2050, "range_f1": (750, 1000), "range_f2": (1700, 2200)},
            "male": {"f1": 660, "f2": 1720, "range_f1": (600, 800), "range_f2": (1400, 1900)}
        }
    },
    "u (too/zoo)": {
        "prefix": "05", "v_key": "high_u", "word": "too", "t_suffix": "tongue",
        "target_px": (247, 109),
        "ref": {
            "female": {"f1": 370, "f2": 950, "range_f1": (300, 450), "range_f2": (700, 1200)},
            "male": {"f1": 300, "f2": 870, "range_f1": (250, 380), "range_f2": (700, 1100)}
        }
    },
    "oʊ (go/no)": {
        "prefix": "06", "v_key": "ou", "word": "go", "t_suffix": "tongue",
        "target_px": (259, 134),
        "ref": {
            "female": {"f1": 500, "f2": 1000, "range_f1": (450, 600), "range_f2": (800, 1300)},
            "male": {"f1": 450, "f2": 900, "range_f1": (380, 520), "range_f2": (750, 1150)}
        }
    },
    "ɔ (dog/law)": {
        "prefix": "07", "v_key": "open_o", "word": "dog", "t_suffix": "tongue",
        "target_px": (247, 146),
        "ref": {
            "female": {"f1": 700, "f2": 1100, "range_f1": (650, 800), "range_f2": (900, 1400)},
            "male": {"f1": 570, "f2": 840, "range_f1": (520, 650), "range_f2": (750, 1000)}
        }
    },
    "ɑ (box/hot)": {
        "prefix": "08", "v_key": "script_a", "word": "box", "t_suffix": "tongue",
        "target_px": (241, 157),
        "ref": {
            "female": {"f1": 850, "f2": 1220, "range_f1": (750, 1000), "range_f2": (1000, 1500)},
            "male": {"f1": 730, "f2": 1090, "range_f1": (650, 850), "range_f2": (900, 1300)}
        }
    }
}

# --- 2. 函式定義 ---

def get_formants(audio_bytes, gender_max_formant):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes)).set_channels(1).set_frame_rate(22050)
    y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    y = scipy.signal.lfilter([1, -0.63], [1], y)
    mid = len(y) // 2
    chunk_size = int(0.2 * 22050)
    chunk = y[mid : mid + chunk_size] if len(y) > chunk_size else y
    n_coeffs = int(22050 / 1000) + (4 if gender_max_formant > 5000 else 2)
    lpc_coeffs = librosa.lpc(chunk, order=n_coeffs)
    roots = np.roots(lpc_coeffs)
    roots = [r for r in roots if np.imag(r) > 0]
    angz = np.arctan2(np.imag(roots), np.real(roots))
    formants = sorted(angz * (22050 / (2 * np.pi)))
    return [f for f in formants if f > 250]

def draw_overlay_result(target_v_info, actual_v_info, st_f1, st_f2, g_key):
    base_path = Path("assets") / f"{target_v_info['prefix']}_{target_v_info['v_key']}_full.png"
    base_img = Image.open(base_path).convert("RGBA")
    base_size = base_img.size
    
    if actual_v_info:
        tongue_path = Path("assets") / f"{actual_v_info['prefix']}_{actual_v_info['v_key']}_{actual_v_info['t_suffix']}.png"
        if tongue_path.exists():
            tongue_img = Image.open(tongue_path).convert("RGBA").resize(base_size, Image.Resampling.LANCZOS)
            alpha = tongue_img.split()[3].point(lambda p: p * 0.5) 
            tongue_img.putalpha(alpha)
            base_img.alpha_composite(tongue_img)
    
    draw = ImageDraw.Draw(base_img)
    tx, ty = target_v_info["target_px"]
    draw.ellipse([tx-8, ty-8, tx+8, ty+8], outline="blue", width=3)
    
    ref_f1 = target_v_info["ref"][g_key]["f1"]
    ref_f2 = target_v_info["ref"][g_key]["f2"]
    st_x = tx - ((st_f2 - ref_f2) * 0.08)
    st_y = ty + ((st_f1 - ref_f1) * 0.15)
    draw.ellipse([st_x-10, st_y-10, st_x+10, st_y+10], fill=(255, 0, 0, 180))
    return base_img

# --- 3. UI 介面 ---
st.title("👅 英語母音發音視覺回饋與診斷系統")

with st.sidebar:
    st.header("1. 設定")
    selected_label = st.selectbox("練習母音：", list(VOWEL_MAP.keys()))
    gender = st.radio("您的性別：", ("女性 / 小孩", "男性"))
    v_data = VOWEL_MAP[selected_label]
    g_key = "female" if "女性" in gender else "male"
    max_f = 5500 if g_key == "female" else 5000
    ref_f1 = v_data["ref"][g_key]["f1"]
    ref_f2 = v_data["ref"][g_key]["f2"]

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"目標：/{v_data['v_key']}/")
    st.image(f"assets/{v_data['prefix']}_{v_data['v_key']}_full.png", width=350)
    st.audio(f"assets/{v_data['prefix']}_{v_data['v_key']}_{v_data['word']}.mp3")

with col2:
    st.subheader("2. 錄音練習")
    rec = mic_recorder(start_prompt="按住錄音 🎤", stop_prompt="停止分析 ⏹️", key='rec')

st.divider()

if rec:
    f_list = get_formants(rec['bytes'], max_f)
    if len(f_list) >= 2:
        f1, f2 = f_list[0], f_list[1]
        
        # 判定學生發音落點
        actual_v_data = None
        for k, info in VOWEL_MAP.items():
            r = info["ref"][g_key]
            if r["range_f1"][0] <= f1 <= r["range_f1"][1] and r["range_f2"][0] <= f2 <= r["range_f2"][1]:
                actual_v_data = info
                break
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.subheader("📊 診斷建議")
            
            # --- F1 (舌位高低) 建議 ---
            f1_diff = f1 - ref_f1
            if abs(f1_diff) < 50:
                st.write("✅ **舌位高低：** 非常標準！")
            elif f1_diff > 0:
                st.write(f"❌ **舌位高低：** 您的 F1 偏高約 {int(f1_diff)} Hz。建議：**舌頭再抬高一點點**。")
            else:
                st.write(f"❌ **舌位高低：** 您的 F1 偏低約 {int(abs(f1_diff))} Hz。建議：**嘴巴再張大一點，舌位放低**。")

            # --- F2 (舌位前後) 建議 ---
            f2_diff = f2 - ref_f2
            if abs(f2_diff) < 150:
                st.write("✅ **舌位前後：** 非常標準！")
            elif f2_diff > 0:
                st.write(f"❌ **舌位前後：** 您的 F2 偏高約 {int(f2_diff)} Hz。建議：**舌頭稍稍往後縮一點**。")
            else:
                st.write(f"❌ **舌位前後：** 您的 F2 偏低約 {int(abs(f2_diff))} Hz。建議：**舌頭再往前推一點**。")

            st.metric("當前 F1", f"{round(f1,1)} Hz", f"{round(f1_diff,1)} Hz", delta_color="inverse")
            st.metric("當前 F2", f"{round(f2,1)} Hz", f"{round(f2_diff,1)} Hz")

        with res_col2:
            st.subheader("📸 視覺比對")
            res_img = draw_overlay_result(v_data, actual_v_data, f1, f2, g_key)
            st.image(res_img, width=350, caption="紅點為您的實際位置")
