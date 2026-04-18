import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
from PIL import Image, ImageDraw
import io
import librosa
import numpy as np
import scipy.signal
from pathlib import Path

# --- 1. 設定與資料結構 ---
st.set_page_config(page_title="英語母音發音視覺回饋系統", layout="wide")

# 整合之前分析出的最高點座標與範圍
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

# --- 2. 工具函數 ---

def get_formants(audio_bytes, gender_max_formant):
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

def draw_overlay_result(target_v_info, actual_v_info, st_f1, st_f2, g_key):
    """
    target_v_info: 學生原本要練習的母音資料
    actual_v_info: 學生實際發出的母音資料 (若判定失敗則可設為 None)
    """
    # 1. 讀取示範底圖 (剖面圖)
    base_path = Path("assets") / f"{target_v_info['prefix']}_{target_v_info['v_key']}_full.png"
    base_img = Image.open(base_path).convert("RGBA")
    
    # 2. 如果有判定出母音，疊加對應的肌肉圖
    if actual_v_info:
        tongue_path = Path("assets") / f"{actual_v_info['prefix']}_{actual_v_info['v_key']}_{actual_v_info['t_suffix']}.png"
        if tongue_path.exists():
            tongue_img = Image.open(tongue_path).convert("RGBA")
            # 製作 50% 透明度圖層
            alpha = tongue_img.split()[3]
            alpha = alpha.point(lambda p: p * 0.5) # 設定 50% 透明度
            tongue_img.putalpha(alpha)
            # 疊加到背景圖
            base_img.alpha_composite(tongue_img)
    
    draw = ImageDraw.Draw(base_img)
    
    # 3. 畫目標藍圈 (學生練習目標)
    ref_f1 = target_v_info["ref"][g_key]["f1"]
    ref_f2 = target_v_info["ref"][g_key]["f2"]
    tx, ty = target_v_info["target_px"]
    r1 = 8
    draw.ellipse([tx-r1, ty-r1, tx+r1, ty+r1], outline="blue", width=3)
    
    # 4. 計算並畫出學生實際發音紅點
    # 根據 F1/F2 相對位移計算紅點像素位置
    st_x = tx - ((st_f2 - ref_f2) * 0.08)
    st_y = ty + ((st_f1 - ref_f1) * 0.15)
    r2 = 10
    draw.ellipse([st_x-r2, st_y-r2, st_x+r2, st_y+r2], fill=(255, 0, 0, 180))
    
    return base_img

# --- 3. UI 介面 ---
st.title("👅 舌頭肌肉即時疊加回饋系統")

with st.sidebar:
    st.header("1. 設定")
    selected_label = st.selectbox("目標練習母音：", list(VOWEL_MAP.keys()))
    gender = st.radio("性別/年齡類型：", ("女性 / 小孩", "男性"))
    v_data = VOWEL_MAP[selected_label]
    g_key = "female" if "女性" in gender else "male"
    max_f = 5500 if g_key == "female" else 5000

# 介面佈局
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"目標：/{v_data['v_key']}/")
    st.image(f"assets/{v_data['prefix']}_{v_data['v_key']}_full.png", width=300)
    st.audio(f"assets/{v_data['prefix']}_{v_data['v_key']}_{v_data['word']}.mp3")

with col2:
    st.subheader("錄音與診斷")
    rec = mic_recorder(start_prompt="請按住並發長音 🎤", stop_prompt="放開停止分析 ⏹️", key='rec')

if rec:
    f_list = get_formants(rec['bytes'], max_f)
    if len(f_list) >= 2:
        f1, f2 = f_list[0], f_list[1]
        
        # 判定學生實際發出的母音
        actual_v_data = None
        for k, info in VOWEL_MAP.items():
            r = info["ref"][g_key]
            if r["range_f1"][0] <= f1 <= r["range_f1"][1] and r["range_f2"][0] <= f2 <= r["range_f2"][1]:
                actual_v_data = info
                break
        
        # 顯示數值
        st.write(f"📊 您的數據：F1={round(f1,1)}Hz, F2={round(f2,1)}Hz")
        
        # 繪製疊加圖
        res_img = draw_overlay_result(v_data, actual_v_data, f1, f2, g_key)
        
        st.image(res_img, caption="半透明舌頭：您實際的發音形狀 | 底圖：練習目標", use_container_width=True)
        
        if actual_v_data and actual_v_data['v_key'] == v_data['v_key']:
            st.balloons()
            st.success("太棒了！您的發音非常標準。")
        elif actual_v_data:
            st.warning(f"偵測到您目前的舌頭形狀較接近 /{actual_v_data['v_key']}/")
        else:
            st.info("您的發音位置在標準範圍之外，請觀察紅點位置進行調整。")
