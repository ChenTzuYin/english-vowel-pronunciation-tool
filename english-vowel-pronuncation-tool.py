import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
from PIL import Image, ImageDraw
import io
import librosa
import numpy as np
import scipy.signal
from pathlib import Path

# --- 1. 初始化設定 ---
st.set_page_config(page_title="英語母音發音視覺回饋系統", layout="wide")

# --- 2. 資料結構整合 ---
VOWEL_MAP = {
    "i (eat/see)": {
        "prefix": "01", "v_key": "high_i", "target_px": (196, 115), "jp_ref": "i",
        "words": ["eat", "see"], "t_suffix": "tougue",
        "ref": {
            "female": {"f1": 310, "f2": 2790, "range_f1": (250, 400), "range_f2": (2300, 3200)},
            "male": {"f1": 270, "f2": 2290, "range_f1": (220, 350), "range_f2": (1900, 2600)}
        }
    },
    "eɪ (ate/say)": {
        "prefix": "02", "v_key": "ei", "target_px": (196, 122), "jp_ref": "e",
        "words": ["ate", "say"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 480, "f2": 2400, "range_f1": (400, 550), "range_f2": (2000, 2600)},
            "male": {"f1": 400, "f2": 2100, "range_f1": (350, 450), "range_f2": (1800, 2300)}
        }
    },
    "ɛ (bed/egg)": {
        "prefix": "03", "v_key": "epsilon", "target_px": (204, 136), "jp_ref": "e",
        "words": ["bed", "egg"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 610, "f2": 2330, "range_f1": (550, 700), "range_f2": (1800, 2400)},
            "male": {"f1": 530, "f2": 1840, "range_f1": (450, 600), "range_f2": (1500, 2000)}
        }
    },
    "æ (bad/cat)": {
        "prefix": "04", "v_key": "ash", "target_px": (214, 153), "jp_ref": "a",
        "words": ["bad", "cat"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 860, "f2": 2050, "range_f1": (750, 1000), "range_f2": (1700, 2200)},
            "male": {"f1": 660, "f2": 1720, "range_f1": (600, 800), "range_f2": (1400, 1900)}
        }
    },
    "u (too/zoo)": {
        "prefix": "05", "v_key": "high_u", "target_px": (247, 109), "jp_ref": "u",
        "words": ["too", "zoo"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 370, "f2": 950, "range_f1": (300, 450), "range_f2": (700, 1200)},
            "male": {"f1": 300, "f2": 870, "range_f1": (250, 380), "range_f2": (700, 1100)}
        }
    },
    "oʊ (go/no)": {
        "prefix": "06", "v_key": "ou", "target_px": (259, 134), "jp_ref": "o",
        "words": ["go", "no"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 500, "f2": 1000, "range_f1": (450, 600), "range_f2": (800, 1300)},
            "male": {"f1": 450, "f2": 900, "range_f1": (380, 520), "range_f2": (750, 1150)}
        }
    },
    "ɔ (dog/law)": {
        "prefix": "07", "v_key": "open_o", "target_px": (247, 146), "jp_ref": "o",
        "words": ["dog", "law"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 700, "f2": 1100, "range_f1": (650, 800), "range_f2": (900, 1400)},
            "male": {"f1": 570, "f2": 840, "range_f1": (520, 650), "range_f2": (750, 1000)}
        }
    },
    "ɑ (box/hot)": {
        "prefix": "08", "v_key": "script_a", "target_px": (241, 157), "jp_ref": "a",
        "words": ["box", "hot"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 850, "f2": 1220, "range_f1": (750, 1000), "range_f2": (1000, 1500)},
            "male": {"f1": 730, "f2": 1090, "range_f1": (650, 850), "range_f2": (900, 1300)}
        }
    }
}

JP_VOWELS = {
    "あ (a)": {"key": "a", "ref_img": "08_script_a_full.png", "audio": "japanese_a.mp3", "target_px": (241, 157)},
    "い (i)": {"key": "i", "ref_img": "01_high_i_full.png", "audio": "japanese_i.mp3", "target_px": (196, 115)},
    "う (u)": {"key": "u", "ref_img": "05_high_u_full.png", "audio": "japanese_u.mp3", "target_px": (247, 109)},
    "え (e)": {"key": "e", "ref_img": "03_epsilon_full.png", "audio": "japanese_e.mp3", "target_px": (204, 136)},
    "お (o)": {"key": "o", "ref_img": "07_open_o_full.png", "audio": "japanese_o.mp3", "target_px": (247, 146)},
}

# --- 3. 核心函式 ---

def get_formants(audio_bytes):
    """分析錄音並回傳前兩個共振峰 (F1, F2)"""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes)).set_channels(1).set_frame_rate(22050)
    y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    y = scipy.signal.lfilter([1, -0.63], [1], y)
    n_coeffs = 12 + 2 
    lpc_coeffs = librosa.lpc(y, order=n_coeffs)
    roots = np.roots(lpc_coeffs)
    roots = [r for r in roots if np.imag(r) > 0]
    angz = np.arctan2(np.imag(roots), np.real(roots))
    formants = sorted(angz * (22050 / (2 * np.pi)))
    return [f for f in formants if f > 250]

def draw_static_target(image_filename, target_px):
    """在指定圖檔的座標上畫出示範紅圈"""
    base_path = Path("assets") / image_filename
    if not base_path.exists():
        return None
    
    img = Image.open(base_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    tx, ty = target_px
    
    draw.ellipse([tx-10, ty-10, tx+10, ty+10], outline="red", width=4)
    return img

def draw_overlay(v_data, f1, f2, g_key, jp_base=None):
    """繪製舌位對比圖"""
    base_path = Path("assets") / f"{v_data['prefix']}_{v_data['v_key']}_full.png"
    if not base_path.exists():
        return None
    
    img = Image.open(base_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    tx, ty = v_data["target_px"]
    
    draw.ellipse([tx-8, ty-8, tx+8, ty+8], outline="red", width=3)
    
    ref = v_data["ref"][g_key]
    st_x = tx - ((f2 - ref["f2"]) * 0.08)
    st_y = ty + ((f1 - ref["f1"]) * 0.15)
    draw.ellipse([st_x-10, st_y-10, st_x+10, st_y+10], fill=(255, 0, 0, 180))
    
    return img

# --- 4. Session State 管理 ---
if 'stage' not in st.session_state:
    st.session_state.stage = "JP_CALIB"
if 'jp_data' not in st.session_state:
    st.session_state.jp_data = {}

# --- 5. UI 介面 ---
st.title("👅 英語母音發音視覺回饋系統")

with st.sidebar:
    st.header("⚙️ 設定")
    gender = st.radio("您的性別：", ("女性 / 小孩", "男性"))
    g_key = "female" if "女性" in gender else "male"
    if st.button("重新開始全部流程"):
        st.session_state.stage = "JP_CALIB"
        st.session_state.jp_data = {}
        st.rerun()

# --- 第一階段：日文母音校正 ---
if st.session_state.stage == "JP_CALIB":
    st.subheader("第一階段：日文母音基準校正")
    
    # --- 補回：進度顯示與導引 ---
    progress = len(st.session_state.jp_data)
    cols_status = st.columns([3, 1])
    with cols_status[0]:
        st.info("請錄製日文的「あいうえお」，系統將根據您的聲音自動調整對比基準。")
    with cols_status[1]:
        st.metric("目前進度", f"{progress} / 5")
    
    # 進度條視覺化
    st.progress(progress / 5)

    selected_jp = st.selectbox("請選擇練習音：", list(JP_VOWELS.keys()))
    jp_v = JP_VOWELS[selected_jp]
    
    col_j1, col_j2 = st.columns(2)
    
    with col_j1:
        jp_target_img = draw_static_target(jp_v['ref_img'], jp_v['target_px'])
        if jp_target_img:
            st.image(jp_target_img, width=350, caption=f"日文「{selected_jp}」的舌面最高點預期位置")
        else:
            st.image(f"assets/{jp_v['ref_img']}", width=350)
        st.audio(f"assets/{jp_v['audio']}")
        
    with col_j2:
        # 顯示該音是否已錄製過
        if jp_v['key'] in st.session_state.jp_data:
            st.success(f"✅ {selected_jp} 已錄製完成")
        
        rec_j = mic_recorder(start_prompt=f"開始錄製 {selected_jp}", key=f"rec_jp_{jp_v['key']}")
        if rec_j:
            f_list = get_formants(rec_j['bytes'])
            if len(f_list) >= 2:
                st.session_state.jp_data[jp_v['key']] = (f_list[0], f_list[1])
                st.balloons() # 增加一點成就感
                st.rerun() # 錄完自動刷新以更新進度
    
    st.divider()
    
    # --- 補回：解鎖英文挑戰按鈕 ---
    if progress >= 5:
        st.success("🎉 太棒了！您已完成所有日文基準校正。")
        if st.button("🔓 點此進入英文挑戰階段 ➔", type="primary", use_container_width=True):
            st.session_state.stage = "EN_LEVEL"
            st.rerun()
    else:
        st.warning(f"還差 {5 - progress} 個音即可解鎖英文挑戰。")
