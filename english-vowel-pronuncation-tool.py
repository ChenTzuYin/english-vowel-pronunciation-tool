import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont
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
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes)).set_channels(1).set_frame_rate(22050)
    y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    y = scipy.signal.lfilter([1, -0.63], [1], y)
    lpc_coeffs = librosa.lpc(y, order=14)
    roots = np.roots(lpc_coeffs)
    roots = [r for r in roots if np.imag(r) > 0]
    angz = np.arctan2(np.imag(roots), np.real(roots))
    formants = sorted(angz * (22050 / (2 * np.pi)))
    return [f for f in formants if f > 250]

def draw_static_target(image_filename, target_px):
    base_path = Path("assets") / image_filename
    if not base_path.exists(): return None
    img = Image.open(base_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    tx, ty = target_px
    draw.ellipse([tx-10, ty-10, tx+10, ty+10], outline="red", width=4)
    return img

def draw_overlay(v_data, f1, f2, g_key, jp_base=None):
    base_path = Path("assets") / f"{v_data['prefix']}_{v_data['v_key']}_full.png"
    if not base_path.exists(): return None
    img = Image.open(base_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    tx, ty = v_data["target_px"]
    draw.ellipse([tx-8, ty-8, tx+8, ty+8], outline="red", width=3)
    ref = v_data["ref"][g_key]
    st_x = tx - ((f2 - ref["f2"]) * 0.08)
    st_y = ty + ((f1 - ref["f1"]) * 0.15)
    draw.ellipse([st_x-10, st_y-10, st_x+10, st_y+10], fill=(255, 0, 0, 180))
    return img

def draw_final_jp_map(jp_data):
    base_img_name = "08_script_a_full.png" 
    base_path = Path("assets") / base_img_name
    if not base_path.exists(): return None
    base_img = Image.open(base_path).convert("RGBA")
    overlay = Image.new('RGBA', base_img.size, (255, 255, 255, 0))
    draw_ov = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(base_img)
    try: font = ImageFont.truetype("Arial Bold.ttf", 24)
    except: font = ImageFont.load_default()
    offsets = {"a": (-50, 40), "i": (-60, -50), "u": (40, -50), "e": (-70, 0), "o": (40, 40)}
    for jp_label, v_info in JP_VOWELS.items():
        jp_key = v_info['key']
        if jp_key in jp_data:
            tx, ty = v_info['target_px']
            draw_base.ellipse([tx-12, ty-12, tx+12, ty+12], fill=(255, 0, 0, 255))
    for jp_label, v_info in JP_VOWELS.items():
        jp_key = v_info['key']
        if jp_key in jp_data:
            f1, f2 = jp_data[jp_key]
            tx, ty = v_info['target_px']
            label_text = f"/{jp_key}/ ({int(f1)}, {int(f2)})  "
            dx, dy = offsets.get(jp_key, (30, 30))
            text_x, text_y = tx + dx, ty + dy
            left, top, right, bottom = draw_base.textbbox((text_x, text_y), label_text, font=font)
            draw_ov.rectangle([left-5, top-2, right+5, bottom+2], fill=(255, 255, 255, 160))
            draw_ov.line([tx, ty, text_x, text_y], fill=(100, 100, 100, 150), width=2)
    base_img = Image.alpha_composite(base_img, overlay)
    draw_final = ImageDraw.Draw(base_img)
    for jp_label, v_info in JP_VOWELS.items():
        jp_key = v_info['key']
        if jp_key in jp_data:
            f1, f2 = jp_data[jp_key]
            tx, ty = v_info['target_px']
            label_text = f"/{jp_key}/ ({int(f1)}, {int(f2)})  "
            dx, dy = offsets.get(jp_key, (30, 30))
            draw_final.text((tx + dx, ty + dy), label_text, fill=(0, 0, 0, 255), font=font)
    return base_img

# --- 4. Session State 管理 ---
if 'stage' not in st.session_state:
    st.session_state.stage = "JP_CALIB"
if 'jp_data' not in st.session_state:
    st.session_state.jp_data = {}
if 'g_key' not in st.session_state:
    st.session_state.g_key = "female" # 預設值

# --- 5. UI 介面 ---
st.title("英語母音發音視覺回饋系統")

# 直接取得當前內部的 g_key
g_key = st.session_state.g_key

# --- 第一階段：日文母音校正 ---
if st.session_state.stage == "JP_CALIB":
    st.subheader("第一階段：日文母音基準校正")
    
    current_keys = list(st.session_state.jp_data.keys())
    progress = len(current_keys)
    
    cols_status = st.columns([3, 1])
    with cols_status[0]:
        st.info("請錄製日文的「あいうえお」，系統將根據您的聲音自動調整對比基準。")
    with cols_status[1]:
        st.metric("目前進度", f"{progress} / 5")
    st.progress(progress / 5)

    col_j1, col_j2 = st.columns(2)
    
    with col_j1:
        selected_jp = st.selectbox("Step 1: 請選擇一種母音並聽取▶️下方示範音檔：", list(JP_VOWELS.keys()))
        jp_v = JP_VOWELS[selected_jp]
        img = draw_static_target(jp_v['ref_img'], jp_v['target_px'])
        if img:
            st.image(img, width=350, caption=f"日文「{selected_jp}」預期位置")
        st.audio(f"assets/{jp_v['audio']}")
        
    with col_j2:
        # 1. 顯示該音「歷史已存」的數據
        if jp_v['key'] in st.session_state.jp_data:
            f1_saved, f2_saved = st.session_state.jp_data[jp_v['key']]
            st.success(f"✅ {selected_jp} 已錄製完成")
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("基準 F1", f"{int(f1_saved)} Hz")
            m_col2.metric("基準 F2", f"{int(f2_saved)} Hz")
        
        st.write("---")
        
        # 2. 錄音組件
        rec_j = mic_recorder(
            start_prompt=f"Step 2: 🎙️錄製你的發音 {selected_jp}", 
            stop_prompt="⏹️停止錄音並分析", 
            key=f"rec_jp_{jp_v['key']}"
        )
        
        # 3. 處理新錄音：顯示臨時結果與確認按鍵
        if rec_j:
            f_list = get_formants(rec_j['bytes'])
            if len(f_list) >= 2:
                new_f1, new_f2 = f_list[0], f_list[1]
                
                # 只有當錄到的數據還沒被存入 jp_data 時，才顯示「剛分析完成」介面
                # 判斷標準：如果 jp_data 裡沒資料，或資料跟剛錄到的不完全一樣
                if jp_v['key'] not in st.session_state.jp_data or \
                   abs(st.session_state.jp_data[jp_v['key']][0] - new_f1) > 0.1:
                    
                    st.write("✨ **剛分析完成：**")
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("分析 F1", f"{int(new_f1)} Hz")
                    res_col2.metric("分析 F2", f"{int(new_f2)} Hz")
                    
                    # 恢復按鍵：點擊後才正式寫入 session_state
                    if st.button("確定並存入基準", key=f"save_{jp_v['key']}", type="primary"):
                        # 這裡才真正執行存入動作
                        st.session_state.jp_data[jp_v['key']] = (new_f1, new_f2)
                        
                        # 自動性別判定邏輯
                        if jp_v['key'] == 'i':
                            st.session_state.g_key = "male" if new_f1 < 290 else "female"
                        
                        st.balloons()
                        st.rerun()

    st.divider()
    
    # 底部解鎖邏輯與地圖顯示
    if len(st.session_state.jp_data) >= 5:
        st.success("🎉 您已完成所有日文校正。")
        final_map_img = draw_final_jp_map(st.session_state.jp_data)
        if final_map_img:
            st.image(final_map_img, width=600, caption="個人化母音地圖")
        
        if st.button("🔓 進入英文挑戰階段 ➔", type="primary", use_container_width=True):
            st.session_state.stage = "EN_LEVEL"
            st.rerun()
    else:
        st.warning(f"還差 {5 - progress} 個音即可解鎖。")

# --- 第二階段：英文母音練習 ---
else:
    st.subheader("第二階段：挑戰英文母音")
    col_target, col_practice = st.columns(2)
    with col_target:
        selected_en = st.selectbox("請選擇挑戰母音：", list(VOWEL_MAP.keys()))
        en_v = VOWEL_MAP[selected_en]
        ipa_symbol = selected_en.split(" ")[0]
        st.markdown(f"### 目標音：`/{ipa_symbol}/`")
        img = draw_static_target(f"{en_v['prefix']}_{en_v['v_key']}_full.png", en_v['target_px'])
        if img: st.image(img, width=350, caption=f"/{ipa_symbol}/ 預期位置")
        st.write("🔊 聽聽示範：")
        word_choice = st.radio("選擇單字：", en_v["words"], horizontal=True, key="en_word")
        st.audio(f"assets/{en_v['prefix']}_{en_v['v_key']}_{word_choice}.mp3")
    with col_practice:
        st.markdown("### 🎙️ 開始練習")
        jp_key = en_v['jp_ref']
        my_jp_ref = st.session_state.jp_data.get(jp_key)
        avg_ref = en_v['ref'][g_key]
        if not my_jp_ref: st.error("請返回第一階段錄製日文音。")
        else:
            rec_en = mic_recorder(start_prompt="請點擊並發音", key=f"rec_en_{en_v['v_key']}")
            if rec_en:
                f_en = get_formants(rec_en['bytes'])
                if len(f_en) >= 2:
                    f1, f2 = f_en[0], f_en[1]
                    res_img = draw_overlay(en_v, f1, f2, g_key, my_jp_ref)
                    if res_img: st.image(res_img, width=400, caption="紅色紅點為您的位置")
                    st.divider()
                    st.subheader("建議")
                    if avg_ref['range_f1'][0] <= f1 <= avg_ref['range_f1'][1]: st.success("✅ 舌位高低標準！")
                    elif f1 < avg_ref['range_f1'][0]: st.warning("❌ 舌頭太高，嘴巴張大點。")
                    else: st.warning("❌ 舌頭太低，抬高舌頭。")
                    if avg_ref['range_f2'][0] <= f2 <= avg_ref['range_f2'][1]: st.success("✅ 舌位前後標準！")
                    elif f2 < avg_ref['range_f2'][0]: st.warning("❌ 舌頭太後，往前推一點。")
                    else: st.warning("❌ 舌頭太前，稍微後縮。")
                    ref_jp_f1, ref_jp_f2 = my_jp_ref
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("您的 F1", f"{int(f1)} Hz", f"{int(f1 - ref_jp_f1)} Hz")
                    m_col2.metric("您的 F2", f"{int(f2)} Hz", f"{int(f2 - ref_jp_f2)} Hz")
    if st.button("⬅️ 返回日文校正階段"):
        st.session_state.stage = "JP_CALIB"
        st.rerun()

# --- 頁面底部：重新開始按鈕 ---
st.divider()
if st.button("🔄 重新開始全部流程 (清除所有數據)"):
    st.session_state.stage = "JP_CALIB"
    st.session_state.jp_data = {}
    st.session_state.g_key = "female"
    st.rerun()
