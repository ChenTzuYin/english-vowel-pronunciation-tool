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
    
    current_keys = list(st.session_state.jp_data.keys())
    progress = len(current_keys)
    
    # UI 狀態顯示
    cols_status = st.columns([3, 1])
    with cols_status[0]:
        st.info("請錄製日文的「あいうえお」，系統將根據您的聲音自動調整對比基準。")
    with cols_status[1]:
        st.metric("目前進度", f"{progress} / 5")
    st.progress(progress / 5)

    col_j1, col_j2 = st.columns(2)
    
    with col_j1:
        selected_jp = st.selectbox("Step 1: 請選擇一種母音並聽取下方示範音檔：", list(JP_VOWELS.keys()))
        jp_v = JP_VOWELS[selected_jp]
        jp_target_img = draw_static_target(jp_v['ref_img'], jp_v['target_px'])
        if jp_target_img:
            st.image(jp_target_img, width=350, caption=f"日文「{selected_jp}」的舌面最高點預期位置")
        else:
            st.image(f"assets/{jp_v['ref_img']}", width=350)
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
            start_prompt=f"Step 2: 錄製你的發音 {selected_jp}", 
            stop_prompt="停止錄音並分析", 
            key=f"rec_jp_{jp_v['key']}"
        )
        
        # 3. 處理新錄音（加入一個簡單判斷，避免重複顯示）
        if rec_j:
            f_list = get_formants(rec_j['bytes'])
            if len(f_list) >= 2:
                new_f1, new_f2 = f_list[0], f_list[1]
                
                # 如果目前的數值跟已存數值不同，或者根本還沒存過，才顯示「剛剛錄製」
                # 這樣按下「確認並繼續」刷新後，因為 jp_data 已經更新，這段就不會再跑出來
                if jp_v['key'] not in st.session_state.jp_data or \
                   abs(st.session_state.jp_data[jp_v['key']][0] - new_f1) > 0.1:
                    
                    st.write("✨ **剛分析完成：**")
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("分析 F1", f"{int(new_f1)} Hz")
                    res_col2.metric("分析 F2", f"{int(new_f2)} Hz")
                    
                    if st.button("確認並存入基準", key=f"save_{jp_v['key']}", type="primary"):
                        st.session_state.jp_data[jp_v['key']] = (new_f1, new_f2)
                        st.balloons()
                        st.rerun()

    st.divider()
    
    # 底部解鎖邏輯
    if len(st.session_state.jp_data) >= 5:
        st.success("🎉 太棒了！您已完成所有日文基準校正。")
        if st.button("🔓 點此進入英文挑戰階段 ➔", type="primary", use_container_width=True):
            st.session_state.stage = "EN_LEVEL"
            st.rerun()
    else:
        st.warning(f"還差 {5 - len(st.session_state.jp_data)} 個音即可解鎖英文挑戰。")
    


# --- 第二階段：英文母音練習 ---
else:
    st.subheader("第二階段：挑戰英文母音")
     
    col_target, col_practice = st.columns(2)

    with col_target:
        selected_en = st.selectbox("請選擇挑戰母音：", list(VOWEL_MAP.keys()))
        en_v = VOWEL_MAP[selected_en]
        
        # --- 顯示專業 IPA 音標 ---
        ipa_symbol = selected_en.split(" ")[0]
        st.markdown(f"### 目標音：`/{ipa_symbol}/`")
        
        en_img_name = f"{en_v['prefix']}_{en_v['v_key']}_full.png"
        en_target_img = draw_static_target(en_img_name, en_v['target_px'])
        
        if en_target_img:
            st.image(en_target_img, width=350, caption=f"紅圈處為 /{ipa_symbol}/ 的舌面最高點預期位置")
        else:
            st.image(f"assets/{en_img_name}", width=350)
        
        st.write("👂 **聽聽示範音檔：**")
        word_choice = st.radio("選擇單字：", en_v["words"], horizontal=True, key="en_word")
        st.audio(f"assets/{en_v['prefix']}_{en_v['v_key']}_{word_choice}.mp3")

    with col_practice:
        st.markdown("### 🎤 開始練習")
        
        # 取得本人對應的日文基準值 (用於計算 Delta)
        jp_key = en_v['jp_ref']
        my_jp_ref = st.session_state.jp_data.get(jp_key)
        
        # 取得該性別的英語平均參考範圍 (用於診斷建議)
        avg_ref = en_v['ref'][g_key]
        
        if not my_jp_ref:
            st.error(f"找不到對應的日文基準音「{jp_key}」，請返回第一階段錄音。")
        else:
            rec_en = mic_recorder(start_prompt="請點擊並發音", key=f"rec_en_{en_v['v_key']}")
            
            if rec_en:
                f_en = get_formants(rec_en['bytes'])
                if len(f_en) >= 2:
                    f1, f2 = f_en[0], f_en[1]
                    
                    # 1. 繪製視覺回饋圖 (傳入本人日文基準做參考點)
                    res_img = draw_overlay(en_v, f1, f2, g_key, my_jp_ref)
                    if res_img:
                        st.image(res_img, width=400, caption=f"對比您的日文「{jp_key}」基準位置")
                    
                    st.divider()
                    st.subheader("📊 診斷建議")
                    
                    # 2. 計算與「英語標準平均值」的差異 (而非單純比日文)
                    f1_diff_avg = f1 - avg_ref['f1']
                    f2_diff_avg = f2 - avg_ref['f2']
                    
                    # --- F1 (舌位高低) 診斷 ---
                    if avg_ref['range_f1'][0] <= f1 <= avg_ref['range_f1'][1]:
                        st.success("✅ **舌位高低：** 非常標準！落在標準範圍內。")
                    elif f1 < avg_ref['range_f1'][0]:
                        st.warning(f"❌ **舌位高低：** 舌頭太高了。建議：**舌頭再放低一點或者嘴巴張大一點**。")
                    else:
                        st.warning(f"❌ **舌位高低：** 舌頭太低了。建議：**舌頭再抬高一點或者嘴巴閉小一點**。")

                    # --- F2 (舌位前後) 診斷 ---
                    if avg_ref['range_f2'][0] <= f2 <= avg_ref['range_f2'][1]:
                        st.success("✅ **舌位前後：** 非常標準！落在標準範圍內。")
                    elif f2 < avg_ref['range_f2'][0]:
                        st.warning(f"❌ **舌位前後：** 舌頭太靠後了。建議：**舌頭再往前推一點**。")
                    else:
                        st.warning(f"❌ **舌位前後：** 舌頭太靠前了。建議：**舌頭稍微向後縮一點**。")

                    # 3. 數值化顯示 (顯示相對於「本人日文基準」的位移量，這對教學最有意義)
                    ref_jp_f1, ref_jp_f2 = my_jp_ref
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric(
                        label=f"您的 F1)", 
                        value=f"{int(f1)} Hz", 
                        delta=f"{int(f1 - ref_jp_f1)} Hz",
                        delta_color="normal"
                    )
                    m_col2.metric(
                        label=f"您的 F2)", 
                        value=f"{int(f2)} Hz", 
                        delta=f"{int(f2 - ref_jp_f2)} Hz"
                    )
                    st.caption(f"💡 目標範圍 (Hz)：F1({avg_ref['range_f1'][0]}-{avg_ref['range_f1'][1]}), F2({avg_ref['range_f2'][0]}-{avg_ref['range_f2'][1]})")

    if st.button("⬅️ 返回日文校正階段"):
        st.session_state.stage = "JP_CALIB"
        st.rerun()
