import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
from PIL import Image, ImageDraw
import io
import librosa
import numpy as np
import scipy.signal
from pathlib import Path

# --- 1. 資料結構定義 ---

# 日文母音設定 (與英文圖檔對應)
JP_VOWELS = {
    "あ (a)": {"key": "a", "ref_img": "08_script_a_full.png", "audio": "japanese_a.mp3"},
    "い (i)": {"key": "i", "ref_img": "01_high_i_full.png", "audio": "japanese_i.mp3"},
    "う (u)": {"key": "u", "ref_img": "05_high_u_full.png", "audio": "japanese_u.mp3"},
    "え (e)": {"key": "e", "ref_img": "03_epsilon_full.png", "audio": "japanese_e.mp3"},
    "お (o)": {"key": "o", "ref_img": "07_open_o_full.png", "audio": "japanese_o.mp3"},
}

# 英文母音設定 (沿用您之前的 VOWEL_MAP)
VOWEL_MAP = {
    "i (eat)": {"prefix": "01", "v_key": "high_i", "target_px": (196, 115), "jp_ref": "i"},
    "ɛ (bed)": {"prefix": "03", "v_key": "epsilon", "target_px": (204, 136), "jp_ref": "e"},
    "u (too)": {"prefix": "05", "v_key": "high_u", "target_px": (247, 109), "jp_ref": "u"},
    "ɔ (dog)": {"prefix": "07", "v_key": "open_o", "target_px": (247, 146), "jp_ref": "o"},
    "ɑ (box)": {"prefix": "08", "v_key": "script_a", "target_px": (241, 157), "jp_ref": "a"},
    # 02, 04, 06 可依此類推
}

# --- 2. 初始化 Session State ---
if 'stage' not in st.session_state:
    st.session_state.stage = "JP_CALIB"  # 初始階段：校正
if 'jp_data' not in st.session_state:
    st.session_state.jp_data = {} # 儲存個人的日文母音座標

# --- 3. 核心工具函式 ---

def get_formants(audio_bytes):
    # 自動偵測性別傾向 (簡易版：根據平均頻率判斷)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes)).set_channels(1).set_frame_rate(22050)
    y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    y = scipy.signal.lfilter([1, -0.63], [1], y)
    lpc_coeffs = librosa.lpc(y, order=12 + 2)
    roots = np.roots(lpc_coeffs)
    roots = [r for r in roots if np.imag(r) > 0]
    angz = np.arctan2(np.imag(roots), np.real(roots))
    formants = sorted(angz * (22050 / (2 * np.pi)))
    return [f for f in formants if f > 200]

def draw_visual_feedback(target_v_info, current_f1, current_f2, jp_f1f2=None):
    base_path = Path("assets") / f"{target_v_info['prefix']}_{target_v_info['v_key']}_full.png"
    img = Image.open(base_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    tx, ty = target_v_info["target_px"]

    # 1. 繪製目標點 (紅色圓圈)
    draw.ellipse([tx-8, ty-8, tx+8, ty+8], outline="red", width=3)

    # 2. 如果有該生的日文對應點 (灰色半透明，作為基準)
    if jp_f1f2:
        # 這裡的公式可以根據你的座標系微調，暫定相對位移
        # jx = tx - ((jp_f1f2[1] - target_ref_f2) * 0.08) ...
        # 為了簡化，我們先顯示目前的「發音點」
        pass 

    # 3. 繪製學生目前的發音點 (紅色實心點)
    # 這裡建議您定義一個簡單的線性轉換，將 F1/F2 轉為圖上的像素
    st_x = tx # 暫代，需根據 F2 差值計算
    st_y = ty # 暫代，需根據 F1 差值計算
    draw.ellipse([st_x-10, st_y-10, st_x+10, st_y+10], fill=(255, 0, 0, 180))
    
    return img

def get_stars(diff_f1, diff_f2):
    total_diff = abs(diff_f1) + (abs(diff_f2) / 2)
    if total_diff < 100: return "⭐⭐⭐ (Perfect!)"
    if total_diff < 250: return "⭐⭐ (Good!)"
    return "⭐ (Keep trying!)"

# --- 4. UI 邏輯 ---

st.title("👅 英語母音發音視覺回饋系統")

# --- 第一階段：日文母音校正 ---
if st.session_state.stage == "JP_CALIB":
    st.subheader("第一階段：日文母音校正 (Calibration)")
    st.write("請依序練習日文「あいうえお」，這將作為您練習英文時的基準點。")
    
    selected_jp = st.selectbox("選擇校正母音：", list(JP_VOWELS.keys()))
    jp_v = JP_VOWELS[selected_jp]
    
    c1, c2 = st.columns(2)
    with c1:
        st.image(f"assets/{jp_v['ref_img']}", width=300, caption=f"日文 {selected_jp} 的位置")
        st.audio(f"assets/{jp_v['audio']}")
    
    with c2:
        rec = mic_recorder(start_prompt=f"錄製 {selected_jp}", stop_prompt="分析中...", key=f"rec_{jp_v['key']}")
        if rec:
            f = get_formants(rec['bytes'])
            if len(f) >= 2:
                st.session_state.jp_data[jp_v['key']] = (f[0], f[1])
                st.success(f"已記錄 {selected_jp} 的特徵：F1={int(f[0])}, F2={int(f[1])}")
    
    st.divider()
    if len(st.session_state.jp_data) >= 5:
        if st.button("我已完成五個母音校正，進入英文挑戰 ➔"):
            st.session_state.stage = "EN_LEVEL"
            st.rerun()

# --- 第二階段：英文母音挑戰 ---
else:
    st.subheader("第二階段：英文母音挑戰")
    
    # 在側邊欄或上方提供切換
    selected_en = st.selectbox("請選擇要練習的英文母音：", list(VOWEL_MAP.keys()))
    en_v = VOWEL_MAP[selected_en]
    
    # 取得對應的日文基準 (用於診斷或視覺對比)
    jp_key = en_v['jp_ref']
    my_jp_ref = st.session_state.jp_data.get(jp_key)

    # 建立兩欄：左邊放目標與示範，右邊放錄音與回饋
    col_target, col_practice = st.columns(2)
    
    with col_target:
        st.markdown(f"### 目標音：`/{en_v['v_key']}/`")
        # 顯示目標發音圖
        st.image(f"assets/{en_v['prefix']}_{en_v['v_key']}_full.png", 
                 width=350, caption=f"標準 /{en_v['v_key']}/ 的舌位圖")
        
        # --- 補回：播放示範音檔 ---
        st.write("👂 **聽聽看標準發音：**")
        # 假設您的英文音檔命名規則為：01_high_i_eat.mp3 (對應 prefix_vkey_word.mp3)
        audio_path = f"assets/{en_v['prefix']}_{en_v['v_key']}_{en_v.get('word', 'audio')}.mp3"
        try:
            st.audio(audio_path)
        except:
            st.warning(f"找不到音檔：{audio_path}")
    
    with col_practice:
        st.markdown("### 2. 錄音練習")
        rec = mic_recorder(
            start_prompt="按住錄音 🎤 (請發長音)", 
            stop_prompt="停止並分析 ⏹️", 
            key=f'en_rec_{en_v["v_key"]}'
        )
        
        if rec:
            f_list = get_formants(rec['bytes'])
            if len(f_list) >= 2:
                f1, f2 = f_list[0], f_list[1]
                
                # 這裡可以根據您之前提供的 VOWEL_MAP['ref'] 進行分數計算
                # 暫時用星星顯示效果
                st.subheader(f"本次評分：{get_stars(50, 50)}") 
                
                # 繪製回饋圖 (傳入學生的日文基準點 my_jp_ref)
                res_img = draw_visual_feedback(en_v, f1, f2, my_jp_ref)
                st.image(res_img, width=400, caption="🔴 紅圈：目標位置 | 🔴 實心點：您的發音位置")
                
                # 顯示詳細數據
                st.metric("當前 F1 (高低)", f"{int(f1)} Hz")
                st.metric("當前 F2 (前後)", f"{int(f2)} Hz")
                
                # 如果有日文基準，可以給出更具體的建議
                if my_jp_ref:
                    jp_f1, jp_f2 = my_jp_ref
                    st.info(f"💡 提示：比起您平常發日文「{jp_key}」音的時候，您的舌頭位置...")

    # 底部導航
    st.divider()
    if st.button("⬅️ 返回第一階段 (重新校正日文母音)"):
        st.session_state.stage = "JP_CALIB"
        st.rerun()
