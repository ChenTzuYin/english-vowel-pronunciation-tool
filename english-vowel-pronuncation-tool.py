import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont
import io
import librosa
import numpy as np
import scipy.signal
from pathlib import Path
import matplotlib.pyplot as plt
import librosa.display

# --- 1. 初期設定 (Initial Settings) ---
st.set_page_config(page_title="英語母音発音視覚化システム", layout="wide")

# --- 2. データ構造 (Data Structures) ---
VOWEL_MAP = {
    "i (eat/see)": {
        "prefix": "01", "v_key": "high_i", "target_px": (196, 115), "jp_ref": "i",
        "words": ["eat", "see"], "t_suffix": "tougue",
        "ref": {
            "female": {"f1": 310, "f2": 2790, "range_f1": (220, 430), "range_f2": (2000, 3500)},
            "male": {"f1": 270, "f2": 2290, "range_f1": (190, 380), "range_f2": (1600, 2900)}
        }
    },
    "eɪ (ate/say)": {
        "prefix": "02", "v_key": "ei", "target_px": (196, 122), "jp_ref": "e",
        "words": ["ate", "say"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 480, "f2": 2400, "range_f1": (370, 580), "range_f2": (1700, 2900)},
            "male": {"f1": 400, "f2": 2100, "range_f1": (320, 480), "range_f2": (1500, 2600)}
        }
    },
    "ɛ (bed/egg)": {
        "prefix": "03", "v_key": "epsilon", "target_px": (204, 136), "jp_ref": "e",
        "words": ["bed", "egg"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 610, "f2": 2330, "range_f1": (520, 730), "range_f2": (1500, 2700)},
            "male": {"f1": 530, "f2": 1840, "range_f1": (420, 630), "range_f2": (1200, 2300)}
        }
    },
    "æ (bad/cat)": {
        "prefix": "04", "v_key": "ash", "target_px": (214, 153), "jp_ref": "a",
        "words": ["bad", "cat"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 860, "f2": 2050, "range_f1": (720, 1030), "range_f2": (1400, 2500)},
            "male": {"f1": 660, "f2": 1720, "range_f1": (570, 830), "range_f2": (1100, 2200)}
        }
    },
    "u (too/zoo)": {
        "prefix": "05", "v_key": "high_u", "target_px": (247, 109), "jp_ref": "u",
        "words": ["too", "zoo"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 370, "f2": 950, "range_f1": (270, 480), "range_f2": (400, 1500)},
            "male": {"f1": 300, "f2": 870, "range_f1": (220, 410), "range_f2": (400, 1400)}
        }
    },
    "oʊ (go/no)": {
        "prefix": "06", "v_key": "ou", "target_px": (259, 134), "jp_ref": "o",
        "words": ["go", "no"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 500, "f2": 1000, "range_f1": (420, 630), "range_f2": (500, 1600)},
            "male": {"f1": 450, "f2": 900, "range_f1": (350, 550), "range_f2": (450, 1450)}
        }
    },
    "ɔ (dog/law)": {
        "prefix": "07", "v_key": "open_o", "target_px": (247, 146), "jp_ref": "o",
        "words": ["dog", "law"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 700, "f2": 1100, "range_f1": (620, 830), "range_f2": (600, 1700)},
            "male": {"f1": 570, "f2": 840, "range_f1": (490, 680), "range_f2": (350, 1300)}
        }
    },
    "ɑ (box/hot)": {
        "prefix": "08", "v_key": "script_a", "target_px": (241, 157), "jp_ref": "a",
        "words": ["box", "hot"], "t_suffix": "tongue",
        "ref": {
            "female": {"f1": 850, "f2": 1220, "range_f1": (720, 1030), "range_f2": (700, 1800)},
            "male": {"f1": 730, "f2": 1090, "range_f1": (620, 880), "range_f2": (600, 1600)}
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

def draw_overlay(en_v, f1, f2, g_key):
    """
    英語の標準値を基準に、発音位置とターゲット（目標）を画像上に描画する関数。
    """
    # 1. ターゲットの標準値と範囲を取得
    avg_ref = en_v['ref'][g_key]
    target_f1 = avg_ref['f1']
    target_f2 = avg_ref['f2']
    range_f1 = avg_ref['range_f1']
    range_f2 = avg_ref['range_f2']
    
    # 画像上のターゲット座標 (目標地点の中心)
    tx, ty = en_v['target_px']
    
    # 2. スナップ効果（合格範囲内なら中心に吸い付く）
    if range_f1[0] <= f1 <= range_f1[1]:
        diff_f1 = 0
    else:
        diff_f1 = f1 - target_f1
        
    if range_f2[0] <= f2 <= range_f2[1]:
        diff_f2 = 0
    else:
        diff_f2 = f2 - target_f2

    # 3. ユーザーの描画位置を計算
    st_x = tx - (diff_f2 * 0.05) 
    st_y = ty + (diff_f1 * 0.1)

    # --- 画像描画処理 ---
    # 背景画像を読み込む
    img_path = Path(f"assets/{en_v['prefix']}_{en_v['v_key']}_full.png")
    if not img_path.exists():
        return None
        
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # A. ターゲット（目標値）を「中抜きの赤い円」で描画
    target_r = 15  # 目標の円の大きさ
    draw.ellipse([tx - target_r, ty - target_r, tx + target_r, ty + target_r], 
                 outline="red", width=3)

    # B. ユーザーの現在位置を「赤い塗りつぶし円」で描画
    user_r = 10   # ユーザーの点の大きさ
    draw.ellipse([st_x - user_r, st_y - user_r, st_x + user_r, st_y + user_r], 
                 fill="red", outline="white", width=2)
    
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

def plot_voice_analysis(y, sr):
    """
    音声信号の波形とスペクトログラムを描画する関数
    """
    # グラフの作成 (2段構成：上が波形、下がスペクトログラム)
    fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    
    # --- 1. 波形図 (Waveform) ---
    # 実際にどの区間の音声が切り出されたかを確認します
    librosa.display.waveshow(y, sr=sr, ax=ax[0], color='blue')
    ax[0].set_title("Waveform", fontsize=8)
    ax[0].set_ylabel("Raw Amplitude (16-bit PCM)")

    # --- 2. スペクトログラム (Spectrogram) ---
    # 短時間フーリエ変換 (STFT) を行い、周波数分布を表示
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], cmap='magma')
    ax[1].set_title("FFT Spectrogram", fontsize=8)
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].set_ylim(0, 8000)  # 母音分析に重要な 8kHz までを表示
    
    # カラーバーの追加（デシベル強度）
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    plt.tight_layout()
    return fig

if 'stage' not in st.session_state:
    st.session_state.stage = "JP_CALIB"
if 'jp_data' not in st.session_state:
    st.session_state.jp_data = {}
if 'g_key' not in st.session_state:
    st.session_state.g_key = "female"

st.title("発音ビジュアル・フィードバック")
g_key = st.session_state.g_key

if st.session_state.stage == "JP_CALIB":
    st.subheader("ステップ1：日本語母音をどうやって発音するか？")
    current_keys = list(st.session_state.jp_data.keys())
    progress = len(current_keys)
    cols_status = st.columns([3, 1])
    with cols_status[0]: st.info("順番で「あいうえお」を録音してください。")
    with cols_status[1]: st.metric("現在の進捗", f"{progress} / 5")
    st.progress(progress / 5)
    col_j1, col_j2 = st.columns(2)
    with col_j1:
        selected_jp = st.selectbox("1. 一つの母音を選んで、▶️お手本を聞いてください：", list(JP_VOWELS.keys()))
        jp_v = JP_VOWELS[selected_jp]
        img = draw_static_target(jp_v['ref_img'], jp_v['target_px'])
        if img: st.image(img, width=350, caption=f"日本語「{selected_jp}」の舌の位置")
        st.audio(f"assets/{jp_v['audio']}")
    with col_j2:
        if jp_v['key'] in st.session_state.jp_data:
            f1_saved, f2_saved = st.session_state.jp_data[jp_v['key']]
            st.success(f"✅ {selected_jp} は録音済みです")
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("第一フォルマント（舌の高さ）", f"{int(f1_saved)} Hz")
            m_col2.metric("第二フォルマント（舌の前後）", f"{int(f2_saved)} Hz")
        st.write("---")
        rec_j = mic_recorder(start_prompt=f"2. {selected_jp} を録音する 🎙️", stop_prompt="録音を止めて解析する ⏹️", key=f"rec_jp_{jp_v['key']}")
        if rec_j:
            f_list = get_formants(rec_j['bytes'])
            if len(f_list) >= 2:
                new_f1, new_f2 = f_list[0], f_list[1]
                if jp_v['key'] not in st.session_state.jp_data or abs(st.session_state.jp_data[jp_v['key']][0] - new_f1) > 0.1:
                    st.write("✨ **解析結果：**")
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("現在の第一フォルマント（舌の高さ）", f"{int(new_f1)} Hz")
                    res_col2.metric("現在の第二フォルマント（舌の前後）", f"{int(new_f2)} Hz")
                    if st.button("確定して基準として保存", key=f"save_{jp_v['key']}", type="primary"):
                        st.session_state.jp_data[jp_v['key']] = (new_f1, new_f2)
                        if jp_v['key'] == 'i':
                            st.session_state.g_key = "male" if new_f1 < 300 else "female"
                        st.balloons()
                        st.rerun()

    st.divider()
    if len(st.session_state.jp_data) >= 5:
        st.success("🎉 すべての母音が録音できました！")
        final_map_img = draw_final_jp_map(st.session_state.jp_data)
        if final_map_img: st.image(final_map_img, width=350, caption="あなたの母音地図です")
        if st.button("🔓 英語母音の練習へ進む ➔", type="primary", use_container_width=True):
            st.session_state.stage = "EN_LEVEL"
            st.rerun()
    else:
        st.warning(f"あと {5 - progress} 個の音を録音すると、英語母音の練習へアクセスできます。")

else:
    st.subheader("ステップ2：英語母音のトレーニング")
    col_target, col_practice = st.columns(2)
    with col_target:
        selected_en = st.selectbox("練習する母音を選んでください：", list(VOWEL_MAP.keys()))
        en_v = VOWEL_MAP[selected_en]
        ipa_symbol = selected_en.split(" ")[0]
        st.markdown(f"### 今練習している母音は：`/{ipa_symbol}/`")
        img = draw_static_target(f"{en_v['prefix']}_{en_v['v_key']}_full.png", en_v['target_px'])
        if img: st.image(img, width=350, caption=f"/{ipa_symbol}/ の舌の位置")
        st.write("好きな單語を選んで、▶️お手本を聞いてください：")
        word_choice = st.radio("", en_v["words"], horizontal=True, key="en_word")
        st.audio(f"assets/{en_v['prefix']}_{en_v['v_key']}_{word_choice}.mp3")
    with col_practice:
        st.markdown("### 🎙️ 録音してください")
        jp_key = en_v['jp_ref']
        my_jp_ref = st.session_state.jp_data.get(jp_key)
        avg_ref = en_v['ref'][g_key]
        if not my_jp_ref:
            st.error("日本語の母音データが見つかりません。ステップ1に戻ってください。")
        else:
            rec_en = mic_recorder(start_prompt="🎙️ クリックして錄音", stop_prompt="⏹️ 錄音を止める", key="fixed_en_mic_recorder")
            if rec_en:
                f_en = get_formants(rec_en['bytes'])
                if len(f_en) >= 2:
                    f1, f2 = f_en[0], f_en[1]
                    res_img = draw_overlay(en_v, f1, f2, g_key)
                    if res_img:
                        st.image(res_img, width=350, caption="赤点は現在の舌の最も高い位置を推定したものです。")
                    
                    st.divider()
                    st.subheader("📊 アドバイス")
                    
                    target_f1 = avg_ref['f1']
                    target_f2 = avg_ref['f2']
                    
                    if avg_ref['range_f1'][0] <= f1 <= avg_ref['range_f1'][1]:
                        st.success("✅ **舌の高さ：** バッチリです！理想的な範囲内です。")
                        f1_diff_display = "OK!"
                        f1_delta_color = "normal"
                    else:
                        f1_diff = f1 - target_f1
                        if f1 < avg_ref['range_f1'][0]:
                            st.warning("❌ **舌の高さ：** 舌の位置が高すぎます。もう少し口を大きく開けてみましょう。")
                        else:
                            st.warning("❌ **舌の高さ：** 舌の位置が低すぎます。もう少し舌を上に持ち上げてみましょう。")
                        f1_diff_display = f"{round(f1_diff, 1)} Hz"
                        f1_delta_color = "inverse"

                    if avg_ref['range_f2'][0] <= f2 <= avg_ref['range_f2'][1]:
                        st.success("✅ **舌の前後：** バッチリです！理想的な位置です。")
                        f2_diff_display = "OK!"
                        f2_delta_color = "normal"
                    else:
                        f2_diff = f2 - target_f2
                        if f2 < avg_ref['range_f2'][0]:
                            st.warning("❌ **舌の前後：** 舌が後ろに下がりすぎです。もう少し前に出してみましょう。")
                        else:
                            st.warning("❌ **舌の前後：** 舌が前に出すぎです。少し後ろに下げてみましょう。")
                        f2_diff_display = f"{round(f2_diff, 1)} Hz"
                        f2_delta_color = "normal"

                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric(
                        label="現在の F1 (舌の高さ)", 
                        value=f"{round(f1, 1)} Hz", 
                        delta=f1_diff_display,
                        delta_color=f1_delta_color
                    )
                    m_col2.metric(
                        label="現在の F2 (舌の前後)", 
                        value=f"{round(f2, 1)} Hz", 
                        delta=f2_diff_display,
                        delta_color=f2_delta_color
                    )
                    st.caption(f"💡 英語の標準平均値（F1:{target_f1}, F2:{target_f2}）との差を表示しています。")
                    st.caption(f"🎯 目標範囲: F1({avg_ref['range_f1'][0]}-{avg_ref['range_f1'][1]}), F2({avg_ref['range_f2'][0]}-{avg_ref['range_f2'][1]})")


    
# --- ナビゲーションボタン (Navigation) ---
if st.session_state.stage == "EN_LEVEL":
    if st.button("⬅️ 日本語の母音練習に戻る", key="back_to_jp"):
        st.session_state.stage = "JP_CALIB"
        st.rerun()
        
# エラー回避: 'rec_en' が定義されており、かつデータが存在する場合のみ実行
if 'rec_en' in locals() and rec_en:
    # 英語トレーニングのステージ (EN_LEVEL) でのみ表示するように限定し、
    # 重複描画による DuplicateElementId エラーを防ぎます。
    if st.session_state.stage == "EN_LEVEL":
        f_en = get_formants(rec_en['bytes'])
        
        if len(f_en) >= 2:
            audio_data = AudioSegment.from_file(io.BytesIO(rec_en['bytes'])).set_channels(1).set_frame_rate(22050)
            y_en = np.array(audio_data.get_array_of_samples(), dtype=np.float32) / 32768.0
            sr_en = 22050

            st.divider()
            st.subheader("🎵 参考：音声信号解析 (FFT Analysis)")
                
            col_chart, col_empty = st.columns([1, 1]) # 1:1 分割，圖表佔左邊一半
    
            with col_chart:
                # グラフの描画
                fig = plot_voice_analysis(y_en, sr_en)
                # key を指定することで、Streamlit 内部での ID 重複を確実に防ぎます
                st.pyplot(fig, clear_figure=True)



