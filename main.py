from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import lightgbm as lgb
import pandas as pd
import numpy as np
import requests
from io import StringIO
import re
import time
import random
import pickle
import math
from datetime import datetime

app = FastAPI(title="競馬シミュレーションAPI (プロ仕様完全版: オッズ取得バグ完全修正/地力ブレンド/馬体重マイルド化)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = lgb.Booster(model_file='history_model_pro.txt')

with open('encoders_pro.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('horse_history_db.pkl', 'rb') as f:
    horse_history_db = pickle.load(f)
    
with open('course_history_db.pkl', 'rb') as f:
    course_history_db = pickle.load(f)

def safe_encode(encoder_name, value):
    if encoder_name not in encoders:
        return 0
    le = encoders[encoder_name]
    if value in le.classes_:
        return le.transform([value])[0]
    return 0

def parse_time_to_seconds(time_val):
    if time_val is None or time_val == '-' or time_val == '':
        return None
    s = str(time_val).strip()
    try:
        if ':' in s:
            parts = s.split(':')
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(s)
    except:
        return None

def get_expected_standard_time(c_stats: dict, course_type: str, distance: int) -> float:
    """コースDBにstandard_timeが無い場合、距離と芝/ダートに応じた標準時計を返す"""
    std_t = c_stats.get('standard_time')
    if std_t is not None:
        try:
            val = float(std_t)
            if val > 30.0:
                return val
        except:
            pass
    sec_per_meter = 0.0595 if course_type == "芝" else 0.0620
    return round(distance * sec_per_meter, 1)

def infer_class_factor(record: dict) -> float:
    """
    DBにclass_factorが存在しない場合のフォールバック判定。
    G1/G2等のグレード表記が無い場合でも、主要重賞名を直接判定して正確にクラス係数を付与する。
    """
    if 'class_factor' in record and record['class_factor'] is not None:
        try:
            return float(record['class_factor'])
        except:
            pass
    r_name = str(record.get('race_name', '')) + str(record.get('grade', ''))

    g1_keywords = [
        'G1', 'GI', '優駿', 'ダービー', '皐月賞', '菊花賞', '桜花賞', '秋華賞',
        '天皇賞', '有馬記念', '宝塚記念', 'ジャパンC', 'ジャパンカップ', 'ホープフル',
        '朝日杯', '阪神JF', 'NHKマイル', '安田記念', 'マイルCS', 'スプリンターズ',
        '高松宮記念', 'ヴィクトリア', 'エリザベス女王', '大阪杯', 'フェブラリー', 'チャンピオンズ'
    ]
    g2_keywords = [
        'G2', 'GII', '神戸新聞', 'セントライト', '京都新聞', '青葉賞', '弥生賞',
        'スプリングS', 'ローズS', '紫苑S', 'フローラS', 'チューリップ', '毎日王冠',
        '京都大賞典', 'オールカマー', '札幌記念', '日経賞', '阪神大賞典', '日経新春', '金鯱賞', '東スポ杯'
    ]
    g3_keywords = [
        'G3', 'GIII', '共同通信', 'きさらぎ', '京成杯', 'シンザン', 'フェアリー',
        'クイーンC', '毎日杯', 'フラワーC', 'ファルコン', 'アーリントン', 'ラジオNIKKEI',
        'レパード', 'ユニコーン', '新潟記念', '小倉記念', '七夕賞', 'エプソム'
    ]

    for kw in g1_keywords:
        if kw in r_name: return 1.20
    for kw in g2_keywords:
        if kw in r_name: return 1.10
    for kw in g3_keywords:
        if kw in r_name: return 1.03

    if 'OP' in r_name or 'L' in r_name or 'ステークス' in r_name or 'S' in r_name or '特別' in r_name:
        if '3勝' in r_name or '1600万' in r_name: return 0.90
        if '2勝' in r_name or '1000万' in r_name: return 0.80
        if '1勝' in r_name or '500万' in r_name: return 0.70
        return 0.96
    if '3勝' in r_name or '1600万' in r_name: return 0.90
    if '2勝' in r_name or '1000万' in r_name: return 0.80
    if '1勝' in r_name or '500万' in r_name: return 0.70
    if '未勝利' in r_name or '新馬' in r_name: return 0.60
    return 0.75

def infer_margin_seconds(record: dict) -> float:
    if 'margin' in record and record['margin'] is not None:
        try:
            val = float(record['margin'])
            return max(0.0, val)
        except:
            pass
    try:
        rank = float(record.get('rank', 8.0))
        if rank <= 1.0: return 0.0
        elif rank <= 2.0: return 0.15
        elif rank <= 3.0: return 0.25
        elif rank <= 5.0: return 0.45
        elif rank <= 8.0: return 0.75
        else: return 1.25
    except:
        return 0.75

def calc_jra_wakuban(umaban: int, total_horses: int) -> int:
    if total_horses <= 8:
        return max(1, min(8, umaban))
    double_frames = min(8, total_horses - 8)
    triple_frames = max(0, total_horses - 16)
    single_frames = 8 - double_frames
    if umaban <= single_frames:
        return umaban
    rem = umaban - single_frames
    normal_double_count = double_frames - triple_frames
    if rem <= normal_double_count * 2:
        return single_frames + math.ceil(rem / 2.0)
    rem2 = rem - (normal_double_count * 2)
    return single_frames + normal_double_count + math.ceil(rem2 / 3.0)

def calc_single_pci(course_str, time_val, up_val):
    try:
        m = re.search(r'(\d{3,4})', str(course_str))
        if not m: return None
        dist = int(m.group(1))
        if dist <= 600: return None
        sec = parse_time_to_seconds(time_val)
        up = float(up_val)
        if sec is None or up <= 0 or sec <= up: return None
        first_3f_equiv = ((sec - up) / (dist - 600.0)) * 600.0
        pci = (first_3f_equiv / up) * 50.0
        if 30.0 <= pci <= 75.0: return round(pci, 1)
        return None
    except:
        return None

def get_pci_label(pci_val):
    if pci_val < 48.0: return "A"
    elif pci_val <= 52.0: return "B"
    else: return "C"

def get_horse_pci_stats(bamei_str):
    records = horse_history_db.get(bamei_str, {}).get('race_records', [])
    if not records: return 50.0, "B"
    good_pcis = []
    all_pcis = []
    for r in records:
        pci = calc_single_pci(r.get('course', ''), r.get('time', ''), r.get('up', ''))
        if pci is not None:
            all_pcis.append(pci)
            try:
                rank = float(r.get('rank', 99))
                if 1.0 <= rank <= 3.0: good_pcis.append(pci)
            except: pass
    target_pcis = good_pcis if good_pcis else all_pcis
    if not target_pcis: return 50.0, "B"
    avg_pci = round(float(np.mean(target_pcis)), 1)
    return avg_pci, get_pci_label(avg_pci)

def get_horse_real_stats(bamei_str):
    records = horse_history_db.get(bamei_str, {}).get('race_records', [])
    if not records: return 8.0, 35.0, 0, 0.5, False
    
    shussou_kaisuu = len(records)
    try:
        raw_zenso = float(records[-1].get('rank', 8.0))
    except:
        raw_zenso = 8.0
    
    up_list = []
    for r in records[-3:]:
        try:
            u = float(r.get('up', 0))
            if 30.0 <= u <= 45.0: up_list.append(u)
        except: pass
    avg_agari = round(float(np.mean(up_list)), 2) if up_list else 35.0

    makuri_flag = False
    recent_scores = []
    all_valid_scores = []
    
    scan_records = records[-7:]
    for idx, r in enumerate(scan_records):
        passing = str(r.get('passing', ''))
        if '-' in passing and idx >= len(scan_records) - 5:
            try:
                p_nums = [int(x) for x in passing.split('-') if x.isdigit()]
                if len(p_nums) >= 2 and (p_nums[0] - p_nums[-1]) >= 3:
                    makuri_flag = True
            except: pass

        cf = infer_class_factor(r)
        margin = infer_margin_seconds(r)
        
        course_str = str(r.get('course', ''))
        dist = 2000.0
        m_dist = re.search(r'(\d{3,4})', course_str)
        if m_dist:
            dist = float(m_dist.group(1))
            
        capped_margin = min(max(margin, 0.0), 2.0)
        adjusted_margin = capped_margin * (2000.0 / dist) if dist > 0 else capped_margin
        run_score = float(np.clip(cf - (adjusted_margin * 0.14), 0.10, 1.20))
        all_valid_scores.append(run_score)
        if idx >= len(scan_records) - 5:
            recent_scores.append(run_score)

    if recent_scores:
        weights = np.linspace(1.0, 2.6, len(recent_scores))
        recent_weighted = float(np.average(recent_scores, weights=weights))
        
        sorted_scores = sorted(all_valid_scores, reverse=True)
        if len(sorted_scores) >= 2:
            top2_peak = float(np.mean(sorted_scores[:2]))
        else:
            top2_peak = sorted_scores[0]
            
        power_score = (recent_weighted * 0.75) + (top2_peak * 0.25)
    else:
        power_score = 0.50

    return raw_zenso, avg_agari, shussou_kaisuu, power_score, makuri_flag

def get_course_rpci_stats(course_key):
    c_stats = course_history_db.get(course_key, {})
    pace_f = c_stats.get('pace_first', [])
    pace_l = c_stats.get('pace_last', [])
    avg_pace_f = float(np.mean(pace_f)) if pace_f else 35.0
    avg_pace_l = float(np.mean(pace_l)) if pace_l else 35.0
    rpci = round((avg_pace_f / avg_pace_l) * 50.0, 1) if avg_pace_l > 0 else 50.0
    rpci_type = get_pci_label(rpci)

    if rpci < 48.0: rec_pace = "H_MAEKUZURE"
    elif rpci < 49.8: rec_pace = "M_LONSPA"
    elif rpci <= 52.0: rec_pace = "M_JIRIKI"
    elif rpci <= 53.2: rec_pace = "S_LONSPA"
    else: rec_pace = "S_YOIDON"
    return rpci, rpci_type, rec_pace

def adjust_pace_times(expected_pace, avg_pace_f, avg_pace_l, avg_req_up):
    if expected_pace in ["S_YOIDON", "S"]: return avg_pace_f + 1.4, avg_pace_l - 1.4, avg_req_up - 1.0
    elif expected_pace == "S_LONSPA": return avg_pace_f + 0.8, avg_pace_l - 0.8, avg_req_up - 0.4
    elif expected_pace == "M_LONSPA": return avg_pace_f - 0.4, avg_pace_l + 0.4, avg_req_up + 0.3
    elif expected_pace in ["H_MAEKUZURE", "H"]: return avg_pace_f - 1.4, avg_pace_l + 1.4, avg_req_up + 1.0
    else: return avg_pace_f, avg_pace_l, avg_req_up


# =====================================================================
# 完全修正版: 終了レース(HTML形式) ＆ 未来レース(JavaScript API形式) オッズ取得
# （1列目のResult_Num[着順]や2列目のWaku[枠番]との混同バグを完全排除）
# =====================================================================
def parse_finished_race_html_odds(
    result_html: str, bamei_to_umaban: Optional[Dict[str, int]] = None
) -> tuple[Dict[int, float], Dict[int, tuple[float, float]]]:
    """
    【終了した過去レース用: HTML形式】
    result.html の結果テーブル(<tr class="HorseList">)から「馬番」ごとの確定単勝オッズと馬体重を抽出する。
    ※ 1列目(Result_Num=着順)、2列目(Waku=枠番)を誤って馬番にしないよう、
       3列目(tds[2])および馬名照合(bamei_to_umaban)で厳密に馬番を特定する。
    """
    odds_map: Dict[int, float] = {}
    weight_map: Dict[int, tuple[float, float]] = {}

    if "HorseList" not in result_html:
        return odds_map, weight_map

    if "Result_Num" not in result_html and "Pay_Table" not in result_html and "All_Result_Table" not in result_html:
        return odds_map, weight_map

    rows = re.findall(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', result_html, re.DOTALL | re.IGNORECASE)
    for row in rows:
        try:
            tds = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
            if len(tds) < 8:
                continue

            umaban = 0

            # 方法1: 馬名から出馬表の馬番辞書(bamei_to_umaban)と照合（最も確実・100%ズレない）
            if bamei_to_umaban:
                b_m = re.search(r'<span[^>]*class="[^"]*Horse_Name[^"]*"[^>]*>(.*?)</span>', row, re.IGNORECASE | re.DOTALL)
                if not b_m:
                    b_m = re.search(r'<a[^>]*href="[^"]*horse/\d+[^"]*"[^>]*>(.*?)</a>', row, re.IGNORECASE | re.DOTALL)
                if b_m:
                    b_clean = re.sub(r'<[^>]*>', '', b_m.group(1)).strip()
                    if b_clean in bamei_to_umaban:
                        umaban = bamei_to_umaban[b_clean]

            # 方法2: result.htmlの列構造では [0]=着順(Result_Num), [1]=枠番(Waku), [2]=馬番(Num Txt_C)
            if umaban == 0 and len(tds) >= 3:
                cand_umaban = re.sub(r'<[^>]*>', '', tds[2]).strip()
                if cand_umaban.isdigit():
                    u_val = int(cand_umaban)
                    if 1 <= u_val <= 18:
                        umaban = u_val

            # 方法3: クラス名が "Num Txt_C" (Result_NumやWakuを含まない) のtdから取得
            if umaban == 0:
                u_m = re.search(
                    r'<td[^>]*class="\s*(?:Num\s+Txt_C|Txt_C\s+Num|Umaban)\s*"[^>]*>(.*?)</td>',
                    row, re.IGNORECASE | re.DOTALL
                )
                if u_m:
                    u_str = re.sub(r'<[^>]*>', '', u_m.group(1)).strip()
                    if u_str.isdigit():
                        umaban = int(u_str)

            if umaban == 0:
                continue

            # 単勝オッズの抽出:
            # result.htmlでは <td class="Odds Txt_R"> に単勝オッズ（小数点付き）、<td class="Odds BgYellow">等に人気（整数）が入る
            odds_val = None
            o_m = re.search(r'<td[^>]*class="[^"]*Odds\s+Txt_R[^"]*"[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
            if o_m:
                o_txt = re.sub(r'<[^>]*>', '', o_m.group(1)).strip().replace(',', '')
                m_fl = re.search(r'(\d+\.\d+)', o_txt)
                if m_fl:
                    odds_val = float(m_fl.group(1))

            if odds_val is None:
                # クラスにOddsを含むセルのうち、小数点（\d+\.\d+）を持つセルだけを抽出（整数の「人気」を除外）
                odds_cells = re.findall(r'<td[^>]*class="[^"]*Odds[^"]*"[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
                for cell in odds_cells:
                    c_txt = re.sub(r'<[^>]*>', '', cell).strip().replace(',', '')
                    m_fl = re.search(r'^(\d+\.\d+)$', c_txt)
                    if m_fl:
                        odds_val = float(m_fl.group(1))
                        break

            if odds_val is None and len(tds) >= 11:
                # フォールバック: 11列目(tds[10])が単勝オッズ列（12列目tds[11]は後3Fなので除外）
                for idx_td in [10, 9]:
                    if idx_td < len(tds):
                        c_txt = re.sub(r'<[^>]*>', '', tds[idx_td]).strip().replace(',', '')
                        if re.match(r'^\d+\.\d+$', c_txt):
                            val_f = float(c_txt)
                            if 1.0 <= val_f <= 9999.9:
                                odds_val = val_f
                                break

            if odds_val is not None and odds_val >= 1.0:
                odds_map[umaban] = odds_val

            # 確定馬体重の抽出
            w_m = re.search(r'<td[^>]*class="[^"]*Weight[^"]*"[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
            if w_m:
                w_str = re.sub(r'<[^>]*>', '', w_m.group(1)).strip()
                m_weight = re.search(r'(\d{3})\s*\(([+-]?\d+)\)', w_str)
                if m_weight:
                    weight_map[umaban] = (float(m_weight.group(1)), float(m_weight.group(2)))
        except:
            continue

    return odds_map, weight_map


def fetch_future_race_js_odds(race_id: str, headers: dict) -> tuple[Dict[int, float], str]:
    """
    【未来のレース・馬券購入可能後のレース用: JavaScript動的API形式 ＋ スマホ版オッズページ補完】
    1. api_get_jra_odds.html (action=init / update) から馬番キー("01"〜"18" または "1"〜"18")の単勝オッズを取得
    2. 万が一APIが空の場合は race.sp.netkeiba.com/?pid=odds_view&type=b1 から取得
    """
    odds_map: Dict[int, float] = {}
    js_headers = {
        'User-Agent': headers.get('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
        'Referer': f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',
        'X-Requested-With': 'XMLHttpRequest',
        'Accept': 'application/json, text/javascript, */*; q=0.01'
    }

    api_urls = [
        f"https://race.netkeiba.com/api/api_get_jra_odds.html?pid=api_get_jra_odds&race_id={race_id}&type=1&action=init&sort=odds&compress=0&output=json",
        f"https://race.netkeiba.com/api/api_get_jra_odds.html?pid=api_get_jra_odds&race_id={race_id}&type=1&action=update&compress=0&output=json",
        f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init&compress=0&output=json"
    ]

    api_status_str = ""
    for url in api_urls:
        try:
            time.sleep(0.2)
            resp = requests.get(url, headers=js_headers, timeout=6)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            if not isinstance(payload, dict):
                continue

            status_val = str(payload.get("status", "")).lower()
            data_obj = payload.get("data")
            if not isinstance(data_obj, dict):
                continue

            odds_root = data_obj.get("odds")
            if not isinstance(odds_root, dict):
                continue

            tan_dict = odds_root.get("1")
            if not isinstance(tan_dict, dict):
                continue

            temp_map: Dict[int, float] = {}
            for k_umaban, v_arr in tan_dict.items():
                try:
                    u_num = int(str(k_umaban).strip())
                    raw_o = v_arr[0] if isinstance(v_arr, (list, tuple)) and len(v_arr) > 0 else v_arr
                    o_str = str(raw_o).strip().replace(',', '')
                    if o_str and o_str not in ['---.-', '--.-', '-', '']:
                        o_val = float(o_str)
                        if 1.0 <= o_val <= 9999.9:
                            temp_map[u_num] = o_val
                except:
                    continue

            if len(temp_map) >= 2:
                odds_map = temp_map
                api_status_str = status_val
                break
        except:
            continue

    if len(odds_map) >= 2:
        if api_status_str == "yoso":
            return odds_map, "js_yoso"
        return odds_map, "js_realtime"

    # バックアップ: スマホ版単勝オッズ一覧ページから馬番とオッズを取得
    try:
        time.sleep(0.2)
        sp_url = f"https://race.sp.netkeiba.com/?pid=odds_view&type=b1&race_id={race_id}"
        sp_res = requests.get(sp_url, headers=headers, timeout=6)
        if sp_res.status_code == 200:
            sp_res.encoding = sp_res.apparent_encoding
            sp_map: Dict[int, float] = {}
            tr_blocks = re.findall(r'<tr[^>]*>(.*?)</tr>', sp_res.text, re.DOTALL | re.IGNORECASE)
            for tr in tr_blocks:
                tds = re.findall(r'<td[^>]*>(.*?)</td>', tr, re.DOTALL | re.IGNORECASE)
                if len(tds) >= 4:
                    u_cand = re.sub(r'<[^>]*>', '', tds[1]).strip()
                    o_cand = re.sub(r'<[^>]*>', '', tds[-1]).strip().replace(',', '')
                    if u_cand.isdigit() and re.match(r'^\d+\.\d+$', o_cand):
                        u_i = int(u_cand)
                        o_f = float(o_cand)
                        if 1 <= u_i <= 18 and 1.0 <= o_f <= 9999.9:
                            sp_map[u_i] = o_f
            if len(sp_map) >= 2:
                return sp_map, "js_realtime"
    except:
        pass

    return {}, "none"


def fetch_hybrid_odds_and_result_weights(
    race_id: str, shutuba_html: str, headers: dict, bamei_to_umaban: Optional[Dict[str, int]] = None
) -> tuple[Dict[int, float], Dict[int, tuple[float, float]], str, str]:
    """
    1. 終了した過去レースなら result.html (HTML形式) から確定単勝オッズを取得（馬名・3列目馬番で厳密照合）
    2. 未来/当日発走前のレースなら JavaScript動的API (api_get_jra_odds.html) からリアルタイム単勝オッズを取得
    """
    # 1. まずレース結果ページ(result.html)を確認し、終了レースならHTML形式で取得
    try:
        time.sleep(0.2)
        res_result = requests.get(
            f"https://race.netkeiba.com/race/result.html?race_id={race_id}",
            headers=headers, timeout=8
        )
        if res_result.status_code == 200:
            res_result.encoding = res_result.apparent_encoding
            html_odds, html_weights = parse_finished_race_html_odds(res_result.text, bamei_to_umaban)
            if len(html_odds) >= 2:
                return (
                    html_odds,
                    html_weights,
                    "html_result",
                    f"✅ オッズ自動取得 [終了レース・HTML形式]: レース結果HTMLから確定単勝オッズ（{len(html_odds)}頭）を正確に反映しました。"
                )
    except:
        pass

    # 予備: 古い過去レース等で db.netkeiba.com 側に結果HTMLがある場合
    try:
        if len(race_id) == 12 and int(race_id[:4]) < datetime.now().year:
            time.sleep(0.2)
            res_db = requests.get(f"https://db.netkeiba.com/race/{race_id}/", headers=headers, timeout=8)
            if res_db.status_code == 200:
                res_db.encoding = res_db.apparent_encoding
                db_odds: Dict[int, float] = {}
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', res_db.text, re.DOTALL | re.IGNORECASE)
                for r_html in rows:
                    tds = re.findall(r'<td[^>]*>(.*?)</td>', r_html, re.DOTALL | re.IGNORECASE)
                    if len(tds) >= 13:
                        u_s = re.sub(r'<[^>]*>', '', tds[2]).strip()
                        o_s = re.sub(r'<[^>]*>', '', tds[12]).strip().replace(',', '')
                        if u_s.isdigit() and re.match(r'^\d+\.\d+$', o_s):
                            db_odds[int(u_s)] = float(o_s)
                if len(db_odds) >= 2:
                    return (
                        db_odds,
                        {},
                        "html_db",
                        f"✅ オッズ自動取得 [過去DB・HTML形式]: 過去データベースHTMLから確定単勝オッズ（{len(db_odds)}頭）を反映しました。"
                    )
    except:
        pass

    # 2. 終了レースではない場合 ➔ 未来/当日発走前レースとしてJavaScript動的API形式で取得
    js_odds, js_mode = fetch_future_race_js_odds(race_id, headers)
    if len(js_odds) >= 2:
        if js_mode == "js_yoso":
            return (
                js_odds,
                {},
                "js_yoso",
                f"ℹ️ オッズ自動取得 [未来レース・JS形式(予想オッズ)]: 馬券発売前のためJavaScript APIから予想オッズ（{len(js_odds)}頭）を反映しました。"
            )
        return (
            js_odds,
            {},
            "js_realtime",
            f"✅ オッズ自動取得 [未来/当日レース・JS形式]: JavaScript動的APIから最新単勝オッズ（{len(js_odds)}頭）を反映しました。"
        )

    # 3. フォールバック: shutuba.html 内に直接埋め込まれているオッズがあれば抽出
    inline_odds: Dict[int, float] = {}
    for m_in in re.finditer(r'id="odds-1_0*(\d+)"[^>]*>\s*(\d+\.\d+)\s*<', shutuba_html):
        try:
            inline_odds[int(m_in.group(1))] = float(m_in.group(2))
        except:
            pass
    if len(inline_odds) >= 2:
        return (
            inline_odds,
            {},
            "html_inline",
            f"✅ オッズ自動取得 [出馬表HTML]: 出馬表ページから単勝オッズ（{len(inline_odds)}頭）を反映しました。"
        )

    return (
        {},
        {},
        "manual_required",
        "⚠️ オッズ未発売（未算出）のため初期値(10.0倍)をセットしました。必要に応じて手動で入力してください。"
    )


def infer_auto_condition_override(
    bamei: str, weight_val: float, change_val: float, weight_announced: bool,
    zenso: float, power_score: float, kaisuu: int
) -> tuple[str, str]:
    if not weight_announced:
        return "1.00", "馬体重未発表（手動設定可）"

    records = horse_history_db.get(bamei, {}).get('race_records', [])
    good_weights = []
    for r in records:
        try:
            rk = float(r.get('rank', 99))
            w = float(r.get('weight', r.get('bataijuu', 0)))
            if 1.0 <= rk <= 3.0 and 350.0 <= w <= 620.0:
                good_weights.append(w)
        except:
            pass
    best_weight = float(np.mean(good_weights)) if good_weights else None

    if change_val <= -14.0 or (change_val <= -10.0 and weight_val < 440.0):
        return "0.88", f"極端な馬体減({int(change_val)}kg)・深刻な輸送細化"

    if -12.0 <= change_val <= -10.0:
        if weight_val >= 500.0 and power_score >= 0.65:
            return "1.00", f"大型馬の絞り込み({int(change_val)}kg)・許容範囲"
        return "0.94", f"馬体減({int(change_val)}kg)・やや細化注意"

    if -8.0 <= change_val <= -4.0:
        if best_weight and abs(weight_val - best_weight) <= 3.0:
            return "1.03", f"ベスト体重へ引き締め({int(change_val)}kg)・好仕上がり"
        if weight_val >= 480.0:
            return "1.00", f"大型馬の引き締め({int(change_val)}kg)・標準範囲"
        if change_val == -8.0 and weight_val < 460.0:
            return "0.97", f"やや馬体減({int(change_val)}kg)・微減点のみ"
        return "1.00", f"本番仕様の引き締め({int(change_val)}kg)・標準"

    if -2.0 <= change_val <= 2.0:
        if power_score >= 0.72 and zenso <= 3.0 and kaisuu >= 2:
            return "1.04", f"好調維持({change_val:+.0f}kg)・好仕上がり"
        return "1.00", "馬体安定（標準）"

    if 4.0 <= change_val <= 10.0:
        if best_weight and abs(weight_val - best_weight) <= 3.0:
            return "1.03", f"ベスト体重合致({change_val:+.0f}kg)・充実"
        if zenso <= 4.0 or power_score >= 0.65:
            return "1.03", f"好走圏内の馬体増({change_val:+.0f}kg)・成長分"
        if change_val >= 8.0 and zenso >= 8.0 and power_score < 0.52:
            return "0.95", f"凡走後の馬体増({change_val:+.0f}kg)・やや太め注意"
        return "1.00", "標準範囲内"

    if change_val >= 12.0:
        if power_score >= 0.68 and zenso <= 3.0:
            return "1.02", f"実績馬の馬体増({change_val:+.0f}kg)・成長分"
        return "0.94", f"大幅馬体増({change_val:+.0f}kg)・太め残り注意"

    return "1.00", "標準"

def get_course_rpci_stats_from_val(rpci: float):
    rpci_type = get_pci_label(rpci)
    if rpci < 48.0: rec_pace = "H_MAEKUZURE"
    elif rpci < 49.8: rec_pace = "M_LONSPA"
    elif rpci <= 52.0: rec_pace = "M_JIRIKI"
    elif rpci <= 53.2: rec_pace = "S_LONSPA"
    else: rec_pace = "S_YOIDON"
    return rpci, rpci_type, rec_pace

def auto_detect_expected_pace(horses_data: list, course_rpci: float) -> tuple[str, str, str]:
    if not horses_data or len(horses_data) < 2:
        return "M_JIRIKI", "manual_required", "⚠️ 出走馬データが不足しているため標準ペース（ミドル①）にセットしました。"

    total_h = len(horses_data)
    with_history = sum(1 for h in horses_data if h.get("shussou_kaisuu", 0) > 0)
    if with_history < max(2, total_h // 3):
        _, _, rec_p = get_course_rpci_stats_from_val(course_rpci)
        return rec_p, "fallback_course", f"ℹ️ 過去走データが少ないため、舞台RPCI({course_rpci})基準で自動セットしました。"

    nige_cnt = sum(1 for h in horses_data if int(round(h.get("kyakushitsu", 2))) == 1)
    senkou_cnt = sum(1 for h in horses_data if int(round(h.get("kyakushitsu", 2))) == 2)
    makuri_cnt = sum(1 for h in horses_data if h.get("makuri_flag", False))
    front_ratio = (nige_cnt + senkou_cnt) / float(total_h)
    avg_pci = float(np.mean([h.get("pci", 50.0) for h in horses_data]))

    if nige_cnt >= 3 or (nige_cnt >= 2 and front_ratio >= 0.48) or (course_rpci < 48.0 and nige_cnt >= 2):
        pace = "H_MAEKUZURE"
        msg = f"✅ 展開自動判定: 逃げ{nige_cnt}頭・先行{senkou_cnt}頭（先行勢{int(front_ratio*100)}%）で先行激化 ➔ 『ハイ① 前崩れ消耗戦』を自動セット"
    elif (nige_cnt >= 2 and makuri_cnt >= 2) or (front_ratio >= 0.36 and avg_pci < 50.3) or (48.0 <= course_rpci < 49.8 and nige_cnt >= 1):
        pace = "M_LONSPA"
        msg = f"✅ 展開自動判定: 逃げ{nige_cnt}頭・先行{senkou_cnt}頭 / マクリ実績{makuri_cnt}頭 ➔ 『ミドル② タフ持続ロンスパ戦』を自動セット"
    elif (nige_cnt == 0 and front_ratio <= 0.25 and avg_pci >= 51.8) or (nige_cnt <= 1 and course_rpci > 53.0 and makuri_cnt == 0):
        pace = "S_YOIDON"
        msg = f"✅ 展開自動判定: 逃げ{nige_cnt}頭・先行{senkou_cnt}頭と手薄＆平均PCI {avg_pci:.1f} ➔ 『スロー① 直線ヨーイドン』を自動セット"
    elif nige_cnt <= 1 and front_ratio <= 0.35 and (makuri_cnt >= 1 or 51.5 <= course_rpci <= 53.2):
        pace = "S_LONSPA"
        msg = f"✅ 展開自動判定: 逃げ{nige_cnt}頭で道中緩むがマクリ・持続力馬あり ➔ 『スロー② ロンスパ・マクリ戦』を自動セット"
    else:
        pace = "M_JIRIKI"
        msg = f"✅ 展開自動判定: 逃げ{nige_cnt}頭・先行{senkou_cnt}頭（先行勢{int(front_ratio*100)}%）の標準隊列 ➔ 『ミドル① 地力・総合力勝負』を自動セット"

    return pace, "auto_detected", msg


def parse_single_race_result_for_bias(res_html: str, target_track_type: str) -> Optional[List[Dict[str, int]]]:
    if "HorseList" not in res_html:
        return None
    if "Result_Num" not in res_html and "PassageRate" not in res_html and "All_Result_Table" not in res_html:
        return None

    m_tr = re.search(r'(芝|ダ|障害)(?:専門)?(\d{3,4})m', res_html)
    if not m_tr:
        return None
    r_type = "障" if m_tr.group(1) == "障害" else m_tr.group(1)
    if r_type != target_track_type:
        return []

    rows = re.findall(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', res_html, re.DOTALL | re.IGNORECASE)
    if len(rows) < 3:
        return None

    total_h = len(rows)
    top3_info = []
    for row in rows[:3]:
        waku = 0
        w_m = re.search(r'<td[^>]*class="[^"]*Waku[^"]*"[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
        if w_m:
            w_s = re.sub(r'<[^>]*>', '', w_m.group(1)).strip()
            if w_s.isdigit(): waku = int(w_s)
        if waku == 0:
            tds = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
            if len(tds) >= 3:
                u_s = re.sub(r'<[^>]*>', '', tds[2]).strip()
                if u_s.isdigit(): waku = calc_jra_wakuban(int(u_s), total_h)

        last_corner = 0
        p_m = re.search(r'<td[^>]*class="[^"]*PassageRate[^"]*"[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
        if p_m:
            p_s = re.sub(r'<[^>]*>', '', p_m.group(1)).strip()
            nums = [int(x) for x in p_s.split('-') if x.strip().isdigit()]
            if nums: last_corner = nums[-1]
        if last_corner == 0:
            p_m2 = re.search(r'>\s*(\d{1,2}(?:-\d{1,2})+)\s*<', row)
            if p_m2:
                nums = [int(x) for x in p_m2.group(1).split('-') if x.strip().isdigit()]
                if nums: last_corner = nums[-1]

        if waku > 0:
            top3_info.append({"waku": waku, "last_corner": last_corner})

    return top3_info if top3_info else None


def detect_target_race_weekday(shutuba_html: str, day_num: int) -> Optional[int]:
    title_m = re.search(r'<title>(.*?)</title>', shutuba_html, re.IGNORECASE | re.DOTALL)
    if title_m:
        t_str = title_m.group(1)
        d_m = re.search(r'(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日', t_str)
        if d_m:
            try:
                dt = datetime(int(d_m.group(1)), int(d_m.group(2)), int(d_m.group(3)))
                return dt.weekday()
            except:
                pass

    act_m = re.search(r'<dd[^>]*class="[^"]*Active[^"]*"[^>]*>(.*?)</dd>', shutuba_html, re.IGNORECASE | re.DOTALL)
    if act_m:
        act_txt = act_m.group(1)
        if "土" in act_txt: return 5
        if "日" in act_txt: return 6
        if "月" in act_txt: return 0

    if day_num == 1:
        return 5
    return None


def auto_detect_track_bias(race_id: str, target_track_type: str, shutuba_html: str) -> tuple[str, str, str, str]:
    try:
        if len(race_id) < 12 or not race_id.isdigit():
            return "フラット", "中", "manual_required", "⚠️ レースID形式から過去レースを参照できないため、トラックバイアスは手動で設定してください。"

        prefix_8 = race_id[:8]
        day_num = int(race_id[8:10])
        race_num = int(race_id[10:12])

        weekday = detect_target_race_weekday(shutuba_html, day_num)
        is_saturday_race = (weekday == 5) or (day_num == 1)

        headers = {'User-Agent': 'Mozilla/5.0'}
        collected_top3 = []
        sampled_same_day = 0
        sampled_prev_day = 0

        same_day_has_results = False
        if race_num >= 2:
            r1_id = f"{prefix_8}{day_num:02d}01"
            try:
                time.sleep(0.2)
                r1_res = requests.get(f"https://race.netkeiba.com/race/result.html?race_id={r1_id}", headers=headers, timeout=4)
                if r1_res.status_code == 200:
                    r1_res.encoding = r1_res.apparent_encoding
                    parsed_r1 = parse_single_race_result_for_bias(r1_res.text, target_track_type)
                    if parsed_r1 is not None:
                        same_day_has_results = True
                        for r_no in range(race_num - 1, 1, -1):
                            if sampled_same_day >= 5:
                                break
                            cid = f"{prefix_8}{day_num:02d}{r_no:02d}"
                            try:
                                time.sleep(0.2)
                                r_res = requests.get(f"https://race.netkeiba.com/race/result.html?race_id={cid}", headers=headers, timeout=4)
                                if r_res.status_code == 200:
                                    r_res.encoding = r_res.apparent_encoding
                                    parsed = parse_single_race_result_for_bias(r_res.text, target_track_type)
                                    if parsed:
                                        collected_top3.extend(parsed)
                                        sampled_same_day += 1
                            except:
                                continue
                        if parsed_r1 and sampled_same_day < 5:
                            collected_top3.extend(parsed_r1)
                            sampled_same_day += 1
            except:
                pass

        if sampled_same_day < 2 and day_num >= 2 and not is_saturday_race:
            prev_day = day_num - 1
            for r_no in range(12, 0, -1):
                if (sampled_same_day + sampled_prev_day) >= 5:
                    break
                cid = f"{prefix_8}{prev_day:02d}{r_no:02d}"
                try:
                    time.sleep(0.2)
                    r_res = requests.get(f"https://race.netkeiba.com/race/result.html?race_id={cid}", headers=headers, timeout=4)
                    if r_res.status_code == 200:
                        r_res.encoding = r_res.apparent_encoding
                        parsed = parse_single_race_result_for_bias(r_res.text, target_track_type)
                        if parsed is None and r_no == 11 and sampled_prev_day == 0:
                            break
                        if parsed:
                            collected_top3.extend(parsed)
                            sampled_prev_day += 1
                except:
                    continue

        total_sampled_races = sampled_same_day + sampled_prev_day

        if total_sampled_races < 2 or len(collected_top3) < 4:
            if is_saturday_race and not same_day_has_results:
                return (
                    "フラット", "中", "manual_required",
                    "⚠️ 土曜開催のレース開始前（前日金曜のレースなし）のため、トラックバイアスは手動で設定してください。"
                )
            return (
                "フラット", "中", "manual_required",
                "⚠️ 参照できる当日・前日の同コース確定レース結果がまだありません。トラックバイアスは手動で設定してください。"
            )

        total_horses = len(collected_top3)
        inner_cnt = sum(1 for x in collected_top3 if x["waku"] <= 4)
        outer_cnt = sum(1 for x in collected_top3 if x["waku"] >= 5)
        inner_rate = inner_cnt / float(total_horses)
        outer_rate = outer_cnt / float(total_horses)

        valid_pass = [x["last_corner"] for x in collected_top3 if x["last_corner"] > 0]
        if valid_pass:
            front_cnt = sum(1 for p in valid_pass if p <= 4)
            closer_cnt = sum(1 for p in valid_pass if p >= 6)
            front_rate = front_cnt / float(len(valid_pass))
            closer_rate = closer_cnt / float(len(valid_pass))
        else:
            front_rate, closer_rate = 0.5, 0.5

        detected_bias = "フラット"
        detected_strength = "中"

        if inner_rate >= 0.58 and front_rate >= 0.55:
            detected_bias = "イン前有利 (1〜4枠・先行)"
            if inner_rate >= 0.70 and front_rate >= 0.65: detected_strength = "極端"
            elif inner_rate >= 0.63 or front_rate >= 0.60: detected_strength = "中"
            else: detected_strength = "弱"
        elif outer_rate >= 0.58 and closer_rate >= 0.50:
            detected_bias = "外差し有利 (5〜8枠・差し追込)"
            if outer_rate >= 0.70 and closer_rate >= 0.60: detected_strength = "極端"
            elif outer_rate >= 0.63 or closer_rate >= 0.55: detected_strength = "中"
            else: detected_strength = "弱"
        elif front_rate >= 0.65:
            detected_bias = "前残り (逃げ・先行有利)"
            detected_strength = "極端" if front_rate >= 0.78 else "中"
        elif closer_rate >= 0.60:
            detected_bias = "差し有利 (外伸び)"
            detected_strength = "極端" if closer_rate >= 0.72 else "中"
        elif inner_rate >= 0.65:
            detected_bias = "内枠有利 (インベタ)"
            detected_strength = "極端" if inner_rate >= 0.75 else "中"
        elif outer_rate >= 0.65:
            detected_bias = "外枠有利"
            detected_strength = "極端" if outer_rate >= 0.75 else "中"

        if sampled_same_day >= 2 and sampled_prev_day == 0:
            source_label = f"当日({day_num}日目)の最新同コース結果 {sampled_same_day}鞍"
        elif sampled_same_day == 0 and sampled_prev_day >= 2:
            source_label = f"前日・土曜({day_num - 1}日目)の同コース確定結果 {sampled_prev_day}鞍"
        else:
            source_label = f"当日{sampled_same_day}鞍＋前日{sampled_prev_day}鞍の同コース結果"

        summary_msg = (
            f"✅ バイアス自動判定 [{source_label} / 上位{total_horses}頭集計]："
            f"内枠率{int(inner_rate*100)}%・4角4番手以内率{int(front_rate*100)}% ➔ "
            f"『{detected_bias}（強度:{detected_strength}）』を自動セットしました。"
        )
        return detected_bias, detected_strength, "auto_detected", summary_msg

    except Exception:
        return (
            "フラット", "中", "manual_required",
            "⚠️ 過去レース結果の取得をスキップしました。トラックバイアスは手動で設定してください。"
        )


class HorseData(BaseModel):
    wakuban: float
    umaban: float
    bamei: str = ""
    kishu: str = "不明"
    odds: float = 0.0
    kinryou: float
    bataijuu: float = 480.0
    weight_change: float = 0.0  
    condition_override: str = "1.00"
    condition_reason: Optional[str] = ""
    zenso_chakujun: float = 8.0
    avg_agari: float = 35.0
    kyakushitsu: float = 2.0  
    shussou_kaisuu: float = 0.0
    pci: Optional[float] = 50.0
    pci_type: Optional[str] = "B"

class RaceRequest(BaseModel):
    race_id: str
    horses: List[HorseData]
    course_type: str
    distance: int
    weather: str
    track_condition: str
    expected_pace: str = "M_JIRIKI"
    track_bias: str = "フラット"
    bias_strength: str = "中" 
    cushion_value: float = 9.0 

@app.get("/fetch_race/{race_id}")
def fetch_race(race_id: str):
    headers = {'User-Agent': 'Mozilla/5.0'}
    html = None
    is_shutuba = False
    
    for attempt in range(3):
        time.sleep(random.uniform(1.0, 2.0)) 
        try:
            res = requests.get(f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}", headers=headers, timeout=15)
            res.raise_for_status()
            res.encoding = res.apparent_encoding
            html = res.text
            if "HorseList" in html or "馬番" in html:
                is_shutuba = True
                break
        except:
            if attempt == 2: return {"status": "error", "message": "通信エラー"}

    if not is_shutuba:
        return {"status": "error", "message": "出馬表データが見つかりませんでした"}

    try:
        title_match = re.search(r'<title>(.*?)</title>', html)
        race_name = title_match.group(1).split("|")[0].strip() if title_match else "レース不明"
        place_code = race_id[4:6] if len(race_id) >= 6 else '05'
        place_dict = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
        place_name = place_dict.get(place_code, "地方/海外")

        track_type, distance, weather, track_condition = "芝", 2000, "晴", "良"
        match_track = re.search(r'(芝|ダ|障害)(?:専門)?(\d{3,4})m', html)
        if match_track:
            track_type = "障" if match_track.group(1) == "障害" else match_track.group(1)
            distance = int(match_track.group(2))

        m_weather = re.search(r'天候\s*:\s*(晴|曇|小雨|雨|小雪|雪)', html)
        if m_weather:
            w_val = m_weather.group(1)
            weather = w_val if w_val in ["晴", "曇", "雨", "小雨"] else "曇"

        m_cond = re.search(r'(?:芝|ダート|馬場)\s*:\s*(良|稍重|稍|重|不良|不)', html)
        if m_cond:
            c_val = m_cond.group(1)
            if c_val == "稍重": c_val = "稍"
            elif c_val == "不良": c_val = "不"
            track_condition = c_val

        horse_blocks = re.findall(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)

        # まず出馬表から「馬名 -> 馬番」の正確な辞書を作成（result.htmlでの着順誤認を100%防ぐため）
        bamei_to_umaban: Dict[str, int] = {}
        for block in horse_blocks:
            try:
                u_num = 0
                u_m = re.search(r'<td[^>]*class="[^"]*(?:Umaban)[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                if u_m:
                    u_str = re.sub(r'<[^>]*>', '', u_m.group(1)).strip()
                    if u_str.isdigit(): u_num = int(u_str)
                if u_num == 0: continue
                b_m = re.search(r'<span[^>]*class="[^"]*HorseName[^"]*"[^>]*>(.*?)</span>', block, re.IGNORECASE | re.DOTALL)
                if not b_m: b_m = re.search(r'<a[^>]*href="[^"]*horse/\d+[^"]*"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
                if b_m:
                    b_str = re.sub(r'<[^>]*>', '', b_m.group(1)).strip()
                    if b_str:
                        bamei_to_umaban[b_str] = u_num
            except:
                continue

        # ★ 終了レース(HTML形式) ＆ 未来レース(JavaScript動的API形式) のハイブリッドオッズ取得
        hybrid_odds_map, result_weight_map, odds_status, odds_msg = fetch_hybrid_odds_and_result_weights(
            race_id, html, headers, bamei_to_umaban
        )

        horses_data = []
        
        def get_auto_kyakushitsu(bamei_str):
            try:
                records = horse_history_db.get(bamei_str, {}).get('race_records', [])
                if not records: return 2.0
                last_corners = []
                for r in records[-5:]:
                    passing = str(r.get('passing', ''))
                    if passing and passing != '-':
                        try: last_corners.append(int(passing.split('-')[-1]))
                        except: pass
                if not last_corners: return 2.0
                avg_pos = np.mean(last_corners)
                if avg_pos <= 2.5: return 1.0
                elif avg_pos <= 6.5: return 2.0
                elif avg_pos <= 10.5: return 3.0
                else: return 4.0
            except: return 2.0

        any_weight_announced = False

        for block in horse_blocks:
            try:
                umaban = 0
                u_m = re.search(r'<td[^>]*class="[^"]*(?:Umaban)[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                if u_m:
                    u_str = re.sub(r'<[^>]*>', '', u_m.group(1)).strip()
                    if u_str.isdigit(): umaban = int(u_str)
                if umaban == 0: continue

                wakuban = 0
                waku_m = re.search(r'<td[^>]*class="[^"]*(?:Waku)[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                if waku_m:
                    waku_str = re.sub(r'<[^>]*>', '', waku_m.group(1)).strip()
                    if waku_str.isdigit(): wakuban = int(waku_str)
                
                bamei = f"馬番{umaban}"
                b_m = re.search(r'<span[^>]*class="[^"]*HorseName[^"]*"[^>]*>(.*?)</span>', block, re.IGNORECASE | re.DOTALL)
                if not b_m: b_m = re.search(r'<a[^>]*href="[^"]*horse/\d+[^"]*"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
                if b_m:
                    b_str = re.sub(r'<[^>]*>', '', b_m.group(1)).strip()
                    if b_str: bamei = b_str

                kishu = "不明"
                k_m = re.search(r'<td[^>]*class="[^"]*Jockey[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                if not k_m: k_m = re.search(r'<a[^>]*href="[^"]*jockey/\d+[^"]*"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
                if k_m:
                    k_str = re.sub(r'<[^>]*>', '', k_m.group(1)).strip()
                    if k_str: kishu = re.sub(r'[\r\n\t\s]+', '', k_str)

                kinryou = 55.0
                kin_matches = re.findall(r'<td[^>]*>\s*(4[8-9](?:\.\d)?|5[0-9](?:\.\d)?|6[0-2](?:\.\d)?)\s*</td>', block)
                if kin_matches:
                    try: kinryou = float(kin_matches[0])
                    except: pass

                odds = 10.0
                if umaban in hybrid_odds_map:
                    odds = hybrid_odds_map[umaban]
                else:
                    o_m = re.search(r'<span[^>]*id="odds[^>]*>(.*?)</span>', block, re.IGNORECASE | re.DOTALL)
                    if o_m:
                        o_str = re.sub(r'<[^>]*>', '', o_m.group(1)).strip().replace(',', '')
                        try:
                            val_o = float(o_str)
                            if val_o >= 1.0:
                                odds = val_o
                        except:
                            pass

                weight_val = 480.0
                change_val = 0.0
                weight_announced = False
                w_m = re.search(r'<td[^>]*class="[^"]*Weight[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                if w_m:
                    w_str = re.sub(r'<[^>]*>', '', w_m.group(1)).strip()
                    m_weight = re.search(r'(\d{3})\s*\(([+-]?\d+)\)', w_str)
                    if m_weight:
                        weight_val = float(m_weight.group(1))
                        change_val = float(m_weight.group(2))
                        weight_announced = True
                        any_weight_announced = True
                    else:
                        m_weight2 = re.search(r'(\d{3})', w_str)
                        if m_weight2:
                            weight_val = float(m_weight2.group(1))

                if not weight_announced and umaban in result_weight_map:
                    weight_val, change_val = result_weight_map[umaban]
                    weight_announced = True
                    any_weight_announced = True

                h_pci, h_pci_type = get_horse_pci_stats(bamei)
                zenso, agari, kaisuu, power_score, makuri_flag = get_horse_real_stats(bamei)

                auto_cond_code, auto_cond_reason = infer_auto_condition_override(
                    bamei, weight_val, change_val, weight_announced, zenso, power_score, kaisuu
                )

                horses_data.append({
                    "wakuban": wakuban,
                    "umaban": umaban, "bamei": bamei, "kishu": kishu,
                    "odds": odds, "kinryou": kinryou, 
                    "bataijuu": weight_val, "weight_change": change_val, 
                    "condition_override": auto_cond_code,
                    "condition_reason": auto_cond_reason,
                    "zenso_chakujun": zenso, "avg_agari": agari,
                    "kyakushitsu": get_auto_kyakushitsu(bamei),
                    "shussou_kaisuu": kaisuu,
                    "pci": h_pci, "pci_type": h_pci_type,
                    "makuri_flag": makuri_flag,
                    "power_score": round(power_score, 3)
                })
            except: continue

        horses_data = sorted(horses_data, key=lambda x: x['umaban'])
        total_h = len(horses_data)
        for h in horses_data:
            if not h["wakuban"] or h["wakuban"] <= 0:
                h["wakuban"] = calc_jra_wakuban(h["umaban"], total_h)

        course_key = f"{place_code}_{track_type}_{distance}"
        rpci, rpci_type, _ = get_course_rpci_stats(course_key)

        auto_pace, pace_status, pace_msg = auto_detect_expected_pace(horses_data, rpci)
        auto_bias, auto_strength, bias_status, bias_msg = auto_detect_track_bias(race_id, track_type, html)

        if any_weight_announced:
            weight_status = "auto_detected"
            weight_msg = "✅ 当日馬体重の増減と過去好走実績から、全頭の『状態補正（本番引き締め・成長分・細化注意等）』をマイルド設定で自動セットしました。"
        else:
            weight_status = "manual_required"
            weight_msg = "⚠️ 当日馬体重が未発表のため『通常（補正なし）』にセットしました。発表後または手動で設定してください。"

        return {
            "status": "success", "race_name": race_name, "place_name": place_name,
            "course_type": track_type, "distance": distance,
            "weather": weather, "track_condition": track_condition,
            "rpci": rpci, "rpci_type": rpci_type,
            "recommended_pace": auto_pace,
            "auto_track_bias": auto_bias,
            "auto_bias_strength": auto_strength,
            "auto_status": {
                "odds_status": odds_status,
                "odds_message": odds_msg,
                "pace_status": pace_status,
                "pace_message": pace_msg,
                "bias_status": bias_status,
                "bias_message": bias_msg,
                "weight_status": weight_status,
                "weight_message": weight_msg
            },
            "horses": horses_data
        }

    except Exception as e:
        return {"status": "error", "message": f"パース処理エラー: {str(e)}"}


def get_time_limit_multiplier(horse_name, course_type, current_distance, expected_standard_time, cushion_val, track_cond):
    is_rough_track = (track_cond in ["稍", "重", "不"]) or (cushion_val < 8.5)
    if is_rough_track:
        return 1.00 

    if current_distance >= 2500:
        return 1.00 

    h_records = horse_history_db.get(horse_name, {}).get('race_records', [])
    if not h_records:
        return 1.00 

    best_diff_per_halon = None
    cur_halons = current_distance / 200.0
    
    same_dist_times = []
    for r in h_records:
        r_c = str(r.get('course', ''))
        r_cond = str(r.get('cond', ''))
        t_sec = parse_time_to_seconds(r.get('time'))
        if t_sec is None: continue
        
        if course_type in r_c:
            m = re.search(r'(\d{3,4})', r_c)
            if m:
                dist = int(m.group(1))
                if dist == current_distance and r_cond in ['良', '稍']:
                    same_dist_times.append(t_sec)
                    
    if same_dist_times:
        best_time = min(same_dist_times)
        best_diff_per_halon = (best_time - expected_standard_time) / cur_halons
    else:
        adjacent_diffs = []
        for r in h_records:
            r_c = str(r.get('course', ''))
            r_cond = str(r.get('cond', ''))
            t_sec = parse_time_to_seconds(r.get('time'))
            if t_sec is None: continue
            
            if course_type in r_c:
                m = re.search(r'(\d{3,4})', r_c)
                if m:
                    dist = int(m.group(1))
                    if 0 < abs(dist - current_distance) <= 400 and r_cond in ['良', '稍']:
                        adj_stats = course_history_db.get(r_c, {})
                        adj_std = adj_stats.get('standard_time')
                        if adj_std is not None:
                            adj_diff = (t_sec - float(adj_std)) / (dist / 200.0)
                        else:
                            halon_gap = (current_distance - dist) / 200.0
                            pace_penalty = halon_gap * 0.45 if halon_gap > 0 else halon_gap * 0.35
                            projected_time = (t_sec * (current_distance / dist)) + pace_penalty
                            adj_diff = (projected_time - expected_standard_time) / cur_halons
                        adjacent_diffs.append(adj_diff)
        
        if adjacent_diffs:
            best_diff_per_halon = min(adjacent_diffs)
            
    if best_diff_per_halon is None:
        return 1.00 

    mult = 1.00
    is_high_speed = (cushion_val >= 9.5)
    
    if best_diff_per_halon <= 0.05:
        mult = 1.00 
    elif best_diff_per_halon <= 0.10:
        mult = 0.95 if is_high_speed else 0.98 
    elif best_diff_per_halon <= 0.15:
        mult = 0.88 if is_high_speed else 0.94 
    else:
        mult = 0.75 if is_high_speed else 0.88 

    return mult


def calculate_stage1_pure_score(raw_lgb_pred_neutral: float, horse: HorseData, power_score: float, time_diff_base: float) -> float:
    if horse.odds > 1.0:
        market_prob = min(0.55, 0.80 / float(horse.odds))
    else:
        market_prob = 0.08

    agari_factor = 1.0
    if time_diff_base <= -0.3: agari_factor = 1.03
    elif time_diff_base <= 0.0: agari_factor = 1.01
    elif time_diff_base >= 0.6: agari_factor = 0.97

    pure_ability = ((raw_lgb_pred_neutral * 0.38) + (power_score * 0.44) + (market_prob * 0.18)) * agari_factor
    return float(pure_ability)


def calculate_comprehensive_score(
    raw_lgb_pred, horse, power_score, makuri_flag, h_pci, h_pci_type, time_diff, 
    expected_pace, track_bias, bias_strength, cushion_value, course_rpci, 
    time_limit_mult
):
    if horse.odds > 1.0:
        market_prob = min(0.55, 0.80 / float(horse.odds))
    else: market_prob = 0.08

    if expected_pace in ["M_JIRIKI", "M"]:
        base_ability = (raw_lgb_pred * 0.40) + (power_score * 0.40) + (market_prob * 0.20)
    else:
        base_ability = (raw_lgb_pred * 0.44) + (power_score * 0.36) + (market_prob * 0.20)

    kyaku = int(round(horse.kyakushitsu))
    waku = horse.wakuban

    pace_mult = 1.0
    if expected_pace in ["S_YOIDON", "S"]:
        if h_pci_type == "C": pace_mult *= 1.10
        elif h_pci_type == "B": pace_mult *= 1.00
        else: pace_mult *= 0.93
        if kyaku == 1: pace_mult *= 1.06
        elif kyaku == 2: pace_mult *= 1.08
        elif kyaku == 3: pace_mult *= 1.04
        elif kyaku == 4: pace_mult *= 0.90
        if time_diff <= -0.3: pace_mult *= 1.06
        elif time_diff <= 0.1: pace_mult *= 1.03
        elif time_diff >= 0.6: pace_mult *= 0.95

    elif expected_pace == "S_LONSPA":
        if 49.5 <= h_pci <= 54.5: pace_mult *= 1.09
        elif h_pci_type == "C": pace_mult *= 1.04
        else: pace_mult *= 0.95
        if kyaku == 1: pace_mult *= 0.96
        elif kyaku == 2: pace_mult *= 1.06
        elif kyaku == 3: pace_mult *= 1.08
        elif kyaku == 4: pace_mult *= 0.96
        if makuri_flag: pace_mult *= 1.06

    elif expected_pace in ["M_JIRIKI", "M"]:
        if h_pci_type == "B": pace_mult *= 1.06
        else: pace_mult *= 0.99
        if kyaku == 1: pace_mult *= 0.98
        elif kyaku == 2: pace_mult *= 1.07
        elif kyaku == 3: pace_mult *= 1.04
        elif kyaku == 4: pace_mult *= 0.94
        if power_score >= 0.72: pace_mult *= 1.05

    elif expected_pace == "M_LONSPA":
        if 46.5 <= h_pci <= 51.5: pace_mult *= 1.08
        elif h_pci_type == "A": pace_mult *= 1.04
        else: pace_mult *= 0.95
        if kyaku == 1: pace_mult *= 0.92
        elif kyaku == 2: pace_mult *= 1.04
        elif kyaku == 3: pace_mult *= 1.08
        elif kyaku == 4: pace_mult *= 1.02
        if makuri_flag: pace_mult *= 1.05

    elif expected_pace in ["H_MAEKUZURE", "H"]:
        if h_pci_type == "A": pace_mult *= 1.12
        elif h_pci_type == "B": pace_mult *= 1.02
        else: pace_mult *= 0.94
        if kyaku == 1: pace_mult *= 0.86
        elif kyaku == 2: pace_mult *= 0.94
        elif kyaku == 3: pace_mult *= 1.08
        elif kyaku == 4: pace_mult *= 1.11

    if course_rpci > 0 and h_pci > 0:
        if abs(course_rpci - h_pci) >= 4.0:
            pace_mult *= 0.95

    bias_mult = 1.0
    boost = 1.08
    nerf = 0.92
    if bias_strength == "弱":
        boost = 1.03; nerf = 0.97
    elif bias_strength == "極端":
        boost = 1.15; nerf = 0.80

    half_boost = 1.0 + (boost - 1.0) * 0.5
    half_nerf = 1.0 - (1.0 - nerf) * 0.5

    if track_bias == "イン前有利 (1〜4枠・先行)":
        if waku <= 4 and kyaku <= 2:
            bias_mult = boost
        elif (waku <= 4 and kyaku == 3) or (waku == 5 and kyaku <= 2):
            bias_mult = half_boost
        elif waku >= 6 and kyaku >= 3:
            bias_mult = nerf
        elif waku >= 6 or kyaku == 4:
            bias_mult = half_nerf
    elif track_bias == "外差し有利 (5〜8枠・差し追込)":
        if waku >= 5 and kyaku >= 3:
            bias_mult = boost
        elif (waku >= 5 and kyaku == 2) or (waku <= 4 and kyaku >= 3):
            bias_mult = half_boost
        elif waku <= 3 and kyaku <= 2:
            bias_mult = nerf
        elif waku <= 4 and kyaku == 1:
            bias_mult = half_nerf
    elif track_bias == "前残り (逃げ・先行有利)":
        if kyaku <= 2: bias_mult = boost
        elif kyaku >= 3: bias_mult = nerf
    elif track_bias == "差し有利 (外伸び)":
        if kyaku >= 3: bias_mult = boost
        elif kyaku <= 2: bias_mult = nerf
    elif track_bias == "内枠有利 (インベタ)":
        if waku <= 3: bias_mult = boost
        elif waku == 4: bias_mult = half_boost
        elif waku >= 6: bias_mult = nerf
    elif track_bias == "外枠有利":
        if waku >= 6: bias_mult = boost
        elif waku == 5: bias_mult = half_boost
        elif waku <= 3: bias_mult = nerf

    condition_mult = 1.00
    try:
        condition_mult = float(horse.condition_override)
    except:
        pass

    total_mult = pace_mult * bias_mult * time_limit_mult * condition_mult
    capped_mult = max(0.78, min(1.22, total_mult))

    return base_ability * capped_mult, round(bias_mult, 3), round(pace_mult, 3), round(condition_mult, 3)


def build_betting_recommendations(horses_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not horses_data:
        return {}

    by_stage2 = sorted(horses_data, key=lambda x: (x["shisu"], x["myoumi"]), reverse=True)
    by_stage1 = sorted(horses_data, key=lambda x: x["stage1_shisu"], reverse=True)

    axis_horse = None
    for h in by_stage2:
        if h.get("time_limit_mult", 1.0) >= 0.94:
            axis_horse = h
            break
    if axis_horse is None:
        axis_horse = by_stage2[0]
    axis_umaban = axis_horse["umaban"]

    main_2nd_horses = []
    for h in by_stage2:
        u = h["umaban"]
        if u == axis_umaban:
            continue
        if h.get("time_limit_mult", 1.0) <= 0.88 or h.get("condition_mult", 1.0) <= 0.88:
            continue
        if len(main_2nd_horses) < 3 and (h["shisu"] >= 53.0 or (h["myoumi"] >= 57.0 and h["shisu"] >= 50.0)):
            main_2nd_horses.append(h)

    if len(main_2nd_horses) < 2:
        for h in by_stage2:
            u = h["umaban"]
            if u != axis_umaban and h not in main_2nd_horses and h.get("time_limit_mult", 1.0) > 0.88:
                main_2nd_horses.append(h)
                if len(main_2nd_horses) >= 2:
                    break

    main_2nd_umabans = [h["umaban"] for h in main_2nd_horses]

    saver_3rd_horses = []
    for h in by_stage1:
        u = h["umaban"]
        if u == axis_umaban or u in main_2nd_umabans:
            continue
        if h.get("condition_mult", 1.0) <= 0.88:
            continue

        is_jiriki_top = (h["stage1_rank"] <= 4) or (h["stage1_shisu"] >= 55.0)
        t_mult = h.get("time_limit_mult", 1.0)

        if is_jiriki_top:
            saver_3rd_horses.append(h)
            h["is_bias_victim_rescued"] = True
            if t_mult <= 0.94:
                h["rescue_type"] = "time_warn"
            else:
                h["rescue_type"] = "bias_victim"
        elif (h["stage1_rank"] <= 6 or h["stage1_shisu"] >= 53.0) and t_mult >= 0.95:
            saver_3rd_horses.append(h)
            h["is_bias_victim_rescued"] = True
            h["rescue_type"] = "bias_victim"

        if len(saver_3rd_horses) >= 3:
            break

    for h in by_stage2:
        u = h["umaban"]
        if u == axis_umaban or u in main_2nd_umabans:
            continue
        if h not in saver_3rd_horses and h["shisu"] >= 54.0 and h.get("time_limit_mult", 1.0) >= 0.95 and len(saver_3rd_horses) < 4:
            h["rescue_type"] = "stage2_next"
            saver_3rd_horses.append(h)

    saver_3rd_umabans = [h["umaban"] for h in saver_3rd_horses]

    eliminated_umabans = set()
    for h in by_stage2:
        u = h["umaban"]
        if u == axis_umaban or u in main_2nd_umabans or u in saver_3rd_umabans:
            continue
        t_mult = h.get("time_limit_mult", 1.0)
        c_mult = h.get("condition_mult", 1.0)
        if t_mult <= 0.88:
            eliminated_umabans.add(u)
            h["role_code"] = "ELIMINATED"
            h["role_label"] = "💀 完全消し (持ち時計不足)"
        elif c_mult <= 0.88:
            eliminated_umabans.add(u)
            h["role_code"] = "ELIMINATED"
            h["role_label"] = "💀 完全消し (極端な馬体減)"

    for h in horses_data:
        u = h["umaban"]
        is_jiriki_top = (h["stage1_rank"] <= 4) or (h["stage1_shisu"] >= 55.0)
        t_mult = h.get("time_limit_mult", 1.0)
        h["is_jiriki_time_warn"] = bool(is_jiriki_top and t_mult <= 0.94)

        if u == axis_umaban:
            h["role_code"] = "AXIS_1ST"
            h["role_label"] = "👑 1着軸 (不動の本命)"
        elif u in main_2nd_umabans:
            h["role_code"] = "MAIN_2ND"
            if h["myoumi"] >= 58.0 and h["odds"] >= 8.0:
                h["role_label"] = "🚀 2列目本線 (バイアス特注穴馬)"
            else:
                h["role_label"] = "🚀 2列目本線 (好位・バイアス合致)"
        elif u in saver_3rd_umabans:
            if h.get("rescue_type") == "time_warn":
                h["role_code"] = "SAVER_3RD_TIME_WARN"
                h["role_label"] = f"⚠️ 地力上位・持ち時計注意 (地力{h['stage1_rank']}位)"
            elif h.get("rescue_type") == "bias_victim":
                h["role_code"] = "SAVER_3RD"
                h["role_label"] = f"🛡️ 地力上位・バイアス救済 (地力{h['stage1_rank']}位)"
            else:
                h["role_code"] = "SAVER_3RD"
                h["role_label"] = "🛡️ 3列目押さえ (展開・能力圏内)"
        elif u not in eliminated_umabans:
            h["role_code"] = "OTHER"
            h["role_label"] = "静観 (指数下位)"

    row1 = [axis_umaban]
    row2 = main_2nd_umabans
    row3 = main_2nd_umabans + saver_3rd_umabans

    value_pick = None
    cands_for_val = [h for h in main_2nd_horses if h["odds"] >= 6.0 and h["myoumi"] >= 55.0]
    if cands_for_val:
        value_pick = sorted(cands_for_val, key=lambda x: x["myoumi"], reverse=True)[0]
    tansho_list = [{"umaban": axis_umaban, "bamei": axis_horse["bamei"], "type": "本命単勝", "odds": axis_horse["odds"]}]
    if value_pick and value_pick["umaban"] != axis_umaban:
        tansho_list.append({"umaban": value_pick["umaban"], "bamei": value_pick["bamei"], "type": "妙味穴単勝", "odds": value_pick["odds"]})

    fukusho_list = [{"umaban": axis_umaban, "bamei": axis_horse["bamei"], "type": "鉄板軸", "odds": axis_horse["odds"]}]
    for mh in sorted(main_2nd_horses, key=lambda x: x["myoumi"], reverse=True)[:2]:
        fukusho_list.append({"umaban": mh["umaban"], "bamei": mh["bamei"], "type": "妙味複勝", "odds": mh["odds"]})

    wide_combos = [f"{axis_umaban} - {u}" for u in row2]

    sanrenpuku_nagashi_combos = []
    for i in range(len(row3)):
        for j in range(i + 1, len(row3)):
            pair = sorted([row3[i], row3[j]])
            sanrenpuku_nagashi_combos.append(f"{axis_umaban} - {pair[0]} - {pair[1]}")

    sanrenpuku_form_set = set()
    for a in row1:
        for b in row2:
            for c in row3:
                if len({a, b, c}) == 3:
                    sanrenpuku_form_set.add(tuple(sorted([a, b, c])))
    sanrenpuku_form_combos = [f"{t[0]} - {t[1]} - {t[2]}" for t in sorted(sanrenpuku_form_set)]

    sanrentan_nagashi_combos = []
    for b in row3:
        for c in row3:
            if b != c:
                sanrentan_nagashi_combos.append(f"{axis_umaban} → {b} → {c}")

    sanrentan_form_combos = []
    for a in row1:
        for b in row2:
            for c in row3:
                if a != b and b != c and a != c:
                    sanrentan_form_combos.append(f"{a} → {b} → {c}")

    eliminated_details = [
        {"umaban": h["umaban"], "bamei": h["bamei"], "reason": h["role_label"], "odds": h["odds"]}
        for h in by_stage2 if h["umaban"] in eliminated_umabans
    ]

    return {
        "formation_groups": {
            "axis_1st": [{"umaban": axis_horse["umaban"], "bamei": axis_horse["bamei"], "shisu": axis_horse["shisu"], "stage1_shisu": axis_horse["stage1_shisu"], "odds": axis_horse["odds"]}],
            "main_2nd": [{"umaban": h["umaban"], "bamei": h["bamei"], "shisu": h["shisu"], "stage1_shisu": h["stage1_shisu"], "myoumi": h["myoumi"], "odds": h["odds"]} for h in main_2nd_horses],
            "saver_3rd": [{
                "umaban": h["umaban"],
                "bamei": h["bamei"],
                "shisu": h["shisu"],
                "stage1_shisu": h["stage1_shisu"],
                "stage1_rank": h["stage1_rank"],
                "odds": h["odds"],
                "rescued": h.get("is_bias_victim_rescued", False),
                "rescue_type": h.get("rescue_type", "normal"),
                "role_label": h.get("role_label", "")
            } for h in saver_3rd_horses],
            "eliminated": eliminated_details
        },
        "tansho": {"picks": tansho_list, "count": len(tansho_list)},
        "fukusho": {"picks": fukusho_list, "count": len(fukusho_list)},
        "wide": {"axis": axis_umaban, "targets": row2, "combos": wide_combos, "count": len(wide_combos)},
        "sanrenpuku_nagashi": {"axis": axis_umaban, "targets": row3, "combos": sanrenpuku_nagashi_combos, "count": len(sanrenpuku_nagashi_combos)},
        "sanrenpuku_formation": {"row1": row1, "row2": row2, "row3": row3, "combos": sanrenpuku_form_combos, "count": len(sanrenpuku_form_combos)},
        "sanrentan_nagashi": {"first": axis_umaban, "targets": row3, "combos": sanrentan_nagashi_combos, "count": len(sanrentan_nagashi_combos)},
        "sanrentan_formation": {"first": row1, "second": row2, "third": row3, "combos": sanrentan_form_combos, "count": len(sanrentan_form_combos)}
    }


@app.post("/diagnose_race")
def diagnose_race(request: RaceRequest):
    horses_data = []
    course_enc = safe_encode('course', request.course_type)
    weather_enc = safe_encode('weather', request.weather)
    cond_enc = safe_encode('condition', request.track_condition)
    place_code = request.race_id[4:6] if len(request.race_id) >= 6 else "05"
    course_key = f"{place_code}_{request.course_type}_{request.distance}"
    
    c_stats = course_history_db.get(course_key, {})
    top3_up = c_stats.get('top3_up', [])
    pace_f = c_stats.get('pace_first', [])
    pace_l = c_stats.get('pace_last', [])
    
    expected_standard_time = get_expected_standard_time(c_stats, request.course_type, request.distance)
    
    base_req_up = float(np.mean(top3_up)) if top3_up else 35.0
    base_pace_f = float(np.mean(pace_f)) if pace_f else 35.0
    base_pace_l = float(np.mean(pace_l)) if pace_l else 35.0

    base_rpci = round((base_pace_f / base_pace_l) * 50.0, 1) if base_pace_l > 0 else 50.0
    base_rpci_type = get_pci_label(base_rpci)
    
    avg_pace_f, avg_pace_l, avg_req_up = adjust_pace_times(
        request.expected_pace, base_pace_f, base_pace_l, base_req_up
    )
    
    place_dict = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
    
    for horse in request.horses:
        h_stats = horse_history_db.get(horse.bamei, {})
        zenso, real_agari, kaisuu, power_score, makuri_flag = get_horse_real_stats(horse.bamei)

        effective_zenso_for_lgb = min(zenso, 6.0) if power_score >= 0.78 else zenso

        course_up_times = h_stats.get('course_up_times', {}).get(course_key, [])
        course_up = float(np.mean(course_up_times)) if course_up_times else real_agari
        time_diff_base = course_up - base_req_up
        time_diff = course_up - avg_req_up
        kishu_enc = safe_encode('kishu', horse.kishu)
        
        time_limit_mult = get_time_limit_multiplier(
            horse.bamei, request.course_type, request.distance, 
            expected_standard_time, request.cushion_value, request.track_condition
        )
        
        stage1_input = pd.DataFrame([{
            '枠番': 4.0, '馬番': 8.0, '斤量': horse.kinryou, '馬体重_数値': 480.0,
            'distance': request.distance, 'course_type_enc': course_enc, 'weather_enc': weather_enc,
            'track_condition_enc': cond_enc, '舞台_平均前半ペース': base_pace_f, '舞台_平均後半ペース': base_pace_l,
            '舞台_平均要求上がり': base_req_up, '過去基本脚質': horse.kyakushitsu, '前走着順': effective_zenso_for_lgb,
            '近3走_平均上がり': real_agari, '同舞台_平均上がり': course_up, '出走回数': kaisuu,
            '上がりタイム差分': time_diff_base, 'kishu_enc': kishu_enc
        }])
        raw_pred_stage1 = float(model.predict(stage1_input)[0])
        stage1_prob = calculate_stage1_pure_score(raw_pred_stage1, horse, power_score, time_diff_base)

        input_data = pd.DataFrame([{
            '枠番': horse.wakuban, '馬番': horse.umaban, '斤量': horse.kinryou, '馬体重_数値': horse.bataijuu,
            'distance': request.distance, 'course_type_enc': course_enc, 'weather_enc': weather_enc,
            'track_condition_enc': cond_enc, '舞台_平均前半ペース': avg_pace_f, '舞台_平均後半ペース': avg_pace_l,
            '舞台_平均要求上がり': avg_req_up, '過去基本脚質': horse.kyakushitsu, '前走着順': effective_zenso_for_lgb,
            '近3走_平均上がり': real_agari, '同舞台_平均上がり': course_up, '出走回数': kaisuu,
            '上がりタイム差分': time_diff, 'kishu_enc': kishu_enc
        }])
        
        raw_pred = float(model.predict(input_data)[0])
        h_pci, h_pci_type = get_horse_pci_stats(horse.bamei)

        final_score, bias_mult, pace_mult, cond_mult = calculate_comprehensive_score(
            raw_pred, horse, power_score, makuri_flag, h_pci, h_pci_type, time_diff, 
            request.expected_pace, request.track_bias, request.bias_strength, request.cushion_value,
            base_rpci, time_limit_mult
        )
        
        records = h_stats.get('race_records', [])[-5:]
        history_details = []
        for r in reversed(records):
            p_code = str(r.get('race_id', ''))[4:6]
            place = place_dict.get(p_code, "地方")
            try:
                c_type = str(r['course']).split('_')[1]
                dist = str(r['course']).split('_')[2]
            except:
                c_type, dist = "", ""
            race_name = f"{place} {c_type}{dist}m ({r.get('cond', '良')})"
            single_pci = calc_single_pci(r.get('course', ''), r.get('time', ''), r.get('up', ''))
            history_details.append({
                "race_name": race_name,
                "rank": f"{int(r.get('rank', 99))}着" if 'rank' in r else "-",
                "time": str(r.get('time', '-')),
                "up": f"{r.get('up', '-')}秒" if r.get('up', '-') != '-' else "-",
                "passing": r.get('passing', '-'),
                "pci": f"{single_pci}" if single_pci is not None else "-"
            })

        horses_data.append({
            "wakuban": int(horse.wakuban),
            "umaban": int(horse.umaban), "bamei": horse.bamei, "odds": float(horse.odds),
            "stage1_prob": stage1_prob,
            "base_prob": final_score, "time_diff": round(time_diff, 2),
            "pci": h_pci, "pci_type": h_pci_type,
            "bataijuu": horse.bataijuu, "weight_change": horse.weight_change,
            "time_limit_mult": round(time_limit_mult, 2),
            "bias_mult": bias_mult,
            "pace_mult": pace_mult,
            "condition_mult": cond_mult,
            "power_score": round(power_score, 2),
            "history_details": history_details 
        })
    
    s1_probs = [h["stage1_prob"] for h in horses_data]
    s1_mean = np.mean(s1_probs) if s1_probs else 0
    s1_std = np.std(s1_probs) if s1_probs else 1
    for h in horses_data:
        s1_val = ((h["stage1_prob"] - s1_mean) / s1_std) * 10 + 50 if s1_std > 0 else 50.0
        h["stage1_shisu"] = round(float(s1_val), 1)

    by_s1 = sorted(horses_data, key=lambda x: x["stage1_shisu"], reverse=True)
    for idx, h in enumerate(by_s1):
        h["stage1_rank"] = idx + 1

    probs = [h["base_prob"] for h in horses_data]
    mean_prob = np.mean(probs) if probs else 0
    std_prob = np.std(probs) if probs else 1
    total_prob = sum(probs) if sum(probs) > 0 else 1.0

    for h in horses_data:
        shisu = ((h["base_prob"] - mean_prob) / std_prob) * 10 + 50 if std_prob > 0 else 50.0
        h["shisu"] = round(float(shisu), 1)
        h["shisu_diff"] = round(h["shisu"] - h["stage1_shisu"], 1)
        h["norm_win_prob"] = h["base_prob"] / total_prob

    myoumi_values = []
    for h in horses_data:
        if h["odds"] > 0:
            effective_odds = min(float(h["odds"]), 35.0)
            ability_weight = 1.0 if h["shisu"] >= 45.0 else 0.5
            ev = h["norm_win_prob"] * effective_odds * ability_weight
            h["raw_ev"] = ev
            myoumi_values.append(ev)
        else:
            h["raw_ev"] = 0.0

    if myoumi_values:
        m_mean = np.mean(myoumi_values); m_std = np.std(myoumi_values)
        for h in horses_data:
            if h["odds"] > 0:
                m_shisu = ((h["raw_ev"] - m_mean) / m_std) * 10 + 50 if m_std > 0 else 50.0
                h["myoumi"] = round(float(m_shisu), 1)
            else:
                h["myoumi"] = 0.0
    else:
        for h in horses_data: h["myoumi"] = 0.0

    horses_data.sort(key=lambda x: x["shisu"], reverse=True)
    for idx, h in enumerate(horses_data):
        h["stage2_rank"] = idx + 1

    betting_tickets = build_betting_recommendations(horses_data)

    return {
        "status": "success",
        "diagnosis": horses_data,
        "betting_tickets": betting_tickets,
        "req_up": round(avg_req_up, 2),
        "pace_f": round(avg_pace_f, 1),
        "pace_l": round(avg_pace_l, 1),
        "rpci": base_rpci,
        "rpci_type": base_rpci_type
    }

@app.post("/simulate_race")
def simulate_race(request: RaceRequest):
    horses_data = []
    course_enc = safe_encode('course', request.course_type)
    weather_enc = safe_encode('weather', request.weather)
    cond_enc = safe_encode('condition', request.track_condition)
    place_code = request.race_id[4:6] if len(request.race_id) >= 6 else "05"
    course_key = f"{place_code}_{request.course_type}_{request.distance}"
    
    c_stats = course_history_db.get(course_key, {})
    top3_up = c_stats.get('top3_up', [])
    pace_f = c_stats.get('pace_first', [])
    pace_l = c_stats.get('pace_last', [])
    expected_standard_time = get_expected_standard_time(c_stats, request.course_type, request.distance)
    
    avg_req_up = float(np.mean(top3_up)) if top3_up else 35.0
    avg_pace_f = float(np.mean(pace_f)) if pace_f else 35.0
    avg_pace_l = float(np.mean(pace_l)) if pace_l else 35.0
    base_rpci = round((avg_pace_f / avg_pace_l) * 50.0, 1) if avg_pace_l > 0 else 50.0
    
    avg_pace_f, avg_pace_l, avg_req_up = adjust_pace_times(request.expected_pace, avg_pace_f, avg_pace_l, avg_req_up)
    
    for horse in request.horses:
        h_stats = horse_history_db.get(horse.bamei, {})
        zenso, real_agari, kaisuu, power_score, makuri_flag = get_horse_real_stats(horse.bamei)
        effective_zenso_for_lgb = min(zenso, 6.0) if power_score >= 0.78 else zenso

        course_up_times = h_stats.get('course_up_times', {}).get(course_key, [])
        course_up = float(np.mean(course_up_times)) if course_up_times else real_agari
        time_diff = course_up - avg_req_up
        kishu_enc = safe_encode('kishu', horse.kishu)
        
        time_limit_mult = get_time_limit_multiplier(
            horse.bamei, request.course_type, request.distance, 
            expected_standard_time, request.cushion_value, request.track_condition
        )
        
        input_data = pd.DataFrame([{
            '枠番': horse.wakuban, '馬番': horse.umaban, '斤量': horse.kinryou, '馬体重_数値': horse.bataijuu,
            'distance': request.distance, 'course_type_enc': course_enc, 'weather_enc': weather_enc,
            'track_condition_enc': cond_enc, '舞台_平均前半ペース': avg_pace_f, '舞台_平均後半ペース': avg_pace_l,
            '舞台_平均要求上がり': avg_req_up, '過去基本脚質': horse.kyakushitsu, '前走着順': effective_zenso_for_lgb,
            '近3走_平均上がり': real_agari, '同舞台_平均上がり': course_up, '出走回数': kaisuu,
            '上がりタイム差分': time_diff, 'kishu_enc': kishu_enc
        }])
        
        raw_pred = float(model.predict(input_data)[0])
        h_pci, h_pci_type = get_horse_pci_stats(horse.bamei)

        final_score, _, _, _ = calculate_comprehensive_score(
            raw_pred, horse, power_score, makuri_flag, h_pci, h_pci_type, time_diff, 
            request.expected_pace, request.track_bias, request.bias_strength, request.cushion_value,
            base_rpci, time_limit_mult
        )
        
        horses_data.append({
            "umaban": int(horse.umaban), "bamei": horse.bamei, "odds": float(horse.odds),
            "base_prob": final_score, "time_diff": round(time_diff, 2),
            "pci": h_pci, "pci_type": h_pci_type
        })
    
    probs = [h["base_prob"] for h in horses_data]
    mean_prob = np.mean(probs) if probs else 0
    std_prob = np.std(probs) if probs else 1
    total_prob = sum(probs) if sum(probs) > 0 else 1.0

    for h in horses_data:
        shisu = ((h["base_prob"] - mean_prob) / std_prob) * 10 + 50 if std_prob > 0 else 50.0
        h["shisu"] = round(shisu, 1)
        h["norm_win_prob"] = h["base_prob"] / total_prob

    myoumi_values = []
    for h in horses_data:
        if h["odds"] > 0:
            effective_odds = min(float(h["odds"]), 35.0)
            ability_weight = 1.0 if h["shisu"] >= 45.0 else 0.5
            ev = h["norm_win_prob"] * effective_odds * ability_weight
            h["raw_ev"] = ev
            myoumi_values.append(ev)
        else:
            h["raw_ev"] = 0.0

    if myoumi_values:
        m_mean = np.mean(myoumi_values); m_std = np.std(myoumi_values)
        for h in horses_data:
            if h["odds"] > 0:
                h["myoumi"] = ((h["raw_ev"] - m_mean) / m_std) * 10 + 50 if m_std > 0 else 50.0
            else:
                h["myoumi"] = 50.0
    else:
        for h in horses_data: h["myoumi"] = 50.0

    sim_scores = []
    for h in horses_data:
        target_index = (h["shisu"] * 0.85) + (h["myoumi"] * 0.15)
        noise = np.random.normal(0, 3.2)
        score = target_index + noise
        sim_scores.append({
            "umaban": h["umaban"], "bamei": h["bamei"],
            "shisu": h["shisu"], "time_diff": h["time_diff"], "score": score
        })
        
    sim_scores.sort(key=lambda x: x["score"], reverse=True)
    
    results = []
    for i, sim in enumerate(sim_scores):
        results.append({
            "umaban": sim["umaban"], "bamei": sim["bamei"],
            "shisu": sim["shisu"], "time_diff": sim["time_diff"], "rank": i + 1
        })
    
    return {
        "status": "success", "ranking": results, "req_up": round(avg_req_up, 2),
        "pace_f": round(avg_pace_f, 1), "pace_l": round(avg_pace_l, 1)
    }
