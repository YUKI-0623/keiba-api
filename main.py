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

app = FastAPI(title="競馬シミュレーションAPI (プロ仕様完全版: 全自動環境判定/2段階指数/3連系馬券構築統合)")

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
    """DBにclass_factorが存在しない場合のフォールバック判定"""
    if 'class_factor' in record and record['class_factor'] is not None:
        try:
            return float(record['class_factor'])
        except:
            pass
    r_name = str(record.get('race_name', '')) + str(record.get('grade', ''))
    if 'G1' in r_name or 'GI' in r_name: return 1.20
    if 'G2' in r_name or 'GII' in r_name: return 1.10
    if 'G3' in r_name or 'GIII' in r_name or 'OP' in r_name or 'L' in r_name or 'ステークス' in r_name: return 1.00
    if '3勝' in r_name or '1600万' in r_name: return 0.90
    if '2勝' in r_name or '1000万' in r_name: return 0.80
    if '1勝' in r_name or '500万' in r_name: return 0.70
    if '未勝利' in r_name or '新馬' in r_name: return 0.60
    return 0.75

def infer_margin_seconds(record: dict) -> float:
    """DBにmarginが存在しない場合のフォールバック判定（マイナス着差は0.0に正規化）"""
    if 'margin' in record and record['margin'] is not None:
        try:
            val = float(record['margin'])
            return max(0.0, val)
        except:
            pass
    try:
        rank = float(record.get('rank', 8.0))
        if rank <= 1.0: return 0.0
        elif rank <= 3.0: return 0.2
        elif rank <= 5.0: return 0.5
        elif rank <= 8.0: return 0.8
        else: return 1.3
    except:
        return 0.8

def calc_jra_wakuban(umaban: int, total_horses: int) -> int:
    """JRA公式ルールに基づく頭数別の正確な枠番算出フォールバック"""
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
    try: zenso_chakujun = float(records[-1].get('rank', 8.0))
    except: zenso_chakujun = 8.0
    
    up_list = []
    for r in records[-3:]:
        try:
            u = float(r.get('up', 0))
            if 30.0 <= u <= 45.0: up_list.append(u)
        except: pass
    avg_agari = round(float(np.mean(up_list)), 2) if up_list else 35.0

    makuri_flag = False
    class_based_scores = []
    
    for r in records[-5:]:
        passing = str(r.get('passing', ''))
        if '-' in passing:
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
        run_score = float(np.clip(cf - (adjusted_margin * 0.15), 0.10, 1.20))
        class_based_scores.append(run_score)

    if class_based_scores:
        weights = np.arange(1, len(class_based_scores) + 1)
        power_score = float(np.average(class_based_scores, weights=weights))
    else:
        power_score = 0.50

    return zenso_chakujun, avg_agari, shussou_kaisuu, power_score, makuri_flag

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
# 自動判定ロジック①: 当日馬体重の増減から状態補正(condition_override)を自動判定
# =====================================================================
def infer_auto_condition_override(
    bamei: str, weight_val: float, change_val: float, weight_announced: bool,
    zenso: float, power_score: float, kaisuu: int
) -> tuple[str, str]:
    """馬体重と過去実績から状態補正倍率と理由ラベルを返す（未発表時は1.00に安全フォールバック）"""
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

    # 1. 大幅馬体減 (-8kg以下)
    if change_val <= -8.0:
        if weight_val <= 465.0 or change_val <= -10.0:
            return "0.88", f"大幅馬体減({int(change_val)}kg)・輸送細化懸念"
        elif zenso >= 6.0 and weight_val >= 485.0 and power_score >= 0.65:
            return "1.04", f"大型馬の絞り込み({int(change_val)}kg)・太め解消"
        else:
            return "0.88", f"馬体減({int(change_val)}kg)・輸送細化判定"

    # 2. やや馬体減 (-6kg 〜 -4kg)
    if -6.0 <= change_val <= -4.0:
        if best_weight and abs(weight_val - best_weight) <= 2.0 and zenso >= 4.0:
            return "1.04", "ベスト体重へ絞り込み・太め解消"
        if weight_val <= 424.0:
            return "0.90", f"小柄な馬の馬体減({int(change_val)}kg)"
        return "1.00", "標準範囲内"

    # 3. 増減ほぼなし (-2kg 〜 +2kg)
    if -2.0 <= change_val <= 2.0:
        if power_score >= 0.68 and zenso <= 3.0 and kaisuu >= 2:
            return "1.05", f"好調維持({change_val:+.0f}kg)・究極仕上げ"
        return "1.00", "馬体安定（標準）"

    # 4. 小〜中幅プラス (+4kg 〜 +10kg)
    if 4.0 <= change_val <= 10.0:
        if best_weight and abs(weight_val - best_weight) <= 3.0:
            return "1.04", f"ベスト体重合致({change_val:+.0f}kg)・太め解消/充実"
        if 6.0 <= change_val <= 8.0 and 485.0 <= weight_val <= 506.0 and power_score >= 0.58:
            return "1.04", f"馬体充実({change_val:+.0f}kg)・太め解消判定"
        if zenso <= 4.0 or power_score >= 0.62:
            return "1.03", f"好走圏内の馬体増({change_val:+.0f}kg)・成長分"
        if change_val >= 8.0 and zenso >= 8.0 and power_score < 0.52:
            return "0.92", f"凡走後の馬体増({change_val:+.0f}kg)・太め残り懸念"
        return "1.00", "標準範囲内"

    # 5. 大幅プラス (+12kg以上)
    if change_val >= 12.0:
        if power_score >= 0.66 and zenso <= 3.0:
            return "1.03", f"実績馬の馬体増({change_val:+.0f}kg)・成長分"
        return "0.92", f"大幅馬体増({change_val:+.0f}kg)・太め残り判定"

    return "1.00", "標準"


# =====================================================================
# 自動判定ロジック②: 出走馬の能力・脚質・PCIから予想展開(expected_pace)を自動判定
# =====================================================================
def auto_detect_expected_pace(horses_data: list, course_rpci: float) -> tuple[str, str, str]:
    """出走メンバーの脚質分布・好走PCI・マクリ実績と舞台RPCIから展開シナリオを自動判定"""
    if not horses_data or len(horses_data) < 2:
        return "M_JIRIKI", "manual_required", "⚠️ 出走馬データが不足しているため標準ペース（ミドル①）にセットしました。"

    total_h = len(horses_data)
    with_history = sum(1 for h in horses_data if h.get("shussou_kaisuu", 0) > 0)
    if with_history < max(2, total_h // 3):
        _, _, rec_p = get_course_rpci_stats_from_val(course_rpci)
        return rec_p, "fallback_course", f"ℹ️ 過去走データが少ない（新馬・未出走多数）ため、舞台RPCI({course_rpci})基準で自動セットしました。必要に応じ手動変更してください。"

    nige_cnt = sum(1 for h in horses_data if int(round(h.get("kyakushitsu", 2))) == 1)
    senkou_cnt = sum(1 for h in horses_data if int(round(h.get("kyakushitsu", 2))) == 2)
    makuri_cnt = sum(1 for h in horses_data if h.get("makuri_flag", False))
    front_ratio = (nige_cnt + senkou_cnt) / float(total_h)
    avg_pci = float(np.mean([h.get("pci", 50.0) for h in horses_data]))

    if nige_cnt >= 3 or (nige_cnt >= 2 and front_ratio >= 0.45) or (course_rpci < 48.0 and nige_cnt >= 2):
        pace = "H_MAEKUZURE"
        msg = f"✅ 展開自動判定: 逃げ{nige_cnt}頭・先行{senkou_cnt}頭（先行勢{int(front_ratio*100)}%）で先行激化 ➔ 『ハイ① 前崩れ消耗戦』を自動セット"
    elif (nige_cnt >= 2 and makuri_cnt >= 2) or (front_ratio >= 0.36 and avg_pci < 50.3) or (48.0 <= course_rpci < 49.8 and nige_cnt >= 1):
        pace = "M_LONSPA"
        msg = f"✅ 展開自動判定: 逃げ{nige_cnt}頭・先行{senkou_cnt}頭 / マクリ実績{makuri_cnt}頭 ➔ 『ミドル② タフ持続ロンスパ戦』を自動セット"
    elif (nige_cnt == 0 and front_ratio <= 0.25 and avg_pci >= 51.8) or (nige_cnt <= 1 and course_rpci > 53.0 and makuri_cnt == 0):
        pace = "S_YOIDON"
        msg = f"✅ 展開自動判定: 逃げ{nige_cnt}頭・先行{senkou_cnt}頭と手薄＆平均PCI {avg_pci:.1f} ➔ 『スロー① 直線ヨーイドン』を自動セット"
    elif nige_cnt <= 1 and front_ratio <= 0.32 and (makuri_cnt >= 1 or 51.5 <= course_rpci <= 53.2):
        pace = "S_LONSPA"
        msg = f"✅ 展開自動判定: 逃げ{nige_cnt}頭で道中緩むがマクリ・持続力馬あり ➔ 『スロー② ロンスパ・マクリ戦』を自動セット"
    else:
        pace = "M_JIRIKI"
        msg = f"✅ 展開自動判定: 逃げ{nige_cnt}頭・先行{senkou_cnt}頭（先行勢{int(front_ratio*100)}%）の標準隊列 ➔ 『ミドル① 地力・総合力勝負』を自動セット"

    return pace, "auto_detected", msg

def get_course_rpci_stats_from_val(rpci: float):
    rpci_type = get_pci_label(rpci)
    if rpci < 48.0: rec_pace = "H_MAEKUZURE"
    elif rpci < 49.8: rec_pace = "M_LONSPA"
    elif rpci <= 52.0: rec_pace = "M_JIRIKI"
    elif rpci <= 53.2: rec_pace = "S_LONSPA"
    else: rec_pace = "S_YOIDON"
    return rpci, rpci_type, rec_pace


# =====================================================================
# 自動判定ロジック③: 前日・当日のレース結果からトラックバイアス＆強度を自動判定
# （土曜開催・結果確定前・過去データなし時はエラーを出さず手動設定へ安全フォールバック）
# =====================================================================
def parse_single_race_result_for_bias(res_html: str, target_track_type: str) -> Optional[List[Dict[str, int]]]:
    """result.htmlから同馬場種別(芝/ダ)の1〜3着馬の枠番・4角通過順位を抽出"""
    if "HorseList" not in res_html:
        return None
    m_tr = re.search(r'(芝|ダ|障害)(?:専門)?(\d{3,4})m', res_html)
    if not m_tr:
        return None
    r_type = "障" if m_tr.group(1) == "障害" else m_tr.group(1)
    if r_type != target_track_type:
        return []  # 馬場種別が違う場合は空リスト（レース自体は確定済み）

    rows = re.findall(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', res_html, re.DOTALL | re.IGNORECASE)
    if len(rows) < 3:
        return None

    total_h = len(rows)
    top3_info = []
    for row in rows[:3]:
        # 枠番抽出
        waku = 0
        w_m = re.search(r'<td[^>]*class="[^"]*Waku[^"]*"[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
        if w_m:
            w_s = re.sub(r'<[^>]*>', '', w_m.group(1)).strip()
            if w_s.isdigit(): waku = int(w_s)
        if waku == 0:
            u_m = re.search(r'<td[^>]*class="[^"]*Umaban[^"]*"[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
            if not u_m:
                u_m = re.search(r'<td[^>]*class="[^"]*Num Txt_C[^"]*"[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
            if u_m:
                u_s = re.sub(r'<[^>]*>', '', u_m.group(1)).strip()
                if u_s.isdigit(): waku = calc_jra_wakuban(int(u_s), total_h)

        # 通過順位抽出
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


def auto_detect_track_bias(race_id: str, target_track_type: str, shutuba_html: str) -> tuple[str, str, str, str]:
    """
    当日および前日の確定済みレース結果からトラックバイアスと強度を自動判定。
    土曜開催（金曜日はレースなし）や結果未確定時は、エラーで止めずに手動設定案内を返す。
    """
    try:
        if len(race_id) < 12 or not race_id.isdigit():
            return "フラット", "中", "manual_required", "⚠️ レースID形式から過去レースを参照できないため、トラックバイアスは手動で設定してください。"

        prefix_8 = race_id[:8]          # YYYYPPKK
        day_num = int(race_id[8:10])    # 日目 (01〜12)
        race_num = int(race_id[10:12])  # R (01〜12)

        # 土曜開催かどうかの判定（HTML内の曜日表記、または開幕初日 day_num == 1）
        is_saturday = ("(土)" in shutuba_html) or ("（土）" in shutuba_html) or (day_num == 1)

        headers = {'User-Agent': 'Mozilla/5.0'}
        collected_top3 = []
        sampled_race_count = 0

        # 1. まず当日の直前までのレースを確認（まず1Rまたは直前Rで当日開催が確定しているかチェック）
        same_day_started = False
        if race_num >= 2:
            test_r = 1 if race_num <= 4 else max(1, race_num - 2)
            test_id = f"{prefix_8}{day_num:02d}{test_r:02d}"
            try:
                time.sleep(0.25)
                r_res = requests.get(f"https://race.netkeiba.com/race/result.html?race_id={test_id}", headers=headers, timeout=4)
                if r_res.status_code == 200:
                    r_res.encoding = r_res.apparent_encoding
                    if "Result_Num" in r_res.text or "PassageRate" in r_res.text:
                        same_day_started = True
                        parsed = parse_single_race_result_for_bias(r_res.text, target_track_type)
                        if parsed:
                            collected_top3.extend(parsed)
                            sampled_race_count += 1
            except:
                pass

        # 当日のレースが進行中・確定済みの場合は、直前レースから遡って最大4鞍まで同コース種別を収集
        if same_day_started:
            for r_no in range(race_num - 1, 0, -1):
                if sampled_race_count >= 4:
                    break
                if r_no == (1 if race_num <= 4 else max(1, race_num - 2)):
                    continue
                cid = f"{prefix_8}{day_num:02d}{r_no:02d}"
                try:
                    time.sleep(0.25)
                    r_res = requests.get(f"https://race.netkeiba.com/race/result.html?race_id={cid}", headers=headers, timeout=4)
                    if r_res.status_code == 200:
                        r_res.encoding = r_res.apparent_encoding
                        parsed = parse_single_race_result_for_bias(r_res.text, target_track_type)
                        if parsed:
                            collected_top3.extend(parsed)
                            sampled_race_count += 1
                except:
                    continue

        # 2. 当日の同コース確定レースがまだ足りない場合（前日夜・朝イチなど）
        if sampled_race_count < 2:
            if is_saturday:
                # 土曜日の場合は金曜日に中央競馬のレースが存在しないため、安全に手動設定へフォールバック
                return (
                    "フラット", "中", "manual_required",
                    "⚠️ 土曜開催（前日金曜のレースなし）かつ当日の同コース確定レースがまだないため、トラックバイアスは手動で設定してください。"
                )
            elif day_num >= 2:
                # 日曜・月曜開催の場合は前日(day_num - 1)の後半レース(12R→7R)から最大4鞍を参照
                prev_day = day_num - 1
                for r_no in range(12, 6, -1):
                    if sampled_race_count >= 4:
                        break
                    cid = f"{prefix_8}{prev_day:02d}{r_no:02d}"
                    try:
                        time.sleep(0.25)
                        r_res = requests.get(f"https://race.netkeiba.com/race/result.html?race_id={cid}", headers=headers, timeout=4)
                        if r_res.status_code == 200:
                            r_res.encoding = r_res.apparent_encoding
                            if "Result_Num" not in r_res.text and "PassageRate" not in r_res.text:
                                break
                            parsed = parse_single_race_result_for_bias(r_res.text, target_track_type)
                            if parsed:
                                collected_top3.extend(parsed)
                                sampled_race_count += 1
                    except:
                        continue

        if sampled_race_count < 2 or len(collected_top3) < 4:
            return (
                "フラット", "中", "manual_required",
                "⚠️ 参照できる前日・当日の同コース確定レース結果がまだありません。トラックバイアスは手動で設定してください。"
            )

        # 3. 集計した1〜3着馬の枠番・通過順位からバイアスと強度を判定
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

        summary_msg = (
            f"✅ 直近の同コース確定結果（計{sampled_race_count}鞍・上位{total_horses}頭："
            f"内枠率{int(inner_rate*100)}% / 4角4番手以内率{int(front_rate*100)}%）から"
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
        time.sleep(random.uniform(1.5, 3.0)) 
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

        horses_data = []
        horse_blocks = re.findall(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
        
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
                o_m = re.search(r'<span[^>]*id="odds[^>]*>(.*?)</span>', block, re.IGNORECASE | re.DOTALL)
                if o_m:
                    o_str = re.sub(r'<[^>]*>', '', o_m.group(1)).strip()
                    try: odds = float(o_str)
                    except: pass

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

        # 1. 出走メンバー能力＆脚質から予想ペースを自動判定
        auto_pace, pace_status, pace_msg = auto_detect_expected_pace(horses_data, rpci)

        # 2. 前日・当日のレース結果からトラックバイアス＆強度を自動判定（土曜等で未確定時は手動案内へフォールバック）
        auto_bias, auto_strength, bias_status, bias_msg = auto_detect_track_bias(race_id, track_type, html)

        # 3. 馬体重発表ステータスメッセージ
        if any_weight_announced:
            weight_status = "auto_detected"
            weight_msg = "✅ 当日馬体重の増減と過去好走実績から、全頭の『状態補正（太め解消・成長分・輸送細化等）』を自動セットしました。"
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
    """
    【第1段階: 素の地力診断】
    枠番・トラックバイアス・当日馬体重増減・展開バイアス・持ち時計壁を加味しない純粋な地力スコア
    """
    if horse.odds > 1.0:
        market_prob = min(0.55, 0.80 / float(horse.odds))
    else:
        market_prob = 0.08

    agari_factor = 1.0
    if time_diff_base <= -0.3: agari_factor = 1.05
    elif time_diff_base <= 0.0: agari_factor = 1.02
    elif time_diff_base >= 0.6: agari_factor = 0.95

    pure_ability = ((raw_lgb_pred_neutral * 0.42) + (power_score * 0.38) + (market_prob * 0.20)) * agari_factor
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
        base_ability = (raw_lgb_pred * 0.45) + (power_score * 0.35) + (market_prob * 0.20)
    else:
        base_ability = (raw_lgb_pred * 0.50) + (power_score * 0.30) + (market_prob * 0.20)

    kyaku = int(round(horse.kyakushitsu))
    waku = horse.wakuban

    pace_mult = 1.0
    if expected_pace in ["S_YOIDON", "S"]:
        if h_pci_type == "C": pace_mult *= 1.12
        elif h_pci_type == "B": pace_mult *= 1.00
        else: pace_mult *= 0.90
        if kyaku == 1: pace_mult *= 1.06
        elif kyaku == 2: pace_mult *= 1.08
        elif kyaku == 3: pace_mult *= 1.04
        elif kyaku == 4: pace_mult *= 0.88
        if time_diff <= -0.3: pace_mult *= 1.08
        elif time_diff <= 0.1: pace_mult *= 1.04
        elif time_diff >= 0.6: pace_mult *= 0.93

    elif expected_pace == "S_LONSPA":
        if 49.5 <= h_pci <= 54.0: pace_mult *= 1.10
        elif h_pci_type == "C": pace_mult *= 1.04
        else: pace_mult *= 0.94
        if kyaku == 1: pace_mult *= 0.96
        elif kyaku == 2: pace_mult *= 1.07
        elif kyaku == 3: pace_mult *= 1.08
        elif kyaku == 4: pace_mult *= 0.95
        if makuri_flag: pace_mult *= 1.07

    elif expected_pace in ["M_JIRIKI", "M"]:
        if h_pci_type == "B": pace_mult *= 1.08
        else: pace_mult *= 0.98
        if kyaku == 1: pace_mult *= 0.98
        elif kyaku == 2: pace_mult *= 1.08
        elif kyaku == 3: pace_mult *= 1.04
        elif kyaku == 4: pace_mult *= 0.92
        if power_score >= 0.70: pace_mult *= 1.06

    elif expected_pace == "M_LONSPA":
        if 46.5 <= h_pci <= 50.8: pace_mult *= 1.10
        elif h_pci_type == "A": pace_mult *= 1.04
        else: pace_mult *= 0.92
        if kyaku == 1: pace_mult *= 0.91
        elif kyaku == 2: pace_mult *= 1.04
        elif kyaku == 3: pace_mult *= 1.09
        elif kyaku == 4: pace_mult *= 1.02
        if makuri_flag: pace_mult *= 1.05

    elif expected_pace in ["H_MAEKUZURE", "H"]:
        if h_pci_type == "A": pace_mult *= 1.14
        elif h_pci_type == "B": pace_mult *= 1.00
        else: pace_mult *= 0.88
        if kyaku == 1: pace_mult *= 0.84
        elif kyaku == 2: pace_mult *= 0.92
        elif kyaku == 3: pace_mult *= 1.09
        elif kyaku == 4: pace_mult *= 1.13

    if course_rpci > 0 and h_pci > 0:
        if abs(course_rpci - h_pci) >= 3.0:
            pace_mult *= 0.92

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
    capped_mult = max(0.75, min(1.25, total_mult))

    return base_ability * capped_mult, round(bias_mult, 3), round(pace_mult, 3), round(condition_mult, 3)


# =====================================================================
# 2段階指数による4分類 ＆ 全5種・7パターンの推奨買い目自動構築
# =====================================================================
def build_betting_recommendations(horses_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stage 1（素の地力指数）と Stage 2（最終指数・妙味指数・持ち時計壁）から、
    【1着軸】【2列目・相手本線】【3列目・地力救済ヒモ】【完全消し】に分類し、
    単勝・複勝・ワイド・3連複(2種)・3連単(2種)の買い目と点数を生成する。
    """
    if not horses_data:
        return {}

    # Stage 2 (shisu) 降順でソート済みのリストを前提
    by_stage2 = sorted(horses_data, key=lambda x: (x["shisu"], x["myoumi"]), reverse=True)
    by_stage1 = sorted(horses_data, key=lambda x: x["stage1_shisu"], reverse=True)

    # 1. 👑 1着軸 (AXIS_1ST) の選定: 持ち時計の壁をクリアしている最上位馬
    axis_horse = None
    for h in by_stage2:
        if h.get("time_limit_mult", 1.0) >= 0.94:
            axis_horse = h
            break
    if axis_horse is None:
        axis_horse = by_stage2[0]
    axis_umaban = axis_horse["umaban"]

    # 2. 💀 完全消し (ELIMINATED) の判定: 持ち時計壁アウト(<=0.88) または 大幅馬体減(<=0.88)で指数下位
    eliminated_umabans = set()
    for h in by_stage2:
        if h["umaban"] == axis_umaban:
            continue
        t_mult = h.get("time_limit_mult", 1.0)
        c_mult = h.get("condition_mult", 1.0)
        if t_mult <= 0.88:
            eliminated_umabans.add(h["umaban"])
            h["role_code"] = "ELIMINATED"
            h["role_label"] = "💀 完全消し (持ち時計不足)"
        elif c_mult <= 0.88 and h["shisu"] < 52.0:
            eliminated_umabans.add(h["umaban"])
            h["role_code"] = "ELIMINATED"
            h["role_label"] = "💀 完全消し (大幅馬体減)"

    # 3. 🚀 2列目・相手本線 (MAIN_2ND) の選定 (最大3頭: Stage 2上位 ＆ バイアス・妙味合致)
    main_2nd_horses = []
    for h in by_stage2:
        u = h["umaban"]
        if u == axis_umaban or u in eliminated_umabans:
            continue
        if len(main_2nd_horses) < 3 and (h["shisu"] >= 53.0 or (h["myoumi"] >= 57.0 and h["shisu"] >= 50.0)):
            main_2nd_horses.append(h)

    # 万が一2列目が2頭未満の場合はStage2上位から補충
    if len(main_2nd_horses) < 2:
        for h in by_stage2:
            u = h["umaban"]
            if u != axis_umaban and u not in eliminated_umabans and h not in main_2nd_horses:
                main_2nd_horses.append(h)
                if len(main_2nd_horses) >= 2:
                    break

    main_2nd_umabans = [h["umaban"] for h in main_2nd_horses]

    # 4. 🛡️ 3列目・地力救済ヒモ (SAVER_3RD) の選定
    # 条件A: Stage 2では沈んだが、Stage 1(素の地力指数)が上位6頭以内(または53.0以上)で、持ち時計壁はクリアしている実力馬（ドリームコア枠）
    # 条件B: Stage 2で準上位(52.0以上)かつ時計壁クリアの馬
    saver_3rd_horses = []
    for h in by_stage1:
        u = h["umaban"]
        if u == axis_umaban or u in main_2nd_umabans or u in eliminated_umabans:
            continue
        if (h["stage1_rank"] <= 6 or h["stage1_shisu"] >= 53.0) and h.get("time_limit_mult", 1.0) >= 0.94:
            saver_3rd_horses.append(h)
            h["is_bias_victim_rescued"] = True
            if len(saver_3rd_horses) >= 3:
                break

    for h in by_stage2:
        u = h["umaban"]
        if u == axis_umaban or u in main_2nd_umabans or u in eliminated_umabans:
            continue
        if h not in saver_3rd_horses and h["shisu"] >= 53.0 and len(saver_3rd_horses) < 4:
            saver_3rd_horses.append(h)

    saver_3rd_umabans = [h["umaban"] for h in saver_3rd_horses]

    # 各馬へのロールラベル付与
    for h in horses_data:
        u = h["umaban"]
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
            h["role_code"] = "SAVER_3RD"
            if h.get("is_bias_victim_rescued"):
                h["role_label"] = f"🛡️ 3列目救済 (地力{h['stage1_rank']}位・バイアス減点)"
            else:
                h["role_label"] = "🛡️ 3列目押さえ (地力・展開圏内)"
        elif u not in eliminated_umabans:
            h["role_code"] = "OTHER"
            h["role_label"] = "静観 (指数下位)"

    # =================================================================
    # 各券種の買い目・組み合わせ点数の厳密計算
    # =================================================================
    row1 = [axis_umaban]
    row2 = main_2nd_umabans
    row3 = main_2nd_umabans + saver_3rd_umabans  # 3列目は「2列目本線 ＋ 3列目救済馬」

    # 1. 単勝（1着軸 ＋ 相手本線の妙味最上位）
    value_pick = None
    cands_for_val = [h for h in main_2nd_horses if h["odds"] >= 6.0 and h["myoumi"] >= 55.0]
    if cands_for_val:
        value_pick = sorted(cands_for_val, key=lambda x: x["myoumi"], reverse=True)[0]
    tansho_list = [{"umaban": axis_umaban, "bamei": axis_horse["bamei"], "type": "本命単勝", "odds": axis_horse["odds"]}]
    if value_pick and value_pick["umaban"] != axis_umaban:
        tansho_list.append({"umaban": value_pick["umaban"], "bamei": value_pick["bamei"], "type": "妙味穴単勝", "odds": value_pick["odds"]})

    # 2. 複勝（1着軸 ＋ 相手本線の妙味上位1〜2頭）
    fukusho_list = [{"umaban": axis_umaban, "bamei": axis_horse["bamei"], "type": "鉄板軸", "odds": axis_horse["odds"]}]
    for mh in sorted(main_2nd_horses, key=lambda x: x["myoumi"], reverse=True)[:2]:
        fukusho_list.append({"umaban": mh["umaban"], "bamei": mh["bamei"], "type": "妙味複勝", "odds": mh["odds"]})

    # 3. ワイド（1着軸 流し ➔ 2列目相手本線）
    wide_combos = [f"{axis_umaban} - {u}" for u in row2]

    # 4. 3連複（1頭軸流し: 軸 row1 ➔ 相手 row3）
    sanrenpuku_nagashi_combos = []
    for i in range(len(row3)):
        for j in range(i + 1, len(row3)):
            pair = sorted([row3[i], row3[j]])
            sanrenpuku_nagashi_combos.append(f"{axis_umaban} - {pair[0]} - {pair[1]}")

    # 5. 3連複（フォーメーション: 1列目 row1 / 2列目 row2 / 3列目 row3）
    sanrenpuku_form_set = set()
    for a in row1:
        for b in row2:
            for c in row3:
                if len({a, b, c}) == 3:
                    sanrenpuku_form_set.add(tuple(sorted([a, b, c])))
    sanrenpuku_form_combos = [f"{t[0]} - {t[1]} - {t[2]}" for t in sorted(sanrenpuku_form_set)]

    # 6. 3連単（1頭軸流し・1着固定: 1着 row1 ➔ 2・3着 row3）
    sanrentan_nagashi_combos = []
    for b in row3:
        for c in row3:
            if b != c:
                sanrentan_nagashi_combos.append(f"{axis_umaban} → {b} → {c}")

    # 7. 3連単（フォーメーション: 1着 row1 / 2着 row2 / 3着 row3）
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
            "saver_3rd": [{"umaban": h["umaban"], "bamei": h["bamei"], "shisu": h["shisu"], "stage1_shisu": h["stage1_shisu"], "stage1_rank": h["stage1_rank"], "odds": h["odds"], "rescued": h.get("is_bias_victim_rescued", False)} for h in saver_3rd_horses],
            "eliminated": eliminated_details
        },
        "tansho": {
            "picks": tansho_list,
            "count": len(tansho_list)
        },
        "fukusho": {
            "picks": fukusho_list,
            "count": len(fukusho_list)
        },
        "wide": {
            "axis": axis_umaban,
            "targets": row2,
            "combos": wide_combos,
            "count": len(wide_combos)
        },
        "sanrenpuku_nagashi": {
            "axis": axis_umaban,
            "targets": row3,
            "combos": sanrenpuku_nagashi_combos,
            "count": len(sanrenpuku_nagashi_combos)
        },
        "sanrenpuku_formation": {
            "row1": row1,
            "row2": row2,
            "row3": row3,
            "combos": sanrenpuku_form_combos,
            "count": len(sanrenpuku_form_combos)
        },
        "sanrentan_nagashi": {
            "first": axis_umaban,
            "targets": row3,
            "combos": sanrentan_nagashi_combos,
            "count": len(sanrentan_nagashi_combos)
        },
        "sanrentan_formation": {
            "first": row1,
            "second": row2,
            "third": row3,
            "combos": sanrentan_form_combos,
            "count": len(sanrentan_form_combos)
        }
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

        course_up_times = h_stats.get('course_up_times', {}).get(course_key, [])
        course_up = float(np.mean(course_up_times)) if course_up_times else real_agari
        time_diff_base = course_up - base_req_up
        time_diff = course_up - avg_req_up
        kishu_enc = safe_encode('kishu', horse.kishu)
        
        time_limit_mult = get_time_limit_multiplier(
            horse.bamei, request.course_type, request.distance, 
            expected_standard_time, request.cushion_value, request.track_condition
        )
        
        # --- Stage 1: 枠番・当日馬体重・展開バイアスを中立化した素の地力推論 ---
        stage1_input = pd.DataFrame([{
            '枠番': 4.0, '馬番': 8.0, '斤量': horse.kinryou, '馬体重_数値': 480.0,
            'distance': request.distance, 'course_type_enc': course_enc, 'weather_enc': weather_enc,
            'track_condition_enc': cond_enc, '舞台_平均前半ペース': base_pace_f, '舞台_平均後半ペース': base_pace_l,
            '舞台_平均要求上がり': base_req_up, '過去基本脚質': horse.kyakushitsu, '前走着順': zenso,
            '近3走_平均上がり': real_agari, '同舞台_平均上がり': course_up, '出走回数': kaisuu,
            '上がりタイム差分': time_diff_base, 'kishu_enc': kishu_enc
        }])
        raw_pred_stage1 = float(model.predict(stage1_input)[0])
        stage1_prob = calculate_stage1_pure_score(raw_pred_stage1, horse, power_score, time_diff_base)

        # --- Stage 2: 枠番・バイアス・馬体重・展開・持ち時計限界を加味した最終推論 ---
        input_data = pd.DataFrame([{
            '枠番': horse.wakuban, '馬番': horse.umaban, '斤量': horse.kinryou, '馬体重_数値': horse.bataijuu,
            'distance': request.distance, 'course_type_enc': course_enc, 'weather_enc': weather_enc,
            'track_condition_enc': cond_enc, '舞台_平均前半ペース': avg_pace_f, '舞台_平均後半ペース': avg_pace_l,
            '舞台_平均要求上がり': avg_req_up, '過去基本脚質': horse.kyakushitsu, '前走着順': zenso,
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
    
    # Stage 1 偏差値（素の地力指数）と順位の算出
    s1_probs = [h["stage1_prob"] for h in horses_data]
    s1_mean = np.mean(s1_probs) if s1_probs else 0
    s1_std = np.std(s1_probs) if s1_probs else 1
    for h in horses_data:
        s1_val = ((h["stage1_prob"] - s1_mean) / s1_std) * 10 + 50 if s1_std > 0 else 50.0
        h["stage1_shisu"] = round(float(s1_val), 1)

    by_s1 = sorted(horses_data, key=lambda x: x["stage1_shisu"], reverse=True)
    for idx, h in enumerate(by_s1):
        h["stage1_rank"] = idx + 1

    # Stage 2 偏差値（最終指数）と順位の算出
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

    # 2段階指数に基づく4分類 ＆ 全5種・7パターンの馬券構築
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
            '舞台_平均要求上がり': avg_req_up, '過去基本脚質': horse.kyakushitsu, '前走着順': zenso,
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
