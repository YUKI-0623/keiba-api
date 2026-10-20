from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

app = FastAPI(title="競馬シミュレーションAPI (プロ仕様完全版: クラス補正/持ち時計壁/定性ファクター統合)")

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

        cf = float(r.get('class_factor', 0.70))  
        margin = float(r.get('margin', 1.0))     
        
        course_str = str(r.get('course', ''))
        dist = 2000.0
        m_dist = re.search(r'(\d{3,4})', course_str)
        if m_dist:
            dist = float(m_dist.group(1))
            
        adjusted_margin = margin * (2000.0 / dist) if dist > 0 else margin
        run_score = max(0.10, cf - (adjusted_margin * 0.15))
        class_based_scores.append(run_score)

    if class_based_scores:
        weights = np.arange(1, len(class_based_scores) + 1)
        power_score = np.average(class_based_scores, weights=weights)
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
        time.sleep(random.uniform(2.0, 4.0)) 
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
            track_type = match_track.group(1)
            distance = int(match_track.group(2))

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

        for block in horse_blocks:
            try:
                umaban = 0
                u_m = re.search(r'<td[^>]*class="[^"]*(?:Umaban)[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                if u_m:
                    u_str = re.sub(r'<[^>]*>', '', u_m.group(1)).strip()
                    if u_str.isdigit(): umaban = int(u_str)
                if umaban == 0: continue
                
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

                odds = 10.0
                o_m = re.search(r'<span[^>]*id="odds[^>]*>(.*?)</span>', block, re.IGNORECASE | re.DOTALL)
                if o_m:
                    o_str = re.sub(r'<[^>]*>', '', o_m.group(1)).strip()
                    try: odds = float(o_str)
                    except: pass

                weight_val = 480.0
                change_val = 0.0
                w_m = re.search(r'<td[^>]*class="[^"]*Weight[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                if w_m:
                    w_str = re.sub(r'<[^>]*>', '', w_m.group(1)).strip()
                    m_weight = re.search(r'(\d{3})\s*\(([+-]?\d+)\)', w_str)
                    if m_weight:
                        weight_val = float(m_weight.group(1))
                        change_val = float(m_weight.group(2))
                    else:
                        m_weight2 = re.search(r'(\d{3})', w_str)
                        if m_weight2: weight_val = float(m_weight2.group(1))

                h_pci, h_pci_type = get_horse_pci_stats(bamei)
                zenso, agari, kaisuu, _, _ = get_horse_real_stats(bamei)

                horses_data.append({
                    "wakuban": min(8, math.ceil(umaban / 2)),
                    "umaban": umaban, "bamei": bamei, "kishu": kishu,
                    "odds": odds, "kinryou": 55.0, 
                    "bataijuu": weight_val, "weight_change": change_val, 
                    "condition_override": "1.00",
                    "zenso_chakujun": zenso, "avg_agari": agari,
                    "kyakushitsu": get_auto_kyakushitsu(bamei),
                    "shussou_kaisuu": kaisuu,
                    "pci": h_pci, "pci_type": h_pci_type
                })
            except: continue

        horses_data = sorted(horses_data, key=lambda x: x['umaban'])
        course_key = f"{place_code}_{track_type}_{distance}"
        rpci, rpci_type, rec_pace = get_course_rpci_stats(course_key)

        return {
            "status": "success", "race_name": race_name, "place_name": place_name,
            "course_type": track_type, "distance": distance,
            "weather": weather, "track_condition": track_condition,
            "rpci": rpci, "rpci_type": rpci_type, "recommended_pace": rec_pace,
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

    best_time = None
    
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
    else:
        adjacent_times = []
        for r in h_records:
            r_c = str(r.get('course', ''))
            r_cond = str(r.get('cond', ''))
            t_sec = parse_time_to_seconds(r.get('time'))
            if t_sec is None: continue
            
            if course_type in r_c:
                m = re.search(r'(\d{3,4})', r_c)
                if m:
                    dist = int(m.group(1))
                    if abs(dist - current_distance) <= 400 and r_cond in ['良', '稍']:
                        avg_speed = t_sec / dist
                        projected_time = avg_speed * current_distance
                        adjacent_times.append(projected_time)
        
        if adjacent_times:
            best_time = min(adjacent_times)
            
    if best_time is None:
        return 1.00 

    diff_sec = best_time - expected_standard_time
    halon_count = current_distance / 200.0
    diff_per_halon = diff_sec / halon_count

    mult = 1.00
    is_high_speed = (cushion_val >= 9.5)
    
    if diff_per_halon <= 0.05:
        mult = 1.00 
    elif diff_per_halon <= 0.10:
        mult = 0.95 if is_high_speed else 0.98 
    elif diff_per_halon <= 0.15:
        mult = 0.88 if is_high_speed else 0.94 
    else:
        mult = 0.75 if is_high_speed else 0.88 

    return mult


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

    if track_bias == "前残り (逃げ・先行有利)":
        if kyaku <= 2: bias_mult = boost
        elif kyaku >= 3: bias_mult = nerf
    elif track_bias == "差し有利 (外伸び)":
        if kyaku >= 3: bias_mult = boost
        elif kyaku <= 2: bias_mult = nerf
    elif track_bias == "内枠有利 (インベタ)":
        if waku <= 3: bias_mult = boost
        elif waku >= 6: bias_mult = nerf
    elif track_bias == "外枠有利":
        if waku >= 6: bias_mult = boost
        elif waku <= 3: bias_mult = nerf

    condition_mult = 1.00
    try:
        condition_mult = float(horse.condition_override)
    except:
        pass

    total_mult = pace_mult * bias_mult * time_limit_mult * condition_mult
    capped_mult = max(0.75, min(1.25, total_mult))

    return base_ability * capped_mult


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
    
    expected_standard_time = c_stats.get('standard_time', 120.0) 
    
    avg_req_up = float(np.mean(top3_up)) if top3_up else 35.0
    avg_pace_f = float(np.mean(pace_f)) if pace_f else 35.0
    avg_pace_l = float(np.mean(pace_l)) if pace_l else 35.0

    base_rpci = round((avg_pace_f / avg_pace_l) * 50.0, 1) if avg_pace_l > 0 else 50.0
    base_rpci_type = get_pci_label(base_rpci)
    
    avg_pace_f, avg_pace_l, avg_req_up = adjust_pace_times(
        request.expected_pace, avg_pace_f, avg_pace_l, avg_req_up
    )
    
    place_dict = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
    
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

        final_score = calculate_comprehensive_score(
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
            "umaban": int(horse.umaban), "bamei": horse.bamei, "odds": float(horse.odds),
            "base_prob": final_score, "time_diff": round(time_diff, 2),
            "pci": h_pci, "pci_type": h_pci_type,
            "bataijuu": horse.bataijuu, "weight_change": horse.weight_change,
            "history_details": history_details 
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
                m_shisu = ((h["raw_ev"] - m_mean) / m_std) * 10 + 50 if m_std > 0 else 50.0
                h["myoumi"] = round(m_shisu, 1)
            else:
                h["myoumi"] = 0.0
    else:
        for h in horses_data: h["myoumi"] = 0.0

    horses_data.sort(key=lambda x: x["shisu"], reverse=True)
    return {
        "status": "success", "diagnosis": horses_data, "req_up": round(avg_req_up, 2),
        "pace_f": round(avg_pace_f, 1), "pace_l": round(avg_pace_l, 1),
        "rpci": base_rpci, "rpci_type": base_rpci_type
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
    expected_standard_time = c_stats.get('standard_time', 120.0)
    
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

        final_score = calculate_comprehensive_score(
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