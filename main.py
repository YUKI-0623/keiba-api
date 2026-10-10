from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
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

app = FastAPI(title="競馬シミュレーションAPI (プロ仕様・トラックバイアス対応版)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
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

class HorseData(BaseModel):
    wakuban: float
    umaban: float
    bamei: str = ""
    kishu: str = "不明"
    odds: float = 0.0
    kinryou: float
    bataijuu: float
    zenso_chakujun: float = 9.0
    avg_agari: float = 35.0
    kyakushitsu: float = 2.0  
    shussou_kaisuu: float = 0.0

class RaceRequest(BaseModel):
    race_id: str
    horses: List[HorseData]
    course_type: str
    distance: int
    weather: str
    track_condition: str
    expected_pace: str = "M"
    track_bias: str = "フラット" # ▼ 追加：トラックバイアス

@app.get("/fetch_race/{race_id}")
def fetch_race(race_id: str):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    time.sleep(random.uniform(1.0, 2.0))
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            res.encoding = 'euc-jp'
            html = res.text
            
            title_match = re.search(r'<title>(.*?)｜', html)
            race_name = title_match.group(1).strip() if title_match else "レース名不明"
            
            place_dict = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
            place_code = race_id[4:6] if len(race_id) >= 6 else "05"
            place_name = place_dict.get(place_code, "地方/海外")
            
            tables = pd.read_html(StringIO(html))
            target_df = None
            
            for tb in tables:
                cols = [str(c).replace(' ', '').replace(' ', '').strip() for c in tb.columns]
                if '枠番' in cols and '馬番' in cols and '馬名' in cols:
                    tb.columns = cols
                    target_df = tb
                    break
                    
            if target_df is None:
                return {"status": "error", "message": "出馬表データが見つかりませんでした"}
            
            track_type, distance, weather, condition = "芝", 2000, "晴", "良"
            track_match = re.search(r'([芝ダ障])(.*?外.*?)?(\d+)m', html)
            if track_match:
                track_type = track_match.group(1) 
                distance = int(track_match.group(3)) 
                
            weather_match = re.search(r'天候\s*:\s*([^\s/]+)', html)
            if weather_match: weather = weather_match.group(1)
                
            condition_match = re.search(r'(?:芝|ダート|馬場)\s*:\s*([^\s/]+)', html)
            if condition_match: condition = condition_match.group(1)

            horses_data = []
            for _, row in target_df.iterrows():
                try:
                    bamei = str(row['馬名'])
                    bataijuu_str = str(row['馬体重'])
                    b_match = re.search(r'(\d+)', bataijuu_str)
                    bataijuu = float(b_match.group(1)) if b_match else 480.0
                    
                    kishu = str(row['騎手']).strip() if '騎手' in row else "不明"
                    kishu = re.sub(r'^[▲△☆★◇]', '', kishu)
                    
                    odds = 0.0
                    if '単勝' in row:
                        try:
                            odds = float(str(row['単勝']).strip())
                        except:
                            odds = 0.0
                    
                    stats = horse_history_db.get(bamei, {})
                    ranks = stats.get('ranks', [])
                    up_times = stats.get('up_times', [])
                    kyaku = stats.get('kyakushitsu', [])
                    
                    zenso = ranks[-1] if ranks else 9.0
                    avg_agari = float(np.mean(up_times[-3:])) if up_times else 35.0
                    base_k = round(float(np.mean(kyaku[-2:]))) if kyaku else 2.0
                    
                    horses_data.append({
                        "wakuban": float(row['枠番']),
                        "umaban": float(row['馬番']),
                        "bamei": bamei,
                        "kishu": kishu,
                        "odds": odds,
                        "kinryou": float(row['斤量']),
                        "bataijuu": bataijuu,
                        "zenso_chakujun": zenso,
                        "avg_agari": round(avg_agari, 1),
                        "kyakushitsu": base_k,
                        "shussou_kaisuu": len(ranks)
                    })
                except Exception:
                    continue
                    
            horses_data = sorted(horses_data, key=lambda x: x["umaban"])
            
            return {
                "status": "success", 
                "race_name": race_name,
                "place_name": place_name,
                "horses": horses_data,
                "course_type": track_type,
                "distance": distance,
                "weather": weather,
                "track_condition": condition
            }
            
        except Exception as e:
            if attempt < max_retries - 1: time.sleep(2.0)
            else: return {"status": "error", "message": str(e)}

def apply_track_bias(prediction, horse, track_bias):
    """ トラックバイアスに基づいてAIスコアにボーナス/ペナルティを加算 """
    bias_bonus = 0.0
    kyaku = horse.kyakushitsu
    waku = horse.wakuban
    
    if track_bias == "前残り (逃げ・先行有利)":
        if kyaku == 1: bias_bonus += 0.06
        elif kyaku == 2: bias_bonus += 0.03
        elif kyaku == 3: bias_bonus -= 0.02
        elif kyaku == 4: bias_bonus -= 0.05
    elif track_bias == "差し有利 (外伸び)":
        if kyaku == 1: bias_bonus -= 0.05
        elif kyaku == 2: bias_bonus -= 0.02
        elif kyaku == 3: bias_bonus += 0.04
        elif kyaku == 4: bias_bonus += 0.06
    elif track_bias == "内枠有利 (インベタ)":
        if waku <= 3: bias_bonus += 0.05
        elif waku >= 6: bias_bonus -= 0.03
    elif track_bias == "外枠有利":
        if waku >= 6: bias_bonus += 0.05
        elif waku <= 3: bias_bonus -= 0.03
        
    return prediction + bias_bonus

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
    
    avg_req_up = float(np.mean(top3_up)) if top3_up else 35.0
    avg_pace_f = float(np.mean(pace_f)) if pace_f else 35.0
    avg_pace_l = float(np.mean(pace_l)) if pace_l else 35.0
    
    if request.expected_pace == "H":
        avg_pace_f -= 1.2; avg_pace_l += 1.2; avg_req_up += 0.8
    elif request.expected_pace == "S":
        avg_pace_f += 1.2; avg_pace_l -= 1.2; avg_req_up -= 0.8
    
    place_dict = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
    
    for horse in request.horses:
        h_stats = horse_history_db.get(horse.bamei, {})
        course_up_times = h_stats.get('course_up_times', {}).get(course_key, [])
        course_up = float(np.mean(course_up_times)) if course_up_times else horse.avg_agari
        
        time_diff = course_up - avg_req_up
        kishu_enc = safe_encode('kishu', horse.kishu)
        
        input_data = pd.DataFrame([{
            '枠番': horse.wakuban, '馬番': horse.umaban, '斤量': horse.kinryou, '馬体重_数値': horse.bataijuu,
            'distance': request.distance, 'course_type_enc': course_enc, 'weather_enc': weather_enc,
            'track_condition_enc': cond_enc, '舞台_平均前半ペース': avg_pace_f, '舞台_平均後半ペース': avg_pace_l,
            '舞台_平均要求上がり': avg_req_up, '過去基本脚質': horse.kyakushitsu, '前走着順': horse.zenso_chakujun,
            '近3走_平均上がり': horse.avg_agari, '同舞台_平均上がり': course_up, '出走回数': horse.shussou_kaisuu,
            '上がりタイム差分': time_diff, 'kishu_enc': kishu_enc
        }])
        
        prediction = float(model.predict(input_data)[0])
        
        # ▼▼ トラックバイアスの適用 ▼▼
        prediction = apply_track_bias(prediction, horse, request.track_bias)
        
        records = h_stats.get('race_records', [])[-5:]
        history_details = []
        for r in reversed(records):
            p_code = str(r['race_id'])[4:6]
            place = place_dict.get(p_code, "地方")
            try:
                c_type = str(r['course']).split('_')[1]
                dist = str(r['course']).split('_')[2]
            except:
                c_type, dist = "", ""
            race_name = f"{place} {c_type}{dist}m ({r['cond']})"
            history_details.append({
                "race_name": race_name, "rank": f"{int(r['rank'])}着",
                "time": str(r['time']), "up": f"{r['up']}秒" if r['up'] != '-' else "-",
                "passing": r.get('passing', '-')
            })

        horses_data.append({
            "umaban": int(horse.umaban), "bamei": horse.bamei, "odds": float(horse.odds),
            "base_prob": prediction, "time_diff": round(time_diff, 2),
            "history_details": history_details 
        })
    
    probs = [h["base_prob"] for h in horses_data]
    mean_prob = np.mean(probs) if probs else 0
    std_prob = np.std(probs) if probs else 1
    for h in horses_data:
        shisu = ((h["base_prob"] - mean_prob) / std_prob) * 10 + 50 if std_prob > 0 else 50.0
        h["shisu"] = round(shisu, 1)

    myoumi_values = [h["base_prob"] * h["odds"] for h in horses_data if h["odds"] > 0]
    if myoumi_values:
        m_mean = np.mean(myoumi_values); m_std = np.std(myoumi_values)
        for h in horses_data:
            if h["odds"] > 0:
                expected_return = h["base_prob"] * h["odds"]
                m_shisu = ((expected_return - m_mean) / m_std) * 10 + 50 if m_std > 0 else 50.0
                h["myoumi"] = round(m_shisu, 1)
            else: h["myoumi"] = 0.0
    else:
        for h in horses_data: h["myoumi"] = 0.0

    horses_data.sort(key=lambda x: x["shisu"], reverse=True)
    return {
        "status": "success", "diagnosis": horses_data, "req_up": round(avg_req_up, 2),
        "pace_f": round(avg_pace_f, 1), "pace_l": round(avg_pace_l, 1)
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
    
    avg_req_up = float(np.mean(top3_up)) if top3_up else 35.0
    avg_pace_f = float(np.mean(pace_f)) if pace_f else 35.0
    avg_pace_l = float(np.mean(pace_l)) if pace_l else 35.0
    
    if request.expected_pace == "H":
        avg_pace_f -= 1.2; avg_pace_l += 1.2; avg_req_up += 0.8
    elif request.expected_pace == "S":
        avg_pace_f += 1.2; avg_pace_l -= 1.2; avg_req_up -= 0.8
    
    for horse in request.horses:
        h_stats = horse_history_db.get(horse.bamei, {})
        course_up_times = h_stats.get('course_up_times', {}).get(course_key, [])
        course_up = float(np.mean(course_up_times)) if course_up_times else horse.avg_agari
        time_diff = course_up - avg_req_up
        kishu_enc = safe_encode('kishu', horse.kishu)
        
        input_data = pd.DataFrame([{
            '枠番': horse.wakuban, '馬番': horse.umaban, '斤量': horse.kinryou, '馬体重_数値': horse.bataijuu,
            'distance': request.distance, 'course_type_enc': course_enc, 'weather_enc': weather_enc,
            'track_condition_enc': cond_enc, '舞台_平均前半ペース': avg_pace_f, '舞台_平均後半ペース': avg_pace_l,
            '舞台_平均要求上がり': avg_req_up, '過去基本脚質': horse.kyakushitsu, '前走着順': horse.zenso_chakujun,
            '近3走_平均上がり': horse.avg_agari, '同舞台_平均上がり': course_up, '出走回数': horse.shussou_kaisuu,
            '上がりタイム差分': time_diff, 'kishu_enc': kishu_enc
        }])
        
        prediction = float(model.predict(input_data)[0])
        
        # ▼▼ トラックバイアスの適用 ▼▼
        prediction = apply_track_bias(prediction, horse, request.track_bias)
        
        horses_data.append({
            "umaban": int(horse.umaban), "bamei": horse.bamei,
            "base_prob": prediction, "time_diff": round(time_diff, 2)
        })
    
    probs = [h["base_prob"] for h in horses_data]
    mean_prob = np.mean(probs) if probs else 0
    std_prob = np.std(probs) if probs else 1
    
    sim_scores = []
    for h in horses_data:
        shisu = ((h["base_prob"] - mean_prob) / std_prob) * 10 + 50 if std_prob > 0 else 50.0
        noise = np.random.normal(0, 0.15) 
        score = h["base_prob"] + noise
        sim_scores.append({
            "umaban": h["umaban"], "bamei": h["bamei"],
            "shisu": round(shisu, 1), "time_diff": h["time_diff"], "score": score
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