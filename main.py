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
    # Bot弾きを回避するため、よりブラウザに近いヘッダーを設定
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3'
    }
    
    html = None
    is_shutuba = False
    
    # =========================================================
    # 💡 確実なデータ取得のための「Wait ＆ リトライ」専用関数
    # =========================================================
    def fetch_with_retry(url, max_retries=3):
        for attempt in range(max_retries):
            # 毎回確実に2〜4秒のWait（待機）を入れてサーバー負荷とBot判定を回避
            time.sleep(random.uniform(2.0, 4.0)) 
            try:
                res = requests.get(url, headers=headers, timeout=15)
                res.raise_for_status()
                res.encoding = 'EUC-JP'
                return res.text
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error fetching {url}: {e}")
                    return None
        return None

    # 【ロジック1】まずは未来の出馬表ページ (shutuba.html) へアクセス
    shutuba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    shutuba_text = fetch_with_retry(shutuba_url)
    
    if shutuba_text and ("HorseList" in shutuba_text or "Umaban" in shutuba_text or "馬番" in shutuba_text):
        html = shutuba_text
        is_shutuba = True

    # 【ロジック2】出馬表が無い（過去レース）場合は、データベース（db.netkeiba）へ自動切替
    if not is_shutuba:
        db_url = f"https://db.netkeiba.com/race/{race_id}/"
        db_text = fetch_with_retry(db_url)
        if db_text:
            html = db_text
        else:
            return {"status": "error", "message": "データ通信エラー: 時間をおいて再度お試しください。"}

    if not html:
        return {"status": "error", "message": "出馬表データが見つかりませんでした"}

    try:
        # レース名などの共通情報取得
        title_match = re.search(r'<title>(.*?)</title>', html)
        race_name = title_match.group(1).strip() if title_match else "レース不明"
        if "|" in race_name:
            race_name = race_name.split("|")[0].strip()

        place_dict = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
        place_code = race_id[4:6] if len(race_id) >= 6 else '05'
        place_name = place_dict.get(place_code, "地方/海外")

        track_type, distance, weather, track_condition = "芝", 2000, "晴", "良"
        match_track = re.search(r'(芝|ダ|障害)(?:専門)?(\d{3,4})m', html)
        if match_track:
            track_type = match_track.group(1)
            distance = int(match_track.group(2))

        horses_data = []

        # === 【抽出パターンA】未来の出馬表専用（正規表現のみで抽出） ===
        if is_shutuba:
            horse_blocks = re.findall(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
            # 構造が違う場合への備え
            if not horse_blocks:
                horse_blocks = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
                
            for block in horse_blocks:
                try:
                    # 馬番
                    umaban = 0
                    u_m = re.search(r'<td[^>]*class="[^"]*(?:Umaban|umaban)[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                    if u_m:
                        u_str = re.sub(r'<[^>]*>', '', u_m.group(1)).strip()
                        if u_str.isdigit(): umaban = int(u_str)
                    if umaban == 0: continue
                    
                    # 馬名
                    bamei = f"馬番{umaban}"
                    b_m = re.search(r'<span[^>]*class="[^"]*HorseName[^"]*"[^>]*>(.*?)</span>', block, re.IGNORECASE | re.DOTALL)
                    if not b_m:
                        b_m = re.search(r'<a[^>]*href="[^"]*horse/\d+[^"]*"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
                    if b_m:
                        b_str = re.sub(r'<[^>]*>', '', b_m.group(1)).strip()
                        if b_str: bamei = b_str

                    # 騎手
                    kishu = "不明"
                    k_m = re.search(r'<td[^>]*class="[^"]*Jockey[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                    if not k_m:
                        k_m = re.search(r'<a[^>]*href="[^"]*jockey/\d+[^"]*"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
                    if k_m:
                        k_str = re.sub(r'<[^>]*>', '', k_m.group(1)).strip()
                        k_str = re.sub(r'[\r\n\t\s]+', '', k_str)
                        if k_str: kishu = k_str

                    # オッズ
                    odds = 10.0
                    o_m = re.search(r'<span[^>]*id="odds[^>]*>(.*?)</span>', block, re.IGNORECASE | re.DOTALL)
                    if not o_m:
                        o_m = re.search(r'<td[^>]*class="[^"]*Txt_R[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                    if o_m:
                        o_str = re.sub(r'<[^>]*>', '', o_m.group(1)).strip()
                        try: odds = float(o_str)
                        except: pass

                    horses_data.append({
                        "wakuban": 1, "umaban": umaban, "bamei": bamei, "kishu": kishu,
                        "odds": odds, "kinryou": 55.0, "bataijuu": 480.0,
                        "zenso_chakujun": 5, "avg_agari": 35.0, "kyakushitsu": 2, "shussou_kaisuu": 10
                    })
                except Exception:
                    continue

        # === 【抽出パターンB】過去のDBページ、または解析漏れ時のPandasフォールバック ===
        if not horses_data:
            tables = pd.read_html(StringIO(html))
            for tb in tables:
                cols = [str(c).replace('\n', '').replace(' ', '').strip() for c in tb.columns]
                if any('馬番' in c for c in cols) and any('馬名' in c for c in cols):
                    for _, row in tb.iterrows():
                        try:
                            umaban = 0
                            for val in row.values:
                                try:
                                    v_int = int(str(val).strip())
                                    if 1 <= v_int <= 18:
                                        umaban = v_int
                                        break
                                except ValueError:
                                    continue
                            if umaban == 0: continue

                            bamei = ""
                            for val in row.values:
                                v_str = str(val).strip()
                                if len(v_str) >= 2 and not v_str.isdigit() and 'netkeiba' not in v_str:
                                    if not any(w in v_str for w in ['斤量', '馬体重', '調教師', '牡', '牝', '騸']):
                                        bamei = v_str
                                        break
                            if not bamei: bamei = f"馬番{umaban}"

                            kishu = "不明"
                            for val in row.values:
                                v_str = str(val).strip()
                                if len(v_str) >= 2 and v_str not in bamei and not v_str.isdigit():
                                    if any(k in v_str for k in ['川田', 'ルメール', '武豊', '横山', '戸崎', '坂井', '松山', '岩田', '西村', '鮫島', '幸', '浜中', '田辺', '津村', '菅原', '丹内', '荻野', '永島', '古川', '斎藤', '角田', '小林', '泉谷', '秋山', '富田', '原', '大野', '江田', '柴田', '石川', '北村', '丸田', '吉田', 'モレイラ', 'レーン', 'マーカ', 'デムーロ']):
                                        kishu = v_str.replace('[地方]', '').strip()
                                        break

                            odds = 10.0
                            for val in row.values:
                                try:
                                    v_float = float(str(val).strip())
                                    if 1.1 <= v_float <= 999.9:
                                        odds = v_float
                                        break
                                except ValueError:
                                    continue

                            horses_data.append({
                                "wakuban": 1, "umaban": umaban, "bamei": bamei, "kishu": kishu,
                                "odds": odds, "kinryou": 55.0, "bataijuu": 480.0,
                                "zenso_chakujun": 5, "avg_agari": 35.0, "kyakushitsu": 2, "shussou_kaisuu": 10
                            })
                        except Exception:
                            continue
                    break

        if not horses_data:
            return {"status": "error", "message": "出馬表の馬データを正しく解析できませんでした"}

        # 馬番順に綺麗に並び替え
        horses_data = sorted(horses_data, key=lambda x: x['umaban'])

        return {
            "status": "success",
            "race_name": race_name,
            "place_name": place_name,
            "course_type": track_type,
            "distance": distance,
            "weather": weather,
            "track_condition": track_condition,
            "horses": horses_data
        }

    except Exception as e:
        return {"status": "error", "message": f"パース処理エラー: {str(e)}"}
        
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
