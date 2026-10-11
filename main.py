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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    time.sleep(random.uniform(1.0, 2.0))
    
    html = None
    is_shutuba = False
    
    # ① まずは未来の出馬表ページ (shutuba.html) を取得しに行く
    shutuba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        res = requests.get(shutuba_url, headers=headers, timeout=10)
        res.raise_for_status()
        res.encoding = 'EUC-JP'
        if "HorseList" in res.text or "umaban" in res.text or "馬番" in res.text:
            html = res.text
            is_shutuba = True
    except Exception:
        pass

    # ② 失敗した場合は過去のデータベースへ
    if not html:
        db_url = f"https://db.netkeiba.com/race/{race_id}/"
        try:
            res = requests.get(db_url, headers=headers, timeout=10)
            res.raise_for_status()
            res.encoding = 'euc-jp'
            html = res.text
            is_shutuba = False
        except Exception as e:
            return {"status": "error", "message": f"データ取得に失敗しました: {str(e)}"}

    if not html:
        return {"status": "error", "message": "出馬表データが見つかりませんでした"}

    try:
        title_match = re.search(r'<title>(.*?)</title>', html)
        race_name = title_match.group(1).strip() if title_match else "レース不明"
        if "|" in race_name:
            race_name = race_name.split("|")[0].strip()

        place_dict = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
        place_code = race_id[4:6] if len(race_id) >= 6 else '05'
        place_name = place_dict.get(place_code, "地方/海外")

        track_type = "芝"
        distance = 2000
        weather = "晴"
        track_condition = "良"

        match_track = re.search(r'(芝|ダ|障害)(?:専門)?(\d{3,4})m', html)
        if match_track:
            track_type = match_track.group(1)
            distance = int(match_track.group(2))

        horses_data = []

        if is_shutuba:
            # === 未来の出馬表ページ専用の安全なテキスト/正規表現抽出 ===
            # HTMLタグを大まかに分解して馬番、馬名、騎手を安全に拾う
            # ネットケイバの出馬表行を探索
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # 各馬の行ブロックを探す
            rows = soup.find_all('tr', class_=re.compile('HorseList'))
            if not rows:
                rows = soup.select('table.Shutuba_Table tr, .race_table_01 tr, tr')

            for row in rows:
                row_text = row.get_text(separator=" ")
                # 数字（馬番）が含まれているかチェック
                match_umaban = re.search(r'\b([1-9]|1[0-8])\b', row.get_text())
                
                # 馬名らしき文字列や騎手名を探すための簡易チェック
                if "馬番" in row_text or "枠番" in row_text or len(row.find_all('td')) < 4:
                    continue
                
                tds = [td.get_text(strip=True) for td in row.find_all('td')]
                if len(tds) >= 5:
                    # 馬番を探す
                    umaban = 0
                    for t in tds[:3]:
                        if t.isdigit() and 1 <= int(t) <= 18:
                            umaban = int(t)
                            break
                    if umaban == 0:
                        continue

                    # 馬名を探す（2文字以上で数字以外のもの）
                    bamei = ""
                    for t in tds:
                        if len(t) >= 2 and not t.isdigit() and 'netkeiba' not in t:
                            if not any(w in t for w in ['斤量', '馬体重', '調教師', '牡', '牝', '騸', '父', '母', '歳', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                                bamei = t
                                break
                    if not bamei:
                        bamei = f"馬番{umaban}"

                    # 騎手を探す
                    kishu = "不明"
                    for t in tds:
                        if any(k in t for k in ['川田', 'ルメール', '武豊', '横山', '戸崎', '坂井', '松山', '岩田', '西村', '鮫島', '幸', '浜中', '田辺', '津村', '菅原', '丹内', '荻野', '永島', '古川', '斎藤', '角田', '小林', '泉谷', '秋山', '富田', '原', '大野', '江田', '柴田', '石川', '北村', '丸田', '吉田', 'モレイラ', 'レーン', 'マーカ', 'デムーロ', '騎手']):
                            kishu = t.replace('[地方]', '').replace('騎手', '').strip()
                            break

                    # 既に追加済みの馬番でなければ追加
                    if not any(h['umaban'] == umaban for h in horses_data):
                        horses_data.append({
                            "wakuban": 1,
                            "umaban": umaban,
                            "bamei": bamei,
                            "kishu": kishu,
                            "odds": 10.0,
                            "kinryou": 55.0,
                            "bataijuu": 480.0,
                            "zenso_chakujun": 5,
                            "avg_agari": 35.0,
                            "kyakushitsu": 2,
                            "shussou_kaisuu": 10
                        })

        # もし出馬表からの抽出で0頭だった場合、または過去結果ページだった場合のフォールバック
        if not horses_data:
            tables = pd.read_html(StringIO(html))
            target_df = None
            for tb in tables:
                cols = [str(c).replace('\n', '').replace(' ', '').strip() for c in tb.columns]
                if any('馬番' in c for c in cols) or any('馬名' in c for c in cols):
                    target_df = tb
                    break
            
            if target_df is not None and not target_df.empty:
                for _, row in target_df.iterrows():
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
                        if umaban == 0:
                            continue

                        bamei = ""
                        for val in row.values:
                            v_str = str(val).strip()
                            if len(v_str) >= 2 and not v_str.isdigit() and 'netkeiba' not in v_str:
                                if not any(w in v_str for w in ['斤量', '馬体重', '調教師', '牡', '牝', '騸']):
                                    bamei = v_str
                                    break
                        if not bamei:
                            bamei = f"馬番{umaban}"

                        kishu = "不明"
                        for val in row.values:
                            v_str = str(val).strip()
                            if len(v_str) >= 2 and v_str not in bamei and not v_str.isdigit():
                                if any(k in v_str for k in ['川田', 'ルメール', '武豊', '横山', '戸崎', '坂井', '松山', '岩田', '西村', '鮫島', '幸', '浜中', '田辺', '津村', '菅原', '丹内', '荻野', '永島', '古川', '斎藤', '角田', '小林', '泉谷', '秋山', '富田', '原', '大野', '江田', '柴田', '石川', '北村', '丸田', '吉田', 'モレイラ', 'レーン', 'マーカ', 'デムーロ']):
                                    kishu = v_str.replace('[地方]', '').strip()
                                    break

                        horses_data.append({
                            "wakuban": 1,
                            "umaban": umaban,
                            "bamei": bamei,
                            "kishu": kishu,
                            "odds": 10.0,
                            "kinryou": 55.0,
                            "bataijuu": 480.0,
                            "zenso_chakujun": 5,
                            "avg_agari": 35.0,
                            "kyakushitsu": 2,
                            "shussou_kaisuu": 10
                        })
                    except Exception:
                        continue

        if not horses_data:
            return {"status": "error", "message": "出馬表の馬データを正しく解析できませんでした"}

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