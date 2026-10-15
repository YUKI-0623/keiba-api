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

app = FastAPI(title="競馬シミュレーションAPI (プロ仕様・PCI/RPCI＆トラックバイアス対応版)")

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

# =========================================================
# 💡 PCI・RPCI 計算用ヘルパー関数群
# =========================================================
def parse_time_to_seconds(time_val):
    """ '1:58.1' のような走破タイム文字列を秒数(118.1)に変換 """
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
    """ 1レース分の「走破タイム」「上がり3F」「距離」からPCIを算出 """
    try:
        m = re.search(r'(\d{3,4})', str(course_str))
        if not m:
            return None
        dist = int(m.group(1))
        if dist <= 600:
            return None

        sec = parse_time_to_seconds(time_val)
        up = float(up_val)
        if sec is None or up <= 0 or sec <= up:
            return None

        # 前半3F換算タイム = (time - up) ÷ (距離 - 600) × 600
        first_3f_equiv = ((sec - up) / (dist - 600.0)) * 600.0
        # PCI = (前半3F換算 ÷ 上がり3F) × 50
        pci = (first_3f_equiv / up) * 50.0
        if 30.0 <= pci <= 75.0:
            return round(pci, 1)
        return None
    except:
        return None

def get_pci_label(pci_val):
    """ 基準値に基づいて A(ハイ) / B(ミドル) / C(スロー) を判定 """
    if pci_val < 48.0:
        return "A"  # ハイペース・消耗戦型
    elif pci_val <= 52.0:
        return "B"  # ミドルペース・平均型
    else:
        return "C"  # スローペース・瞬発力型

def get_horse_pci_stats(bamei_str):
    """ 馬の過去戦績から「好走時(1〜3着)の平均PCI」と適性タイプ(A/B/C)を算出 """
    records = horse_history_db.get(bamei_str, {}).get('race_records', [])
    if not records:
        return 50.0, "B"

    good_pcis = []
    all_pcis = []

    for r in records:
        pci = calc_single_pci(r.get('course', ''), r.get('time', ''), r.get('up', ''))
        if pci is not None:
            all_pcis.append(pci)
            try:
                rank = int(r.get('rank', 99))
                if 1 <= rank <= 3:
                    good_pcis.append(pci)
            except:
                pass

    # 1〜3着の好走履歴があればそれを優先、なければ全走平均
    target_pcis = good_pcis if good_pcis else all_pcis
    if not target_pcis:
        return 50.0, "B"

    avg_pci = round(float(np.mean(target_pcis)), 1)
    return avg_pci, get_pci_label(avg_pci)

def get_course_rpci_stats(course_key):
    """ コースの過去平均ペースからRPCIとペース傾向(A/B/C)を算出 """
    c_stats = course_history_db.get(course_key, {})
    pace_f = c_stats.get('pace_first', [])
    pace_l = c_stats.get('pace_last', [])

    avg_pace_f = float(np.mean(pace_f)) if pace_f else 35.0
    avg_pace_l = float(np.mean(pace_l)) if pace_l else 35.0

    # RPCI = (前半3F ÷ 後半3F) × 50
    rpci = round((avg_pace_f / avg_pace_l) * 50.0, 1) if avg_pace_l > 0 else 50.0
    rpci_type = get_pci_label(rpci)

    # AならH(ハイ)、BならM(ミドル)、CならS(スロー)を推奨ペースにする
    if rpci_type == "A":
        rec_pace = "H"
    elif rpci_type == "C":
        rec_pace = "S"
    else:
        rec_pace = "M"

    return rpci, rpci_type, rec_pace

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
    pci: Optional[float] = 50.0
    pci_type: Optional[str] = "B"

class RaceRequest(BaseModel):
    race_id: str
    horses: List[HorseData]
    course_type: str
    distance: int
    weather: str
    track_condition: str
    expected_pace: str = "M"
    track_bias: str = "フラット"

# =========================================================
# ① 出馬表データ取得API（脚質＋PCI＋RPCI自動判定版）
# =========================================================
@app.get("/fetch_race/{race_id}")
def fetch_race(race_id: str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3'
    }
    
    html = None
    is_shutuba = False
    
    def fetch_with_retry(url, max_retries=3):
        for attempt in range(max_retries):
            time.sleep(random.uniform(2.0, 4.0)) 
            try:
                res = requests.get(url, headers=headers, timeout=15)
                res.raise_for_status()
                res.encoding = res.apparent_encoding
                return res.text
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error fetching {url}: {e}")
                    return None
        return None

    def get_auto_kyakushitsu(bamei_str):
        try:
            records = horse_history_db.get(bamei_str, {}).get('race_records', [])
            if not records:
                return 2.0
            last_corners = []
            for r in records[-5:]:
                passing = r.get('passing', '')
                if passing and passing != '-':
                    parts = str(passing).split('-')
                    try:
                        last_corners.append(int(parts[-1]))
                    except ValueError:
                        pass
            if not last_corners:
                return 2.0
            avg_pos = np.mean(last_corners)
            if avg_pos <= 2.5: return 1.0
            elif avg_pos <= 6.5: return 2.0
            elif avg_pos <= 10.5: return 3.0
            else: return 4.0
        except Exception:
            return 2.0

    shutuba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    shutuba_text = fetch_with_retry(shutuba_url)
    
    if shutuba_text and ("HorseList" in shutuba_text or "Umaban" in shutuba_text or "馬番" in shutuba_text):
        html = shutuba_text
        is_shutuba = True

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

        if is_shutuba:
            horse_blocks = re.findall(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
            if not horse_blocks:
                horse_blocks = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
                
            for block in horse_blocks:
                try:
                    umaban = 0
                    u_m = re.search(r'<td[^>]*class="[^"]*(?:Umaban|umaban)[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                    if u_m:
                        u_str = re.sub(r'<[^>]*>', '', u_m.group(1)).strip()
                        if u_str.isdigit(): umaban = int(u_str)
                    if umaban == 0: continue
                    
                    bamei = f"馬番{umaban}"
                    b_m = re.search(r'<span[^>]*class="[^"]*HorseName[^"]*"[^>]*>(.*?)</span>', block, re.IGNORECASE | re.DOTALL)
                    if not b_m:
                        b_m = re.search(r'<a[^>]*href="[^"]*horse/\d+[^"]*"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
                    if b_m:
                        b_str = re.sub(r'<[^>]*>', '', b_m.group(1)).strip()
                        if b_str: bamei = b_str

                    kishu = "不明"
                    k_m = re.search(r'<td[^>]*class="[^"]*Jockey[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                    if not k_m:
                        k_m = re.search(r'<a[^>]*href="[^"]*jockey/\d+[^"]*"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
                    if k_m:
                        k_str = re.sub(r'<[^>]*>', '', k_m.group(1)).strip()
                        k_str = re.sub(r'[\r\n\t\s]+', '', k_str)
                        if k_str: kishu = k_str

                    odds = 10.0
                    o_m = re.search(r'<span[^>]*id="odds[^>]*>(.*?)</span>', block, re.IGNORECASE | re.DOTALL)
                    if not o_m:
                        o_m = re.search(r'<td[^>]*class="[^"]*Txt_R[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                    if o_m:
                        o_str = re.sub(r'<[^>]*>', '', o_m.group(1)).strip()
                        try: odds = float(o_str)
                        except: pass

                    h_pci, h_pci_type = get_horse_pci_stats(bamei)

                    horses_data.append({
                        "wakuban": min(8, math.ceil(umaban / 2)),
                        "umaban": umaban, "bamei": bamei, "kishu": kishu,
                        "odds": odds, "kinryou": 55.0, "bataijuu": 480.0,
                        "zenso_chakujun": 5, "avg_agari": 35.0,
                        "kyakushitsu": get_auto_kyakushitsu(bamei),
                        "shussou_kaisuu": 10,
                        "pci": h_pci,
                        "pci_type": h_pci_type
                    })
                except Exception:
                    continue

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
                                    if any(k in v_str for k in ['川田', 'ルメール', '武豊', '横山', '戸崎', '坂井', '松山', '岩田', '西村', '鮫島', '幸', '浜中', '田辺', '津村', '菅原', '丹内', '荻野', '永島', '古川', '斎藤', '角田', '小林', '泉谷', '秋山', '富田', '原', '大野', '江田', '柴田', '石川', '北村', '丸田', '吉田', 'モレイラ', 'レーン', 'マーカ', 'デムーロ', '菊沢', '木幡', '吉村']):
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

                            h_pci, h_pci_type = get_horse_pci_stats(bamei)

                            horses_data.append({
                                "wakuban": min(8, math.ceil(umaban / 2)),
                                "umaban": umaban, "bamei": bamei, "kishu": kishu,
                                "odds": odds, "kinryou": 55.0, "bataijuu": 480.0,
                                "zenso_chakujun": 5, "avg_agari": 35.0,
                                "kyakushitsu": get_auto_kyakushitsu(bamei),
                                "shussou_kaisuu": 10,
                                "pci": h_pci,
                                "pci_type": h_pci_type
                            })
                        except Exception:
                            continue
                    break

        if not horses_data:
            return {"status": "error", "message": "出馬表の馬データを正しく解析できませんでした"}

        horses_data = sorted(horses_data, key=lambda x: x['umaban'])

        # コースの過去データからRPCIと推奨ペースを算出
        course_key = f"{place_code}_{track_type}_{distance}"
        rpci, rpci_type, rec_pace = get_course_rpci_stats(course_key)

        return {
            "status": "success",
            "race_name": race_name,
            "place_name": place_name,
            "course_type": track_type,
            "distance": distance,
            "weather": weather,
            "track_condition": track_condition,
            "rpci": rpci,
            "rpci_type": rpci_type,
            "recommended_pace": rec_pace,
            "horses": horses_data
        }

    except Exception as e:
        return {"status": "error", "message": f"パース処理エラー: {str(e)}"}


# =========================================================
# ② AI予測＆トラックバイアス＋PCI適性適用API
# =========================================================
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

def apply_pci_bonus(prediction, horse_pci, horse_pci_type, expected_pace):
    """ 想定ペース(H/M/S)と馬の好走PCI適性(A/B/C)の合致度でAIスコアを補正 """
    pci_bonus = 0.0

    # H(ハイペース＝タイプA：消耗戦想定)
    if expected_pace == "H":
        if horse_pci_type == "A":
            pci_bonus += 0.05
        elif horse_pci_type == "C":
            pci_bonus -= 0.04
    # S(スローペース＝タイプC：瞬発力戦想定)
    elif expected_pace == "S":
        if horse_pci_type == "C":
            pci_bonus += 0.05
        elif horse_pci_type == "A":
            pci_bonus -= 0.04
    # M(ミドルペース＝タイプB：平均ペース想定)
    else:
        if horse_pci_type == "B":
            pci_bonus += 0.03

    return prediction + pci_bonus

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

    # コース本来のRPCI（補正前）を計算
    base_rpci = round((avg_pace_f / avg_pace_l) * 50.0, 1) if avg_pace_l > 0 else 50.0
    base_rpci_type = get_pci_label(base_rpci)
    
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
        
        # 馬ごとの好走PCIと適性タイプを取得
        h_pci, h_pci_type = get_horse_pci_stats(horse.bamei)

        # ① トラックバイアスの適用
        prediction = apply_track_bias(prediction, horse, request.track_bias)
        # ② PCI適性ボーナスの適用
        prediction = apply_pci_bonus(prediction, h_pci, h_pci_type, request.expected_pace)
        
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
            single_pci = calc_single_pci(r.get('course', ''), r.get('time', ''), r.get('up', ''))
            history_details.append({
                "race_name": race_name, "rank": f"{int(r['rank'])}着",
                "time": str(r['time']), "up": f"{r['up']}秒" if r['up'] != '-' else "-",
                "passing": r.get('passing', '-'),
                "pci": f"{single_pci}" if single_pci is not None else "-"
            })

        horses_data.append({
            "umaban": int(horse.umaban), "bamei": horse.bamei, "odds": float(horse.odds),
            "base_prob": prediction, "time_diff": round(time_diff, 2),
            "pci": h_pci, "pci_type": h_pci_type,
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
        
        h_pci, h_pci_type = get_horse_pci_stats(horse.bamei)

        # ① トラックバイアスの適用
        prediction = apply_track_bias(prediction, horse, request.track_bias)
        # ② PCI適性ボーナスの適用
        prediction = apply_pci_bonus(prediction, h_pci, h_pci_type, request.expected_pace)
        
        horses_data.append({
            "umaban": int(horse.umaban), "bamei": horse.bamei,
            "base_prob": prediction, "time_diff": round(time_diff, 2),
            "pci": h_pci, "pci_type": h_pci_type
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
