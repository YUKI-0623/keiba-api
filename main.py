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

app = FastAPI(title="競馬シミュレーションAPI (実力・期待値収束＆PCI対応版)")

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
# 💡 PCI・RPCI ＆ 馬の実力値 計算用ヘルパー関数群
# =========================================================
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
        if not m:
            return None
        dist = int(m.group(1))
        if dist <= 600:
            return None

        sec = parse_time_to_seconds(time_val)
        up = float(up_val)
        if sec is None or up <= 0 or sec <= up:
            return None

        first_3f_equiv = ((sec - up) / (dist - 600.0)) * 600.0
        pci = (first_3f_equiv / up) * 50.0
        if 30.0 <= pci <= 75.0:
            return round(pci, 1)
        return None
    except:
        return None

def get_pci_label(pci_val):
    if pci_val < 48.0:
        return "A"  # ハイペース・消耗戦型
    elif pci_val <= 52.0:
        return "B"  # ミドルペース・平均型
    else:
        return "C"  # スローペース・瞬発力型

def get_horse_pci_stats(bamei_str):
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

    target_pcis = good_pcis if good_pcis else all_pcis
    if not target_pcis:
        return 50.0, "B"

    avg_pci = round(float(np.mean(target_pcis)), 1)
    return avg_pci, get_pci_label(avg_pci)

def get_horse_real_stats(bamei_str):
    """ 過去DBから馬の実際の『前走着順』『近3走平均上がり』『出走回数』『地力スコア』を抽出 """
    records = horse_history_db.get(bamei_str, {}).get('race_records', [])
    if not records:
        return 8.0, 35.0, 0, 0.5

    shussou_kaisuu = len(records)

    # 1. 前走着順の取得
    try:
        zenso_chakujun = float(records[-1].get('rank', 8.0))
    except:
        zenso_chakujun = 8.0

    # 2. 近3走の平均上がり3Fの取得
    up_list = []
    for r in records[-3:]:
        try:
            u = float(r.get('up', 0))
            if 30.0 <= u <= 45.0:
                up_list.append(u)
        except:
            pass
    avg_agari = round(float(np.mean(up_list)), 2) if up_list else 35.0

    # 3. 直近5走の着順から「地力（馬の基礎的な強さ）」を0.0〜1.0で評価
    recent_ranks = []
    for r in records[-5:]:
        try:
            rk = float(r.get('rank', 10.0))
            recent_ranks.append(min(18.0, max(1.0, rk)))
        except:
            pass
    if recent_ranks:
        weights = np.arange(1, len(recent_ranks) + 1)
        weighted_avg_rank = np.average(recent_ranks, weights=weights)
        power_score = max(0.1, (16.0 - weighted_avg_rank) / 15.0)
    else:
        power_score = 0.5

    return zenso_chakujun, avg_agari, shussou_kaisuu, power_score

def get_course_rpci_stats(course_key):
    c_stats = course_history_db.get(course_key, {})
    pace_f = c_stats.get('pace_first', [])
    pace_l = c_stats.get('pace_last', [])

    avg_pace_f = float(np.mean(pace_f)) if pace_f else 35.0
    avg_pace_l = float(np.mean(pace_l)) if pace_l else 35.0

    rpci = round((avg_pace_f / avg_pace_l) * 50.0, 1) if avg_pace_l > 0 else 50.0
    rpci_type = get_pci_label(rpci)

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
    expected_pace: str = "M"
    track_bias: str = "フラット"

# =========================================================
# ① 出馬表データ取得API（実力データ自動紐付け版）
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
                    zenso, agari, kaisuu, _ = get_horse_real_stats(bamei)

                    horses_data.append({
                        "wakuban": min(8, math.ceil(umaban / 2)),
                        "umaban": umaban, "bamei": bamei, "kishu": kishu,
                        "odds": odds, "kinryou": 55.0, "bataijuu": 480.0,
                        "zenso_chakujun": zenso, "avg_agari": agari,
                        "kyakushitsu": get_auto_kyakushitsu(bamei),
                        "shussou_kaisuu": kaisuu,
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
                            zenso, agari, kaisuu, _ = get_horse_real_stats(bamei)

                            horses_data.append({
                                "wakuban": min(8, math.ceil(umaban / 2)),
                                "umaban": umaban, "bamei": bamei, "kishu": kishu,
                                "odds": odds, "kinryou": 55.0, "bataijuu": 480.0,
                                "zenso_chakujun": zenso, "avg_agari": agari,
                                "kyakushitsu": get_auto_kyakushitsu(bamei),
                                "shussou_kaisuu": kaisuu,
                                "pci": h_pci,
                                "pci_type": h_pci_type
                            })
                        except Exception:
                            continue
                    break

        if not horses_data:
            return {"status": "error", "message": "出馬表の馬データを正しく解析できませんでした"}

        horses_data = sorted(horses_data, key=lambda x: x['umaban'])

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
# ② 基礎能力 ＆ 展開・PCI補正の総合計算エンジン
# =========================================================
def calculate_comprehensive_score(raw_lgb_pred, horse, power_score, h_pci_type, expected_pace, track_bias):
    """
    AIの基礎予測値、過去着順の地力、市場評価(オッズ)を統合し、
    馬の本来の強さをしっかりと反映させた上で展開・PCI補正を倍率で適用する
    """
    # 1. オッズから逆算される市場の基礎支持率（極端な能力乖離を防ぐアンカー）
    if horse.odds > 1.0:
        market_prob = min(0.55, 0.80 / float(horse.odds))
    else:
        market_prob = 0.08

    # 2. 馬の基礎能力スコアの統合 (LightGBM予測 + 過去着順の地力 + 市場評価のブレンド)
    base_ability = (raw_lgb_pred * 0.50) + (power_score * 0.30) + (market_prob * 0.20)

    # 3. トラックバイアス補正（基礎能力を土台とした倍率方式：最大±12%程度）
    bias_mult = 1.0
    kyaku = horse.kyakushitsu
    waku = horse.wakuban

    if track_bias == "前残り (逃げ・先行有利)":
        if kyaku == 1: bias_mult = 1.12
        elif kyaku == 2: bias_mult = 1.06
        elif kyaku == 3: bias_mult = 0.96
        elif kyaku == 4: bias_mult = 0.90
    elif track_bias == "差し有利 (外伸び)":
        if kyaku == 1: bias_mult = 0.90
        elif kyaku == 2: bias_mult = 0.96
        elif kyaku == 3: bias_mult = 1.08
        elif kyaku == 4: bias_mult = 1.12
    elif track_bias == "内枠有利 (インベタ)":
        if waku <= 3: bias_mult = 1.10
        elif waku >= 6: bias_mult = 0.94
    elif track_bias == "外枠有利":
        if waku >= 6: bias_mult = 1.10
        elif waku <= 3: bias_mult = 0.94

    # 4. PCIペース適性補正（最大±8%程度）
    pci_mult = 1.0
    if expected_pace == "H":
        if h_pci_type == "A": pci_mult = 1.08
        elif h_pci_type == "C": pci_mult = 0.93
    elif expected_pace == "S":
        if h_pci_type == "C": pci_mult = 1.08
        elif h_pci_type == "A": pci_mult = 0.93
    else:
        if h_pci_type == "B": pci_mult = 1.05

    return base_ability * bias_mult * pci_mult


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

    base_rpci = round((avg_pace_f / avg_pace_l) * 50.0, 1) if avg_pace_l > 0 else 50.0
    base_rpci_type = get_pci_label(base_rpci)
    
    if request.expected_pace == "H":
        avg_pace_f -= 1.2; avg_pace_l += 1.2; avg_req_up += 0.8
    elif request.expected_pace == "S":
        avg_pace_f += 1.2; avg_pace_l -= 1.2; avg_req_up -= 0.8
    
    place_dict = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
    
    for horse in request.horses:
        h_stats = horse_history_db.get(horse.bamei, {})
        zenso, real_agari, kaisuu, power_score = get_horse_real_stats(horse.bamei)

        course_up_times = h_stats.get('course_up_times', {}).get(course_key, [])
        course_up = float(np.mean(course_up_times)) if course_up_times else real_agari
        
        time_diff = course_up - avg_req_up
        kishu_enc = safe_encode('kishu', horse.kishu)
        
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

        # 地力・AI予測・展開・PCIを統合した最終スコア算出
        final_score = calculate_comprehensive_score(
            raw_pred, horse, power_score, h_pci_type, request.expected_pace, request.track_bias
        )
        
        # 現在のpage.tsxに100%対応した過去5走データ形式
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
                "rank": f"{int(r.get('rank', 99))}着",
                "time": str(r.get('time', '-')),
                "up": f"{r.get('up', '-')}秒" if r.get('up', '-') != '-' else "-",
                "passing": r.get('passing', '-'),
                "pci": f"{single_pci}" if single_pci is not None else "-"
            })

        horses_data.append({
            "umaban": int(horse.umaban), "bamei": horse.bamei, "odds": float(horse.odds),
            "base_prob": final_score, "time_diff": round(time_diff, 2),
            "pci": h_pci, "pci_type": h_pci_type,
            "history_details": history_details 
        })
    
    # AI基本指数（偏差値化）
    probs = [h["base_prob"] for h in horses_data]
    mean_prob = np.mean(probs) if probs else 0
    std_prob = np.std(probs) if probs else 1
    total_prob = sum(probs) if sum(probs) > 0 else 1.0

    for h in horses_data:
        shisu = ((h["base_prob"] - mean_prob) / std_prob) * 10 + 50 if std_prob > 0 else 50.0
        h["shisu"] = round(shisu, 1)
        h["norm_win_prob"] = h["base_prob"] / total_prob

    # 妙味指数（期待値）：超低能力馬の暴走を防ぐフィルター付き
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
        for h in horses_data:
            h["myoumi"] = 0.0

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
        zenso, real_agari, kaisuu, power_score = get_horse_real_stats(horse.bamei)

        course_up_times = h_stats.get('course_up_times', {}).get(course_key, [])
        course_up = float(np.mean(course_up_times)) if course_up_times else real_agari
        time_diff = course_up - avg_req_up
        kishu_enc = safe_encode('kishu', horse.kishu)
        
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
            raw_pred, horse, power_score, h_pci_type, request.expected_pace, request.track_bias
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
        for h in horses_data:
            h["myoumi"] = 50.0

    # 🏁 実力（AI基本指数 85%）と期待値（妙味指数 15%）に収束させた高精度シミュレーション
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