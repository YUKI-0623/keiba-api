from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import lightgbm as lgb
import pandas as pd
import numpy as np
import requests
import re
import time
import pickle
import math
import copy
from datetime import datetime

app = FastAPI(title="競馬シミュレーションAPI (WIN5対応・芝/ダート雨完全分離・高速完全版)")

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

runtime_enriched_db: Dict[str, List[Dict[str, Any]]] = {}
runtime_pedigree_db: Dict[str, Dict[str, str]] = {}
html_response_cache: Dict[str, str] = {}
http_session = requests.Session()

SIRE_APTITUDE_DB: Dict[str, Dict[str, float]] = {
    "キタサンブラック": {"stamina": 3.0, "heavy": 2.8, "lonspa": 2.2, "dirt_wet": 1.5},
    "エピファネイア": {"stamina": 3.0, "heavy": 2.6, "lonspa": 2.2, "dirt_wet": 1.2},
    "ワールドプレミア": {"stamina": 3.0, "heavy": 2.5, "lonspa": 2.0, "dirt_wet": 1.0},
    "ゴールドシップ": {"stamina": 3.0, "heavy": 3.2, "lonspa": 2.2, "dirt_wet": 1.0},
    "オルフェーヴル": {"stamina": 3.0, "heavy": 3.2, "lonspa": 2.2, "dirt_wet": 2.2},
    "ハーツクライ": {"stamina": 2.8, "heavy": 1.4, "lonspa": 2.0, "dirt_wet": 1.2},
    "ステイゴールド": {"stamina": 3.0, "heavy": 3.0, "lonspa": 2.0, "dirt_wet": 1.2},
    "フィエールマン": {"stamina": 3.0, "heavy": 1.8, "lonspa": 2.0, "dirt_wet": 1.0},
    "サトノダイヤモンド": {"stamina": 2.8, "heavy": 1.8, "lonspa": 1.8, "dirt_wet": 1.0},
    "シュヴァルグラン": {"stamina": 2.8, "heavy": 2.2, "lonspa": 1.8, "dirt_wet": 1.2},
    "マンハッタンカフェ": {"stamina": 2.8, "heavy": 2.2, "lonspa": 1.8, "dirt_wet": 1.5},
    "スクリーンヒーロー": {"stamina": 2.5, "heavy": 2.8, "lonspa": 1.8, "dirt_wet": 1.5},
    "ダンスインザダーク": {"stamina": 3.0, "heavy": 1.6, "lonspa": 2.0, "dirt_wet": 1.0},
    "トニービン": {"stamina": 2.8, "heavy": 2.4, "lonspa": 2.0, "dirt_wet": 1.0},
    "スワーヴリチャード": {"stamina": 2.4, "heavy": 2.6, "lonspa": 2.0, "dirt_wet": 1.8},
    "キズナ": {"stamina": 2.5, "heavy": 3.0, "lonspa": 2.0, "dirt_wet": 2.4},
    "ドゥラメンテ": {"stamina": 2.4, "heavy": 2.5, "lonspa": 1.9, "dirt_wet": 2.2},
    "コントレイル": {"stamina": 2.2, "heavy": 1.2, "lonspa": 1.8, "dirt_wet": 1.4},
    "ディープインパクト": {"stamina": 2.3, "heavy": 0.6, "lonspa": 2.0, "dirt_wet": 0.8},
    "ブラックタイド": {"stamina": 2.4, "heavy": 2.2, "lonspa": 1.8, "dirt_wet": 1.2},
    "レイデオロ": {"stamina": 2.4, "heavy": 2.2, "lonspa": 1.8, "dirt_wet": 1.6},
    "ルーラーシップ": {"stamina": 2.4, "heavy": 2.9, "lonspa": 1.9, "dirt_wet": 2.0},
    "ハービンジャー": {"stamina": 2.3, "heavy": 3.1, "lonspa": 1.9, "dirt_wet": 1.0},
    "キングカメハメハ": {"stamina": 2.0, "heavy": 2.2, "lonspa": 1.5, "dirt_wet": 2.2},
    "サートゥルナーリア": {"stamina": 1.8, "heavy": 1.8, "lonspa": 1.6, "dirt_wet": 1.5},
    "モーリス": {"stamina": 1.9, "heavy": 2.9, "lonspa": 1.8, "dirt_wet": 1.8},
    "ロベルト": {"stamina": 2.2, "heavy": 3.0, "lonspa": 1.8, "dirt_wet": 1.5},
    "サドラーズウェルズ": {"stamina": 3.0, "heavy": 3.2, "lonspa": 2.0, "dirt_wet": 1.0},
    "Galileo": {"stamina": 3.0, "heavy": 3.0, "lonspa": 2.0, "dirt_wet": 1.0},
    "Montjeu": {"stamina": 3.0, "heavy": 3.2, "lonspa": 2.0, "dirt_wet": 1.0},
    "バゴ": {"stamina": 2.4, "heavy": 3.2, "lonspa": 1.8, "dirt_wet": 1.2},
    "ジャスタウェイ": {"stamina": 1.8, "heavy": 1.8, "lonspa": 1.5, "dirt_wet": 1.4},
    "リアルスティール": {"stamina": 1.8, "heavy": 1.4, "lonspa": 1.5, "dirt_wet": 2.0},
    "シルバーステート": {"stamina": 1.7, "heavy": 2.5, "lonspa": 1.5, "dirt_wet": 1.4},
    "アルアイン": {"stamina": 1.8, "heavy": 2.3, "lonspa": 1.5, "dirt_wet": 1.4},
    "リオンディーズ": {"stamina": 2.0, "heavy": 2.2, "lonspa": 1.5, "dirt_wet": 1.8},
    "ブリックスアンドモルタル": {"stamina": 1.6, "heavy": 1.6, "lonspa": 1.4, "dirt_wet": 1.2},
    "エイシンフラッシュ": {"stamina": 2.0, "heavy": 2.0, "lonspa": 1.5, "dirt_wet": 1.2},
    "ノヴェリスト": {"stamina": 2.4, "heavy": 2.7, "lonspa": 1.6, "dirt_wet": 1.2},
    "シンボリクリスエス": {"stamina": 2.2, "heavy": 2.5, "lonspa": 1.8, "dirt_wet": 2.2},
    "ネオユニヴァース": {"stamina": 2.2, "heavy": 2.6, "lonspa": 1.6, "dirt_wet": 2.0},
    "Malibu Moon": {"stamina": 1.8, "heavy": 2.0, "lonspa": 1.5, "dirt_wet": 2.5},
    "ロードカナロア": {"stamina": 0.8, "heavy": 1.4, "lonspa": 1.0, "dirt_wet": 2.4},
    "ダイワメジャー": {"stamina": -0.5, "heavy": 1.9, "lonspa": 0.5, "dirt_wet": 1.8},
    "クロフネ": {"stamina": 0.0, "heavy": 2.2, "lonspa": 0.8, "dirt_wet": 2.8},
    "イスラボニータ": {"stamina": 0.5, "heavy": 1.5, "lonspa": 0.8, "dirt_wet": 1.4},
    "シスキン": {"stamina": 0.5, "heavy": 1.5, "lonspa": 0.8, "dirt_wet": 1.8},
    "ポエティックフレア": {"stamina": 0.5, "heavy": 1.8, "lonspa": 0.8, "dirt_wet": 1.2},
    "ナダル": {"stamina": 1.0, "heavy": 2.4, "lonspa": 1.0, "dirt_wet": 2.8},
    "ドレフォン": {"stamina": 0.5, "heavy": 2.2, "lonspa": 0.8, "dirt_wet": 3.0},
    "マインドユアビスケッツ": {"stamina": 0.5, "heavy": 2.2, "lonspa": 0.8, "dirt_wet": 2.8},
    "グレーターロンドン": {"stamina": -1.2, "heavy": 0.0, "lonspa": 0.5, "dirt_wet": 1.0},
    "ミッキーアイル": {"stamina": -1.5, "heavy": 1.0, "lonspa": 0.2, "dirt_wet": 2.0},
    "ビッグアーサー": {"stamina": -2.0, "heavy": 1.0, "lonspa": 0.0, "dirt_wet": 1.2},
    "ファインニードル": {"stamina": -2.0, "heavy": 1.0, "lonspa": 0.0, "dirt_wet": 1.2},
    "タワーオブロンドン": {"stamina": -1.8, "heavy": 1.0, "lonspa": 0.0, "dirt_wet": 1.2},
    "アドマイヤムーン": {"stamina": -1.5, "heavy": 1.2, "lonspa": 0.0, "dirt_wet": 1.0},
    "サクラバクシンオー": {"stamina": -2.0, "heavy": 1.0, "lonspa": 0.0, "dirt_wet": 1.2},
    "ヘニーヒューズ": {"stamina": -1.5, "heavy": 2.2, "lonspa": 0.2, "dirt_wet": 3.0},
    "シニスターミニスター": {"stamina": -0.5, "heavy": 2.5, "lonspa": 0.5, "dirt_wet": 3.0},
}

KNOWN_HORSE_PEDIGREE: Dict[str, Dict[str, str]] = {
    "ロブチェン": {"sire": "ワールドプレミア", "bms": "Giant's Causeway"},
    "ダノンヒストリー": {"sire": "エピファネイア", "bms": "Malibu Moon"},
    "バステール": {"sire": "キタサンブラック", "bms": "マンハッタンカフェ"},
    "ゴーイントゥスカイ": {"sire": "コントレイル", "bms": "Galileo"},
    "ジャスティンシカゴ": {"sire": "スワーヴリチャード", "bms": "ディープインパクト"},
    "アスクエジンバラ": {"sire": "ドゥラメンテ", "bms": "ハービンジャー"},
    "コンジェスタス": {"sire": "キズナ", "bms": "サドラーズウェルズ"},
    "アウダーシア": {"sire": "キズナ", "bms": "ルーラーシップ"},
    "エムズビギン": {"sire": "キタサンブラック", "bms": "ディープインパクト"},
    "サノノグレーター": {"sire": "グレーターロンドン", "bms": "キングカメハメハ"},
    "シートゥサミット": {"sire": "モーリス", "bms": "サドラーズウェルズ"},
    "マジョレルブルー": {"sire": "レイデオロ", "bms": "Galileo"},
    "ショウナンハヤナミ": {"sire": "エピファネイア", "bms": "ディープインパクト"},
    "ナリタエスペランサ": {"sire": "キタサンブラック", "bms": "キングカメハメハ"},
    "キンググローリー": {"sire": "ハービンジャー", "bms": "ディープインパクト"},
    "ラージアンサンブル": {"sire": "ルーラーシップ", "bms": "ディープインパクト"},
    "チャリングクロス": {"sire": "サートゥルナーリア", "bms": "キングカメハメハ"},
    "ショウナンガルフ": {"sire": "キズナ", "bms": "キングカメハメハ"},
    "レガレイラ": {"sire": "スワーヴリチャード", "bms": "ハービンジャー"},
}

TRAINING_RANK_MULTIPLIERS: Dict[str, float] = {
    "S": 1.05,
    "A": 1.03,
    "B": 1.00,
    "C": 0.92,
}

JOCKEY_STAGE_BONUSES: Dict[str, Dict[str, float]] = {
    "ルメール": {"high_speed": 1.022, "long_dist": 1.022, "outer_smooth": 1.015, "heavy_track": 1.018},
    "C.ルメール": {"high_speed": 1.022, "long_dist": 1.022, "outer_smooth": 1.015, "heavy_track": 1.018},
    "川田": {"good_pos": 1.020, "inner_pocket": 1.018, "heavy_track": 1.018},
    "川田将雅": {"good_pos": 1.020, "inner_pocket": 1.018, "heavy_track": 1.018},
    "戸崎圭": {"outer_smooth": 1.018, "high_speed": 1.015},
    "戸崎圭太": {"outer_smooth": 1.018, "high_speed": 1.015},
    "武豊": {"pace_make": 1.022, "long_dist": 1.020, "heavy_track": 1.018},
    "坂井": {"front_keep": 1.020, "inner_pocket": 1.015},
    "坂井瑠星": {"front_keep": 1.020, "inner_pocket": 1.015},
    "横山典": {"pace_make": 1.020, "inner_pocket": 1.018, "heavy_track": 1.022},
    "横山典弘": {"pace_make": 1.020, "inner_pocket": 1.018, "heavy_track": 1.022},
    "横山武": {"front_keep": 1.016, "makuri": 1.018, "heavy_track": 1.018},
    "横山武史": {"front_keep": 1.016, "makuri": 1.018, "heavy_track": 1.018},
    "松山": {"outer_smooth": 1.015, "front_keep": 1.015, "heavy_track": 1.018},
    "松山弘平": {"outer_smooth": 1.015, "front_keep": 1.015, "heavy_track": 1.018},
    "岩田望": {"inner_pocket": 1.016, "high_speed": 1.015},
    "岩田望来": {"inner_pocket": 1.016, "high_speed": 1.015},
    "岩田康": {"inner_pocket": 1.022, "pace_make": 1.018, "heavy_track": 1.022},
    "岩田康誠": {"inner_pocket": 1.022, "pace_make": 1.018, "heavy_track": 1.022},
    "C.デムーロ": {"inner_pocket": 1.020, "high_speed": 1.018, "heavy_track": 1.022},
    "M.デムーロ": {"makuri": 1.020, "outer_smooth": 1.015, "heavy_track": 1.020},
    "丹内": {"makuri": 1.018, "front_keep": 1.015, "heavy_track": 1.018},
    "丹内祐次": {"makuri": 1.018, "front_keep": 1.015, "heavy_track": 1.018},
    "古川吉": {"makuri": 1.016, "outer_smooth": 1.015},
    "古川吉洋": {"makuri": 1.016, "outer_smooth": 1.015},
    "幸": {"front_keep": 1.015, "inner_pocket": 1.015, "heavy_track": 1.018},
    "幸英明": {"front_keep": 1.015, "inner_pocket": 1.015, "heavy_track": 1.018},
    "津村": {"front_keep": 1.015, "inner_pocket": 1.015},
    "津村明秀": {"front_keep": 1.015, "inner_pocket": 1.015},
    "菊沢": {"front_keep": 1.012, "outer_smooth": 1.012},
    "菊沢一樹": {"front_keep": 1.012, "outer_smooth": 1.012},
}


def cached_get_html(url: str, headers: dict, timeout: float = 3.5, use_cache: bool = True) -> Optional[str]:
    if use_cache and url in html_response_cache:
        return html_response_cache[url]
    try:
        time.sleep(0.08)
        res = http_session.get(url, headers=headers, timeout=timeout)
        if res.status_code != 200:
            return None
        if "euc-jp" in res.text[:400].lower():
            res.encoding = "euc-jp"
        else:
            res.encoding = res.apparent_encoding
        text = res.text
        if use_cache and len(html_response_cache) < 120:
            html_response_cache[url] = text
        return text
    except Exception:
        return None


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


def format_seconds_to_time_str(sec_val: float) -> str:
    try:
        m = int(sec_val // 60)
        s = sec_val - (m * 60)
        return f"{m}:{s:04.1f}"
    except:
        return "-"


def parse_date_str(d_str: str) -> Optional[datetime]:
    if not d_str:
        return None
    s = str(d_str).strip().replace('.', '/').replace('-', '/')
    m = re.search(r'(20\d{2})/(\d{1,2})/(\d{1,2})', s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except:
            return None
    return None


def extract_target_race_date(shutuba_html: str) -> Optional[datetime]:
    title_m = re.search(r'<title>(.*?)</title>', shutuba_html, re.IGNORECASE | re.DOTALL)
    if title_m:
        t_str = title_m.group(1)
        d_m = re.search(r'(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日', t_str)
        if d_m:
            try:
                return datetime(int(d_m.group(1)), int(d_m.group(2)), int(d_m.group(3)))
            except:
                pass
    return None


def extract_course_rail_info(shutuba_html: str) -> tuple[str, str]:
    m_rail = re.search(r'\b([A-DＡ-Ｄ])コース\b', shutuba_html)
    if m_rail:
        rail_char = m_rail.group(1).translate(str.maketrans('ＡＢＣＤ', 'ABCD'))
        if rail_char == 'A':
            return 'A', '🌱 仮柵: Aコース（内ラチ最内・イン好位および標準状態）'
        elif rail_char == 'B':
            return 'B', '🌱 仮柵: Bコース（Aから3m外移動・傷んだイン保護で好位抜け出し有利）'
        elif rail_char == 'C':
            return 'C', '🌱 仮柵: Cコース（Aから6m外移動・中枠〜外目の差しが届きやすい傾向）'
        elif rail_char == 'D':
            return 'D', '🌱 仮柵: Dコース（Aから9m外移動・直線外持ち出しの持続差し警戒）'
    return '', ''


def get_expected_standard_time(c_stats: dict, course_type: str, distance: int) -> float:
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
    if 'class_factor' in record and record['class_factor'] is not None:
        try:
            val = float(record['class_factor'])
            r_check = str(record.get('race_name', '')) + str(record.get('grade', ''))
            if any(k in r_check for k in ['G1', 'GI', '優駿', 'ダービー', '皐月賞', '菊花賞', '桜花賞', '秋華賞', 'ホープフル', '天皇賞', '有馬記念', '宝塚記念', 'ジャパンC', 'エリザベス女王', 'マイルCS']):
                return max(val, 1.22)
            if any(k in r_check for k in ['G2', 'GII', '神戸新聞', 'セントライト', '京都新聞', '青葉賞', '弥生賞', 'スプリングS', 'ローズS', '紫苑S', 'アルゼンチン', 'オールカマー']):
                return max(val, 1.12)
            return val
        except:
            pass

    r_name = str(record.get('race_name', '')) + " " + str(record.get('grade', ''))
    g1_keywords = [
        'G1', 'GI', '優駿', 'ダービー', '皐月賞', '菊花賞', '桜花賞', '秋華賞',
        '天皇賞', '有馬記念', '宝塚記念', 'ジャパンC', 'ジャパンカップ', 'ホープフル',
        '朝日杯', '阪神JF', 'NHKマイル', '安田記念', 'マイルCS', 'スプリンターズ',
        '高松宮記念', 'ヴィクトリア', 'エリザベス女王', '大阪杯', 'フェブラリー', 'チャンピオンズ'
    ]
    g2_keywords = [
        'G2', 'GII', '神戸新聞', 'セントライト', '京都新聞', '青葉賞', '弥生賞',
        'スプリングS', 'ローズS', '紫苑S', 'フローラS', 'チューリップ', '毎日王冠',
        '京都大賞典', 'オールカマー', '札幌記念', '日経賞', '阪神大賞典', '日経新春', '金鯱賞', '東スポ杯', 'アルゼンチン'
    ]
    g3_keywords = [
        'G3', 'GIII', '共同通信', 'きさらぎ', '京成杯', 'シンザン', 'フェアリー',
        'クイーンC', '毎日杯', 'フラワーC', 'ファルコン', 'アーリントン', 'ラジオNIKKEI',
        'レパード', 'ユニコーン', '新潟記念', '小倉記念', '七夕賞', 'エプソム', 'みやこS', '武蔵野S', 'デイリー杯', '福島記念'
    ]

    for kw in g1_keywords:
        if kw in r_name: return 1.22
    for kw in g2_keywords:
        if kw in r_name: return 1.12
    for kw in g3_keywords:
        if kw in r_name: return 1.04

    if '3勝' in r_name or '1600万' in r_name: return 0.88
    if '2勝' in r_name or '1000万' in r_name: return 0.78
    if '1勝' in r_name or '500万' in r_name: return 0.68
    if '未勝利' in r_name or '新馬' in r_name: return 0.58
    if 'OP' in r_name or 'L' in r_name or 'ステークス' in r_name or '賞' in r_name or '特別' in r_name:
        return 0.94
    return 0.72


def infer_margin_seconds(record: dict) -> float:
    if 'margin' in record and record['margin'] is not None:
        try:
            return max(0.0, float(record['margin']))
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


def estimate_horse_early_3f_speed(bamei_str: str) -> float:
    records = get_active_horse_records(bamei_str)[-5:]
    if not records:
        return 35.8

    early_3f_estimates = []
    for r in records:
        try:
            c_str = str(r.get('course', ''))
            m_d = re.search(r'(\d{3,4})', c_str)
            if not m_d: continue
            dist = int(m_d.group(1))
            if dist <= 800: continue
            sec = parse_time_to_seconds(r.get('time'))
            up = float(r.get('up', 0))
            if sec is None or up <= 30.0 or sec <= up: continue

            base_first_3f = ((sec - up) / (dist - 600.0)) * 600.0
            passing = str(r.get('passing', ''))
            first_corner = 6
            if passing and passing != '-':
                p_nums = [int(x) for x in passing.split('-') if x.strip().isdigit()]
                if p_nums: first_corner = p_nums[0]

            pos_adj = (first_corner - 5) * 0.08
            est_val = base_first_3f + pos_adj
            if 32.5 <= est_val <= 39.5:
                early_3f_estimates.append(est_val)
        except:
            continue

    if not early_3f_estimates:
        return 35.8
    return round(float(np.mean(early_3f_estimates)), 2)


def evaluate_lap_gear_aptitude(
    bamei_str: str,
    course_type: str,
    target_distance: int,
    cushion_val: float,
    track_cond: str,
    expected_pace: str
) -> tuple[float, str]:
    """
    ・芝×良馬場: 33秒台前半の『⚡極限瞬発ギア◎』や『🔥L4F持続ロンスパ◎』を評価
    ・芝×道悪（稍重/重/不良 または クッション値8.9以下）: タフ馬場・道悪好走実績を『☔芝道悪・パワーギア◎』として評価
    ・ダート×道悪（稍重/重/不良）: 砂が締まって脚抜きが良くなるため、ダート道悪好走や高速巡航実績を『🌊ダート脚抜き高速ギア◎』として評価
    """
    records = get_active_horse_records(bamei_str)[-8:]
    if not records:
        return 1.00, ""

    best_turf_up = 99.0
    agari_top2_cnt = 0
    sustained_good_runs = 0
    wet_turf_good_runs = 0
    wet_dirt_good_runs = 0
    tough_agari_good_runs = 0

    for r in records:
        try:
            c_str = str(r.get('course', ''))
            rk = float(r.get('rank', 99))
            mg = infer_margin_seconds(r)
            up_val = float(r.get('up', 99.0))
            ag_rk = int(r.get('agari_rank', 99))
            cnd = str(r.get('cond', '良'))
            pci_val = calc_single_pci(c_str, r.get('time', ''), r.get('up', ''))

            if cnd in ['稍', '重', '不'] and (rk <= 3.0 or mg <= 0.3):
                if 'ダ' in c_str:
                    wet_dirt_good_runs += 1
                else:
                    wet_turf_good_runs += 1

            if up_val >= 35.0 and (ag_rk <= 3 or rk <= 2.0) and (rk <= 4.0 or mg <= 0.35):
                tough_agari_good_runs += 1

            if '芝' in c_str and 31.5 <= up_val <= 38.0:
                if rk <= 5.0 or mg <= 0.5:
                    if up_val < best_turf_up:
                        best_turf_up = up_val

            if ag_rk in [1, 2] and (rk <= 5.0 or mg <= 0.4):
                agari_top2_cnt += 1

            if pci_val is not None and 46.5 <= pci_val <= 51.5 and (rk <= 3.0 or mg <= 0.25):
                sustained_good_runs += 1
        except:
            continue

    gear_mult = 1.00
    gear_label = ""

    if course_type == "ダ" and track_cond in ["稍", "重", "不"]:
        # ダート道悪（脚抜き高速馬場）
        if wet_dirt_good_runs >= 2:
            gear_mult = 1.045
            gear_label = f"🌊ダート脚抜き高速巧者◎(道悪好走{wet_dirt_good_runs}回)"
        elif wet_dirt_good_runs == 1 or agari_top2_cnt >= 2:
            gear_mult = 1.025
            gear_label = "🌊湿ダート高速適性○"
    elif course_type == "芝" and (track_cond in ["稍", "重", "不"] or cushion_val <= 8.9):
        # 芝道悪（タフ・パワー消耗馬場）
        if wet_turf_good_runs >= 2 or (wet_turf_good_runs >= 1 and tough_agari_good_runs >= 1):
            gear_mult = 1.050 if track_cond in ["重", "不"] else 1.038
            gear_label = f"☔芝道悪巧者・パワーギア◎(道悪好走{wet_turf_good_runs}回)"
        elif wet_turf_good_runs == 1 or tough_agari_good_runs >= 2:
            gear_mult = 1.030 if track_cond in ["重", "不"] else 1.022
            gear_label = "🌧️タフ馬場・消耗戦ギア○"
        elif sustained_good_runs >= 2:
            gear_mult = 1.018
            gear_label = "🔥L4F持続ロンスパ◎"
    else:
        # 良馬場（芝・ダート標準〜高速クッション）
        is_fast_turf = (course_type == "芝" and track_cond == "良" and cushion_val >= 9.4)
        if is_fast_turf and best_turf_up <= 33.4:
            gear_mult = 1.035 if agari_top2_cnt >= 1 else 1.025
            gear_label = f"⚡極限瞬発ギア◎(最高上り{best_turf_up:.1f}秒)"
        elif course_type == "芝" and best_turf_up <= 33.8 and agari_top2_cnt >= 2:
            gear_mult = 1.022
            gear_label = f"⚡鋭伸ギア○(上り上位{agari_top2_cnt}回)"
        elif sustained_good_runs >= 2 and expected_pace in ["S_LONSPA", "M_LONSPA", "H_MAEKUZURE", "M_JIRIKI"]:
            gear_mult = 1.022
            gear_label = "🔥L4F持続ロンスパ◎"

    return gear_mult, gear_label


def evaluate_box_in_and_lead_risk(
    wakuban_val: float,
    umaban_val: float,
    kyakushitsu_val: float,
    odds_val: float,
    kinryou_val: float,
    stage1_rank: int,
    stage1_gap_to_2nd: float,
    total_horses: int,
    all_horses_early_info: List[Dict[str, Any]],
    course_type: str,
    distance: int,
    cushion_val: float,
    track_cond: str = "良",
    fav_closer_skew: bool = False
) -> tuple[float, List[str], str]:
    waku = int(round(float(wakuban_val)))
    umaban = int(round(float(umaban_val)))
    kyaku = int(round(float(kyakushitsu_val)))
    odds = float(odds_val) if odds_val else 10.0
    kinryou = float(kinryou_val) if kinryou_val else 55.0

    is_dominant_lone_leader = (stage1_rank == 1 and stage1_gap_to_2nd >= 3.0)
    is_wet_dirt = (course_type == "ダ" and track_cond in ["稍", "重", "不"])

    my_early_info = next((x for x in all_horses_early_info if x["umaban"] == umaban), None)
    my_early_3f = my_early_info["early_3f"] if my_early_info else 35.8

    outer_front_runners = sum(
        1 for x in all_horses_early_info
        if x["umaban"] > umaban and x["kyakushitsu"] <= 2 and x["early_3f"] <= my_early_3f + 0.3
    )
    total_nige_horses = [x for x in all_horses_early_info if x["kyakushitsu"] == 1]
    total_front_horses = [x for x in all_horses_early_info if x["kyakushitsu"] <= 2]

    is_front_scarce_or_skewed = fav_closer_skew or (len(total_front_horses) <= max(3, total_horses // 4))

    tactical_mult = 1.000
    badges: List[str] = []
    risk_level = "NONE"

    if kyaku == 1:
        faster_or_equal_nige = [
            x for x in total_nige_horses
            if x["umaban"] != umaban and x["early_3f"] <= my_early_3f + 0.25
        ]
        if len(total_nige_horses) == 1 or len(faster_or_equal_nige) == 0:
            if is_wet_dirt:
                tactical_mult *= 1.045
                badges.append("🌊ダート脚抜き高速・単騎逃げ◎")
            else:
                tactical_mult *= 1.038 if is_front_scarce_or_skewed else 1.025
                badges.append("🚀単騎ハナ濃厚・セーフティリード◎" if is_front_scarce_or_skewed else "🚀単騎ハナ濃厚◎")
        elif len(faster_or_equal_nige) >= 2:
            if is_dominant_lone_leader:
                tactical_mult *= 0.992
                badges.append("🛡️1強地力(同型多数も克服圏)")
            else:
                tactical_mult *= 0.972
                badges.append("⚡ハナ競り合い消耗注意")
                risk_level = "PACE_PRESSURE"

    elif kyaku == 2:
        if is_wet_dirt and my_early_3f <= 35.3:
            tactical_mult *= 1.032
            badges.append("🌊ダート脚抜き高速・先行押し切り◎")
        elif waku <= 5 and outer_front_runners <= 2 and my_early_3f <= 35.5:
            tactical_mult *= 1.028 if is_front_scarce_or_skewed else 1.015
            badges.append("🎯好位先手スムーズ○")
        elif is_front_scarce_or_skewed:
            tactical_mult *= 1.025
            if odds >= 7.0:
                badges.append("🔥前残り警戒穴(人気馬後方牽制)")
            else:
                badges.append("🔥好位セーフティリード◎")

        if distance >= 2000 and kinryou <= 56.0 and odds >= 6.0:
            tactical_mult *= 1.018
            if "🔥前残り警戒穴(人気馬後方牽制)" not in badges:
                badges.append("🔥軽斤量×好位粘り込み警戒")

    is_crowded_field = (total_horses >= 14)
    is_inner_trap_zone = (waku <= 3) or (waku == 4 and kyaku >= 3 and total_horses >= 15)

    if is_crowded_field and is_inner_trap_zone:
        if kyaku >= 3 and outer_front_runners >= 3:
            if is_dominant_lone_leader:
                tactical_mult *= 0.992
                badges.append("🛡️1強地力(内枠包まれ克服圏)")
                risk_level = "PROTECTED_STRONG"
            elif stage1_rank <= 5 or odds <= 10.0:
                tactical_mult *= 0.968 if waku <= 2 else 0.975
                badges.append("🚨多頭数イン包まれ警戒")
                risk_level = "HIGH_BOX_RISK"
            else:
                badges.append("💡内枠イン突き穴候補")

        elif kyaku == 2 and waku <= 2 and outer_front_runners >= 4:
            if is_dominant_lone_leader:
                tactical_mult *= 0.994
                badges.append("🛡️1強地力(最内包囲網も突破圏)")
                risk_level = "PROTECTED_STRONG"
            elif stage1_rank <= 4 or odds <= 6.0:
                tactical_mult *= 0.978
                badges.append("⚠️内枠包まれ・仕掛け遅れ注意")
                risk_level = "MOD_BOX_RISK"

    if is_crowded_field and waku >= 6 and kyaku >= 2 and stage1_rank <= 3:
        tactical_mult *= 1.015
        badges.append("🟢外枠スムーズ加速◎(包まれ回避)")

    return round(tactical_mult, 3), badges, risk_level


def evaluate_jockey_stage_synergy(
    kishu_name: str,
    wakuban: int,
    kyakushitsu: int,
    distance: int,
    cushion_val: float,
    makuri_flag: bool,
    track_cond: str = "良"
) -> tuple[float, str]:
    clean_k = re.sub(r'[▲△☆◇★\s]', '', str(kishu_name or ""))
    prof = None
    for k_key, k_val in JOCKEY_STAGE_BONUSES.items():
        if k_key in clean_k:
            prof = k_val
            break
    if not prof:
        return 1.000, ""

    mult = 1.000
    notes = []
    if (track_cond in ["稍", "重", "不"] or cushion_val <= 8.9) and "heavy_track" in prof:
        mult = max(mult, prof["heavy_track"])
        notes.append("道悪エスコート◎")
    if distance >= 2400 and "long_dist" in prof:
        mult = max(mult, prof["long_dist"])
        notes.append("長距離手腕◎")
    if cushion_val >= 9.5 and track_cond == "良" and "high_speed" in prof:
        mult = max(mult, prof["high_speed"])
        notes.append("高速馬場手腕◎")
    if wakuban <= 3 and "inner_pocket" in prof:
        mult = max(mult, prof["inner_pocket"])
        notes.append("イン捌き◎")
    if wakuban >= 6 and kyakushitsu >= 3 and "outer_smooth" in prof:
        mult = max(mult, prof["outer_smooth"])
        notes.append("外エスコート◎")
    if kyakushitsu == 1 and "pace_make" in prof:
        mult = max(mult, prof["pace_make"])
        notes.append("逃げペース配分◎")
    if makuri_flag and "makuri" in prof:
        mult = max(mult, prof["makuri"])
        notes.append("まくり仕掛け◎")

    label = f"🏇{notes[0]}" if notes else ""
    return round(mult, 3), label


def build_horse_history_details(bamei_str: str, place_dict: Dict[str, str]) -> List[Dict[str, str]]:
    records = get_active_horse_records(bamei_str)[-5:]
    history_details = []
    for r in reversed(records):
        course_raw = str(r.get('course', '05_芝_2000'))
        parts = course_raw.split('_')
        p_code = parts[0] if len(parts) >= 1 else '05'
        c_type = parts[1] if len(parts) >= 2 else '芝'
        dist = parts[2] if len(parts) >= 3 else '2000'
        place = place_dict.get(p_code, "地方")
        cond_str = str(r.get('cond', '良'))

        raw_rname = str(r.get('race_name', '')).strip()
        grade_s = str(r.get('grade', '')).strip()
        distance_label = f"{place}{c_type}{dist}m({cond_str})"

        if raw_rname:
            display_race_name = f"{raw_rname}({grade_s})" if (grade_s and grade_s not in raw_rname) else raw_rname
        else:
            display_race_name = f"{place} {c_type}{dist}m ({cond_str})"

        single_pci = calc_single_pci(course_raw, r.get('time', ''), r.get('up', ''))
        rank_num = r.get('rank', 99)
        try:
            rank_str = f"{int(float(rank_num))}着"
        except:
            rank_str = "-"

        up_raw = str(r.get('up', '-')).strip()
        up_str = f"{up_raw}秒" if up_raw not in ['-', '', 'None'] else "-"

        history_details.append({
            "date": str(r.get('date', '')),
            "race_name": display_race_name,
            "distance": distance_label,
            "rank": rank_str,
            "time": str(r.get('time', '-')),
            "up": up_str,
            "passing": str(r.get('passing', '-')),
            "pci": f"{single_pci}" if single_pci is not None else "-"
        })
    return history_details


def normalize_race_title(raw_title: str) -> str:
    if not raw_title:
        return ""
    s = str(raw_title)
    s = re.split(r'[|｜]', s)[0]
    s = re.sub(r'出馬表.*$', '', s)
    s = re.sub(r'結果・払戻.*$', '', s)
    s = re.sub(r'データ分析.*$', '', s)
    s = re.sub(r'第\d+回\s*', '', s)
    s = s.replace('（', '(').replace('）', ')')
    s = s.translate(str.maketrans('０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    s = re.sub(r'\((?:G1|G2|G3|GI|GII|GIII|J\.G1|J\.G2|J\.G3|L|OP)\)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'【.*?】', '', s)
    s = re.sub(r'\s+', '', s).strip()
    return s


def is_generic_class_race_name(clean_name: str) -> bool:
    if not clean_name:
        return True
    generic_keywords = ['新馬', '未勝利', '1勝クラス', '2勝クラス', '3勝クラス', '500万下', '1000万下', '1600万下', 'オープン']
    return any(kw in clean_name for kw in generic_keywords)


def is_same_race_series(target_clean: str, cand_clean: str) -> bool:
    if not target_clean or not cand_clean:
        return False
    if target_clean == cand_clean:
        return True
    if "天皇賞" in target_clean or "天皇賞" in cand_clean:
        if ("春" in target_clean) != ("春" in cand_clean): return False
        if ("秋" in target_clean) != ("秋" in cand_clean): return False
        return "天皇賞" in target_clean and "天皇賞" in cand_clean

    t_short = target_clean.replace('ステークス', 'S').replace('カップ', 'C').replace('記念', '')
    c_short = cand_clean.replace('ステークス', 'S').replace('カップ', 'C').replace('記念', '')
    if t_short and c_short and (t_short == c_short or t_short in c_short or c_short in t_short):
        return True
    return False


def extract_html_race_metadata(html_text: str) -> tuple[str, Optional[str], Optional[int]]:
    title_str = ""
    m_rname = re.search(r'<div[^>]*class="[^"]*RaceName[^"]*"[^>]*>(.*?)</div>', html_text, re.IGNORECASE | re.DOTALL)
    if m_rname:
        title_str = re.sub(r'<[^>]*>', '', m_rname.group(1)).strip()
    if not title_str:
        m_h1 = re.search(r'<h1[^>]*>(.*?)</h1>', html_text, re.IGNORECASE | re.DOTALL)
        if m_h1:
            title_str = re.sub(r'<[^>]*>', '', m_h1.group(1)).strip()
    if not title_str:
        m_title = re.search(r'<title>(.*?)</title>', html_text, re.IGNORECASE | re.DOTALL)
        if m_title:
            title_str = re.sub(r'<[^>]*>', '', m_title.group(1)).strip()

    clean_title = normalize_race_title(title_str)
    c_type = None
    dist_val = None
    m_td = re.search(r'(芝|ダート|ダ|障害|障)\s*(?:右|左|直|外|内|\s|・)*?(\d{3,4})\s*m', html_text)
    if m_td:
        raw_t = m_td.group(1)
        c_type = 'ダ' if raw_t in ['ダート', 'ダ'] else ('障' if raw_t in ['障害', '障'] else '芝')
        dist_val = int(m_td.group(2))
    return clean_title, c_type, dist_val


def parse_race_result_benchmark_from_html(
    html_text: str,
    label_prefix: str,
    expected_course_type: Optional[str] = None,
    expected_distance: Optional[int] = None,
    expected_race_name: Optional[str] = None,
    strict_validation: bool = False
) -> Optional[Dict[str, Any]]:
    if not html_text or ("HorseList" not in html_text and "race_table_01" not in html_text):
        return None

    cand_clean_title, cand_course_type, cand_distance = extract_html_race_metadata(html_text)
    target_clean_title = normalize_race_title(expected_race_name or "")
    is_named_race = not is_generic_class_race_name(target_clean_title)

    if strict_validation:
        if expected_course_type and cand_course_type and expected_course_type != cand_course_type:
            return None
        if is_named_race and target_clean_title and cand_clean_title:
            if not is_same_race_series(target_clean_title, cand_clean_title):
                return None
            if expected_distance and cand_distance and abs(expected_distance - cand_distance) > 400:
                return None
        else:
            if expected_distance and cand_distance and expected_distance != cand_distance:
                return None

    rows = re.findall(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', html_text, re.DOTALL | re.IGNORECASE)
    if not rows:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html_text, re.DOTALL | re.IGNORECASE)

    top3_list = []
    for row in rows:
        tds = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
        if len(tds) < 8:
            continue
        rank_str = re.sub(r'<[^>]*>', '', tds[0]).strip()
        if not rank_str.isdigit():
            continue
        rk = int(rank_str)
        if rk < 1 or rk > 3:
            continue

        umaban = 0
        if len(tds) >= 3:
            u_cand = re.sub(r'<[^>]*>', '', tds[2]).strip()
            if u_cand.isdigit():
                umaban = int(u_cand)

        bamei = ""
        b_m = re.search(r'<span[^>]*class="[^"]*Horse_Name[^"]*"[^>]*>(.*?)</span>', row, re.IGNORECASE | re.DOTALL)
        if not b_m:
            b_m = re.search(r'<a[^>]*href="[^"]*horse/\d+[^"]*"[^>]*>(.*?)</a>', row, re.IGNORECASE | re.DOTALL)
        if b_m:
            bamei = re.sub(r'<[^>]*>', '', b_m.group(1)).strip()

        row_clean = re.sub(r'\s+', ' ', re.sub(r'<[^>]*>', ' ', row)).strip()
        time_str = "-"
        m_tm = re.search(r'(\d:\d{2}\.\d)', row_clean)
        if m_tm:
            time_str = m_tm.group(1)

        passing_str = "-"
        p_m = re.search(r'<td[^>]*class="[^"]*PassageRate[^"]*"[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
        if p_m:
            p_cand = re.sub(r'<[^>]*>', '', p_m.group(1)).strip()
            if p_cand:
                passing_str = p_cand
        if passing_str == "-":
            m_ps = re.search(r'\b(\d{1,2}(?:-\d{1,2})+)\b', row_clean)
            if m_ps:
                passing_str = m_ps.group(1)

        up_str = "-"
        for td_cell in reversed(tds[7:]):
            c_txt = re.sub(r'<[^>]*>', '', td_cell).strip()
            if re.match(r'^(3[0-9]\.\d|4[0-5]\.\d)$', c_txt):
                up_str = c_txt
                break

        top3_list.append({
            "rank": rk,
            "umaban": umaban,
            "bamei": bamei or f"{umaban}番",
            "time": time_str,
            "up": up_str,
            "passing": passing_str
        })
        if len(top3_list) >= 3:
            break

    if not top3_list:
        return None

    top3_list.sort(key=lambda x: x["rank"])
    winner_time = top3_list[0]["time"] if top3_list else "-"
    if winner_time == "-":
        return None

    if expected_distance:
        win_sec = parse_time_to_seconds(winner_time)
        if win_sec is not None:
            if not (expected_distance * 0.052 <= win_sec <= expected_distance * 0.075):
                return None

    year_m = re.search(r'(20\d{2})年', html_text)
    year_str = f"{year_m.group(1)}年" if year_m else label_prefix
    display_label = f"{label_prefix} [{cand_clean_title}]" if (cand_clean_title and cand_clean_title not in label_prefix) else label_prefix

    return {
        "label": display_label,
        "year_label": year_str,
        "winner_time": winner_time,
        "winner_bamei": top3_list[0]["bamei"],
        "winner_umaban": top3_list[0]["umaban"],
        "top3": top3_list
    }


def fetch_reference_race_benchmarks(
    race_id: str,
    headers: dict,
    c_stats: dict,
    course_type: str,
    distance: int,
    race_name: str = "",
    start_ts: float = 0.0
) -> Dict[str, Any]:
    current_year_bench = None
    prev_year_bench = None

    cur_html = cached_get_html(
        f"https://race.netkeiba.com/race/result.html?race_id={race_id}",
        headers, timeout=3.0
    )
    if cur_html:
        current_year_bench = parse_race_result_benchmark_from_html(
            cur_html,
            f"今年度確定 ({race_id[:4]}年)",
            expected_course_type=course_type,
            expected_distance=distance,
            expected_race_name=race_name,
            strict_validation=False
        )

    if len(race_id) == 12 and race_id.isdigit() and (start_ts == 0.0 or (time.time() - start_ts) < 7.0):
        prev_year = int(race_id[:4]) - 1
        place_code = race_id[4:6]
        kai_num = int(race_id[6:8])
        day_num = int(race_id[8:10])
        r_num = int(race_id[10:12])

        cand_ids: List[str] = []
        data_html = cached_get_html(
            f"https://race.netkeiba.com/race/data_list.html?race_id={race_id}",
            headers, timeout=2.5
        )
        if data_html:
            for fid in re.findall(rf'\b({prev_year}\d{{8}})\b', data_html):
                if fid not in cand_ids:
                    cand_ids.append(fid)

        direct_prev_id = f"{prev_year}{race_id[4:]}"
        if direct_prev_id not in cand_ids:
            cand_ids.append(direct_prev_id)

        for k_diff in [-1, 1]:
            adj_kai = kai_num + k_diff
            if 1 <= adj_kai <= 6:
                adj_id = f"{prev_year}{place_code}{adj_kai:02d}{day_num:02d}{r_num:02d}"
                if adj_id not in cand_ids:
                    cand_ids.append(adj_id)

        for cid in cand_ids[:3]:
            if start_ts > 0.0 and (time.time() - start_ts) > 7.5:
                break
            p_html = cached_get_html(
                f"https://race.netkeiba.com/race/result.html?race_id={cid}",
                headers, timeout=2.5
            )
            if p_html:
                parsed_p = parse_race_result_benchmark_from_html(
                    p_html,
                    f"前年実績 ({prev_year}年)",
                    expected_course_type=course_type,
                    expected_distance=distance,
                    expected_race_name=race_name,
                    strict_validation=True
                )
                if parsed_p:
                    prev_year_bench = parsed_p
                    break

    std_sec = get_expected_standard_time(c_stats, course_type, distance)
    std_time_str = format_seconds_to_time_str(std_sec)
    top3_up_list = c_stats.get('top3_up', [])
    avg_up_str = f"{float(np.mean(top3_up_list)):.1f}" if top3_up_list else "34.8"

    fallback_bench = {
        "label": f"同舞台コース基準値 ({course_type}{distance}m)",
        "year_label": "コース統計平均",
        "winner_time": std_time_str,
        "winner_bamei": "同舞台平均",
        "winner_umaban": 0,
        "top3": [
            {"rank": 1, "umaban": 0, "bamei": "1着平均目安", "time": std_time_str, "up": avg_up_str, "passing": "-"},
            {"rank": 2, "umaban": 0, "bamei": "2着平均目安", "time": "-", "up": avg_up_str, "passing": "-"},
            {"rank": 3, "umaban": 0, "bamei": "3着平均目安", "time": "-", "up": avg_up_str, "passing": "-"},
        ]
    }

    primary_bench = current_year_bench or prev_year_bench or fallback_bench
    return {
        "is_finished_race": current_year_bench is not None,
        "primary": primary_bench,
        "current_year": current_year_bench,
        "prev_year": prev_year_bench or fallback_bench
    }


def extract_pedigree_from_row(row_html: str, bamei: str) -> Dict[str, str]:
    sire = ""
    bms = ""
    db_entry = horse_history_db.get(bamei, {})
    if isinstance(db_entry, dict):
        sire = str(db_entry.get('sire', db_entry.get('father', db_entry.get('父', '')))).strip()
        bms = str(db_entry.get('bms', db_entry.get('mother_father', db_entry.get('母父', '')))).strip()

    first_past_pos = row_html.find('class="Past')
    info_chunk = row_html[:first_past_pos] if first_past_pos != -1 else row_html

    if not sire:
        m_h1 = re.search(r'<div[^>]*class="[^"]*Horse01[^"]*"[^>]*>(.*?)</div>', info_chunk, re.IGNORECASE | re.DOTALL)
        if m_h1:
            cand = re.sub(r'<[^>]*>', '', m_h1.group(1)).strip()
            if cand and cand != bamei:
                sire = cand

    if not bms:
        m_h3 = re.search(r'<div[^>]*class="[^"]*(?:Horse03|Horse04)[^"]*"[^>]*>(.*?)</div>', info_chunk, re.IGNORECASE | re.DOTALL)
        if m_h3:
            c3 = re.sub(r'<[^>]*>', '', m_h3.group(1)).strip()
            m_paren = re.search(r'[（\(]([^）\)]+)[）\)]', c3)
            if m_paren:
                bms = m_paren.group(1).strip()

    clean_info = re.sub(r'<[^>]*>', ' ', info_chunk)
    bamei_pos = clean_info.find(bamei)
    if bamei_pos != -1:
        before_bamei = clean_info[:bamei_pos]
        after_bamei = clean_info[bamei_pos + len(bamei):]
        if not sire:
            for s_name in SIRE_APTITUDE_DB.keys():
                if s_name in before_bamei:
                    sire = s_name
                    break
        if not bms:
            for s_name in SIRE_APTITUDE_DB.keys():
                if s_name in after_bamei:
                    bms = s_name
                    break

    if bamei in KNOWN_HORSE_PEDIGREE:
        if not sire:
            sire = KNOWN_HORSE_PEDIGREE[bamei].get("sire", "")
        if not bms:
            bms = KNOWN_HORSE_PEDIGREE[bamei].get("bms", "")

    return {"sire": sire, "bms": bms}


def fetch_and_merge_shutuba_past(race_id: str, headers: dict, bamei_list: List[str]) -> tuple[bool, int]:
    place_map_rev = {'札幌':'01', '函館':'02', '福島':'03', '新潟':'04', '東京':'05', '中山':'06', '中京':'07', '京都':'08', '阪神':'09', '小倉':'10'}
    enriched_horse_count = 0

    for bamei in bamei_list:
        base_recs = copy.deepcopy(horse_history_db.get(bamei, {}).get('race_records', []))
        runtime_enriched_db[bamei] = base_recs
        if bamei in KNOWN_HORSE_PEDIGREE and bamei not in runtime_pedigree_db:
            runtime_pedigree_db[bamei] = copy.deepcopy(KNOWN_HORSE_PEDIGREE[bamei])

    try:
        url = f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}"
        past_html = cached_get_html(url, headers, timeout=4.5)
        if not past_html or ("Past" not in past_html and "HorseList" not in past_html):
            return False, 0

        rows = re.findall(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', past_html, re.DOTALL | re.IGNORECASE)
        for row in rows:
            bamei = ""
            b_m = re.search(r'<div[^>]*class="[^"]*Horse02[^"]*"[^>]*>(.*?)</div>', row, re.IGNORECASE | re.DOTALL)
            if not b_m:
                b_m = re.search(r'<span[^>]*class="[^"]*HorseName[^"]*"[^>]*>(.*?)</span>', row, re.IGNORECASE | re.DOTALL)
            if not b_m:
                b_m = re.search(r'<a[^>]*href="[^"]*horse/\d+[^"]*"[^>]*>(.*?)</a>', row, re.IGNORECASE | re.DOTALL)
            if b_m:
                bamei = re.sub(r'<[^>]*>', '', b_m.group(1)).strip()
            if not bamei or bamei not in runtime_enriched_db:
                continue

            ped_info = extract_pedigree_from_row(row, bamei)
            if ped_info.get("sire") or ped_info.get("bms"):
                runtime_pedigree_db[bamei] = ped_info

            past_tds = re.findall(r'<td[^>]*class="[^"]*Past[^"]*"[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
            if not past_tds:
                continue

            parsed_recent = []
            for p_html in reversed(past_tds):
                p_text = re.sub(r'\s+', ' ', re.sub(r'<[^>]*>', ' ', p_html)).strip()
                if not p_text or len(p_text) < 10:
                    continue

                date_str = ""
                m_date = re.search(r'(20\d{2}[./]\d{1,2}[./]\d{1,2})', p_text)
                if m_date:
                    date_str = m_date.group(1).replace('.', '/')

                place_code = "05"
                for p_name, p_c in place_map_rev.items():
                    if p_name in p_text:
                        place_code = p_c
                        break

                rank_val = None
                m_rk = re.search(r'<span[^>]*class="[^"]*Num[^"]*"[^>]*>\s*(\d{1,2})\s*</span>', p_html, re.IGNORECASE)
                if m_rk:
                    rank_val = float(m_rk.group(1))
                else:
                    m_rk2 = re.search(r'20\d{2}[./]\d{1,2}[./]\d{1,2}\s+\S+\s+(\d{1,2})\b', p_text)
                    if m_rk2:
                        rank_val = float(m_rk2.group(1))
                if rank_val is None:
                    continue

                grade_str = ""
                if "Icon_GradeType1" in p_html or "(GI)" in p_text or "(G1)" in p_text or "GI" in p_text:
                    grade_str = "G1"
                elif "Icon_GradeType2" in p_html or "(GII)" in p_text or "(G2)" in p_text or "GII" in p_text:
                    grade_str = "G2"
                elif "Icon_GradeType3" in p_html or "(GIII)" in p_text or "(G3)" in p_text or "GIII" in p_text:
                    grade_str = "G3"
                elif "Icon_GradeType4" in p_html or "Icon_GradeType5" in p_html or "(L)" in p_text or "(OP)" in p_text:
                    grade_str = "OP"

                race_title = ""
                m_rname = re.search(r'<div[^>]*class="[^"]*Data02[^"]*"[^>]*>(.*?)</div>', p_html, re.IGNORECASE | re.DOTALL)
                if m_rname:
                    race_title = re.sub(r'\s+', ' ', re.sub(r'<[^>]*>', '', m_rname.group(1)).strip())

                c_type, dist_val, time_str, cond_str = "芝", 2000, "-", "良"
                m_cd = re.search(r'(芝|ダ|障)\s*(\d{3,4})', p_text)
                if m_cd:
                    c_type = m_cd.group(1)
                    dist_val = int(m_cd.group(2))
                m_tm = re.search(r'(\d:\d{2}\.\d)', p_text)
                if m_tm:
                    time_str = m_tm.group(1)
                m_cnd = re.search(r'\b(良|稍|重|不)\b', p_text)
                if m_cnd:
                    cond_str = m_cnd.group(1)

                passing_str = "-"
                m_pass = re.search(r'(\d{1,2}(?:-\d{1,2})+)', p_text)
                if m_pass:
                    passing_str = m_pass.group(1)

                up_val = "-"
                m_up = re.search(r'\((\d{2}\.\d)\)', p_text)
                if m_up and 31.0 <= float(m_up.group(1)) <= 44.0:
                    up_val = m_up.group(1)
                else:
                    for m_cand in re.finditer(r'\b(3[1-9]\.\d|4[0-3]\.\d)\b', p_text):
                        up_val = m_cand.group(1)
                        break

                agari_rank = 99
                if re.search(r'class="[^"]*(?:Rank_1|BgYellow)[^"]*"', p_html, re.IGNORECASE):
                    agari_rank = 1
                elif re.search(r'class="[^"]*(?:Rank_2|BgOrange)[^"]*"', p_html, re.IGNORECASE):
                    agari_rank = 2
                elif re.search(r'class="[^"]*(?:Rank_3|BgBlue|Rank_3)[^"]*"', p_html, re.IGNORECASE):
                    agari_rank = 3

                margin_val = None
                m_data06 = re.search(r'<div[^>]*class="[^"]*Data06[^"]*"[^>]*>(.*?)</div>', p_html, re.IGNORECASE | re.DOTALL)
                if m_data06:
                    d6_txt = re.sub(r'<[^>]*>', '', m_data06.group(1)).strip()
                    m_mg = re.search(r'\(([+-]?\d+\.\d+)\)', d6_txt)
                    if m_mg:
                        raw_mg = float(m_mg.group(1))
                        margin_val = 0.0 if rank_val == 1.0 or raw_mg < 0 else raw_mg

                w_val = None
                m_wt = re.search(r'(\d{3})\([+-]?\d+\)', p_text)
                if m_wt:
                    w_val = float(m_wt.group(1))

                parsed_recent.append({
                    "date": date_str,
                    "course": f"{place_code}_{c_type}_{dist_val}",
                    "race_name": race_title,
                    "grade": grade_str,
                    "rank": rank_val,
                    "time": time_str,
                    "up": up_val,
                    "agari_rank": agari_rank,
                    "passing": passing_str,
                    "cond": cond_str,
                    "margin": margin_val,
                    "weight": w_val
                })

            if parsed_recent:
                existing_list = runtime_enriched_db[bamei]
                for new_r in parsed_recent:
                    matched = False
                    for ex_r in existing_list:
                        same_time = (new_r["time"] != "-" and str(ex_r.get("time", "")) == new_r["time"])
                        same_course = (new_r["course"] == str(ex_r.get("course", "")))
                        same_rank = (int(new_r["rank"]) == int(float(ex_r.get("rank", 99))))
                        if (same_time and same_course) or (same_course and same_rank and new_r["up"] == str(ex_r.get("up", ""))):
                            if new_r["date"]: ex_r["date"] = new_r["date"]
                            if new_r["margin"] is not None: ex_r["margin"] = new_r["margin"]
                            if new_r["grade"]: ex_r["grade"] = new_r["grade"]
                            if new_r["race_name"]: ex_r["race_name"] = new_r["race_name"]
                            if new_r["passing"] != "-": ex_r["passing"] = new_r["passing"]
                            if new_r["agari_rank"] <= 3: ex_r["agari_rank"] = new_r["agari_rank"]
                            if new_r["weight"] and not ex_r.get("weight"): ex_r["weight"] = new_r["weight"]
                            matched = True
                            break
                    if not matched:
                        existing_list.append(new_r)

                runtime_enriched_db[bamei] = existing_list
                enriched_horse_count += 1

        return enriched_horse_count > 0, enriched_horse_count
    except Exception:
        return False, 0


def get_active_horse_records(bamei_str: str) -> List[Dict[str, Any]]:
    if bamei_str in runtime_enriched_db and runtime_enriched_db[bamei_str]:
        return runtime_enriched_db[bamei_str]
    return horse_history_db.get(bamei_str, {}).get('race_records', [])


def get_horse_pedigree_info(bamei_str: str) -> Dict[str, str]:
    if bamei_str in runtime_pedigree_db and (runtime_pedigree_db[bamei_str].get("sire") or runtime_pedigree_db[bamei_str].get("bms")):
        return runtime_pedigree_db[bamei_str]
    if bamei_str in KNOWN_HORSE_PEDIGREE:
        return KNOWN_HORSE_PEDIGREE[bamei_str]
    db_entry = horse_history_db.get(bamei_str, {})
    if isinstance(db_entry, dict):
        sire = str(db_entry.get('sire', db_entry.get('father', db_entry.get('父', '')))).strip()
        bms = str(db_entry.get('bms', db_entry.get('mother_father', db_entry.get('母父', '')))).strip()
        if sire or bms:
            return {"sire": sire, "bms": bms}
    return {"sire": "", "bms": ""}


def evaluate_pedigree_and_distance_aptitude(
    bamei: str, course_type: str, target_distance: int, track_cond: str, cushion_val: float
) -> tuple[float, float, str]:
    """
    ・芝×道悪: パワー・道悪血統（heavy）＋芝道悪好走実績を最大1.08倍ブースト
    ・ダート×道悪: 脚抜き高速スピード血統（dirt_wet）＋湿ダート好走実績をブースト
    """
    ped = get_horse_pedigree_info(bamei)
    sire = ped.get("sire", "")
    bms = ped.get("bms", "")

    default_prof = {"stamina": 1.0, "heavy": 1.0, "lonspa": 1.0, "dirt_wet": 1.5}
    sire_prof = SIRE_APTITUDE_DB.get(sire, default_prof)
    bms_prof = SIRE_APTITUDE_DB.get(bms, default_prof)

    stamina_index = (sire_prof["stamina"] * 0.75) + (bms_prof["stamina"] * 0.25) if bms else sire_prof["stamina"]
    heavy_index = (sire_prof["heavy"] * 0.70) + (bms_prof["heavy"] * 0.30) if bms else sire_prof["heavy"]
    lonspa_index = (sire_prof["lonspa"] * 0.75) + (bms_prof["lonspa"] * 0.25) if bms else sire_prof["lonspa"]
    dirt_wet_index = (sire_prof.get("dirt_wet", 1.5) * 0.75) + (bms_prof.get("dirt_wet", 1.5) * 0.25) if bms else sire_prof.get("dirt_wet", 1.5)

    records = get_active_horse_records(bamei)
    max_good_dist = 0
    proven_long_dist_wins = 0
    proven_heavy_turf_runs = 0
    proven_heavy_dirt_runs = 0

    for r in records:
        try:
            rk = float(r.get('rank', 99))
            mg = infer_margin_seconds(r)
            c_str = str(r.get('course', ''))
            m_d = re.search(r'(\d{3,4})', c_str)
            d_val = int(m_d.group(1)) if m_d else 0
            cnd = str(r.get('cond', '良'))

            if rk <= 3.0 or mg <= 0.3:
                if d_val > max_good_dist:
                    max_good_dist = d_val
                if target_distance >= 2200 and d_val >= 2200 and rk <= 2.0:
                    proven_long_dist_wins += 1
                if cnd in ['稍', '重', '不'] and (rk <= 3.0 or mg <= 0.25):
                    if 'ダ' in c_str:
                        proven_heavy_dirt_runs += 1
                    else:
                        proven_heavy_turf_runs += 1
        except:
            continue

    dist_extension = (target_distance - max_good_dist) if max_good_dist > 0 else 0
    dist_mult = 1.00
    notes = []

    if target_distance >= 2200:
        if stamina_index >= 2.5:
            dist_mult *= 1.035 if target_distance >= 2500 else 1.025
            notes.append("中長距離血統◎")
        elif stamina_index >= 2.0:
            dist_mult *= 1.015
            notes.append("距離血統○")
        elif stamina_index <= -0.5:
            dist_mult *= 0.955

        if dist_extension >= 400 and stamina_index >= 2.6:
            dist_mult *= 1.012
        if proven_long_dist_wins >= 1:
            dist_mult *= 1.022
            notes.append("距離実績◎")

    elif target_distance <= 1400:
        if stamina_index <= -1.0:
            dist_mult *= 1.02
        elif stamina_index >= 2.8:
            dist_mult *= 0.97

    heavy_mult = 1.00
    if course_type == "ダ" and track_cond in ["稍", "重", "不"]:
        if dirt_wet_index >= 2.5:
            heavy_mult *= 1.035
            notes.append("🌊湿ダート血統◎")
        if proven_heavy_dirt_runs >= 1:
            heavy_mult *= 1.025
            notes.append("🌊湿ダート実績◎")
    elif course_type == "芝" and (track_cond in ["稍", "重", "不"] or cushion_val <= 8.9):
        if heavy_index >= 2.5:
            heavy_mult *= 1.045 if track_cond in ["重", "不"] else 1.030
            notes.append("☔芝道悪血統◎")
        elif heavy_index >= 2.0:
            heavy_mult *= 1.025 if track_cond in ["重", "不"] else 1.015
            notes.append("☔芝道悪血統○")
        elif heavy_index <= 0.7 and track_cond in ["重", "不"]:
            heavy_mult *= 0.960

        if proven_heavy_turf_runs >= 2:
            heavy_mult *= 1.035
            notes.append("☔道悪実績◎")
        elif proven_heavy_turf_runs == 1:
            heavy_mult *= 1.020
            notes.append("☔道悪実績○")

    lonspa_mult = 1.012 if (target_distance >= 2000 and lonspa_index >= 1.9) else 1.00

    stage1_ped_mult = float(np.clip(dist_mult, 0.94, 1.05))
    stage2_ped_mult = float(np.clip(dist_mult * heavy_mult * lonspa_mult, 0.92, 1.09))

    ped_label = ""
    if sire:
        bms_str = f"×母父{bms}" if bms else ""
        note_str = f"({','.join(notes[:2])})" if notes else ""
        ped_label = f"🧬父{sire}{bms_str}{note_str}"
    elif notes:
        ped_label = f"🧬{','.join(notes[:2])}"

    return stage1_ped_mult, stage2_ped_mult, ped_label


def evaluate_rotation_aptitude(
    bamei: str, target_race_date: Optional[datetime], power_score: float, target_distance: int
) -> tuple[float, str]:
    records = get_active_horse_records(bamei)
    if not records:
        return 1.00, "初出走/履歴なし"

    dated_recs = []
    for r in records:
        dt = parse_date_str(r.get("date", ""))
        if dt is not None:
            dated_recs.append((dt, r))

    if not dated_recs:
        return 1.00, "ローテ標準"

    dated_recs.sort(key=lambda x: x[0])
    last_dt, last_rec = dated_recs[-1]

    if target_race_date is None or target_race_date <= last_dt:
        if len(dated_recs) >= 2:
            prev_dt, _ = dated_recs[-2]
            if (last_dt - prev_dt).days >= 65:
                return (1.032 if power_score >= 0.74 else 1.022), "🔄叩き2走目(上昇◎)"
        return 1.00, "順調"

    days_since_last = (target_race_date - last_dt).days
    weeks_approx = max(1, round(days_since_last / 7.0)) - 1
    gap_before_last = (last_dt - dated_recs[-2][0]).days if len(dated_recs) >= 2 else 0
    gap_before_prev2 = (dated_recs[-2][0] - dated_recs[-3][0]).days if len(dated_recs) >= 3 else 0
    last_cf = infer_class_factor(last_rec)

    if gap_before_last >= 60 and 18 <= days_since_last <= 52:
        if power_score >= 0.75 or last_cf >= 1.04:
            return 1.035, f"🔄叩き2走目・中{weeks_approx}週(本番上昇◎)"
        return 1.022, f"🔄叩き2走目・中{weeks_approx}週(上積み○)"

    if gap_before_prev2 >= 60 and 18 <= gap_before_last <= 48 and 18 <= days_since_last <= 52:
        return 1.018, f"🔄叩き3走目・中{weeks_approx}週(充実期○)"

    if days_since_last >= 150:
        return (0.990 if power_score >= 0.80 else 0.965), f"休養明け({days_since_last}日ぶり)"

    if 70 <= days_since_last < 150:
        return (1.005 if power_score >= 0.78 else 0.985), f"休み明け・中{weeks_approx}週"

    if days_since_last <= 14:
        return (0.970 if target_distance >= 2200 else 0.990), f"間隔詰まり(中{max(0, weeks_approx)}週)"

    if 21 <= days_since_last <= 52:
        return 1.010, f"好ローテ(中{weeks_approx}週)"

    return 1.000, f"中{weeks_approx}週"


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
    records = get_active_horse_records(bamei_str)
    if not records: return 50.0, "B"
    good_pcis = []
    all_pcis = []
    for r in records:
        pci = calc_single_pci(r.get('course', ''), r.get('time', ''), r.get('up', ''))
        if pci is not None:
            all_pcis.append(pci)
            try:
                if 1.0 <= float(r.get('rank', 99)) <= 3.0:
                    good_pcis.append(pci)
            except:
                pass
    target_pcis = good_pcis if good_pcis else all_pcis
    if not target_pcis: return 50.0, "B"
    avg_pci = round(float(np.mean(target_pcis)), 1)
    return avg_pci, get_pci_label(avg_pci)


def get_horse_real_stats(bamei_str):
    records = get_active_horse_records(bamei_str)
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
        except:
            pass
    avg_agari = round(float(np.mean(up_list)), 2) if up_list else 35.0

    makuri_flag = False
    recent_scores = []
    career_scores = []
    graded_floor_score = 0.0
    agari_rank_bonus = 0.0

    scan_records = records[-12:]
    recent_start_idx = max(0, len(scan_records) - 5)

    for idx, r in enumerate(scan_records):
        is_recent_5 = (idx >= recent_start_idx)
        passing = str(r.get('passing', ''))
        if '-' in passing and is_recent_5:
            try:
                p_nums = [int(x) for x in passing.split('-') if x.isdigit()]
                if len(p_nums) >= 2 and (p_nums[0] - p_nums[-1]) >= 3:
                    makuri_flag = True
            except:
                pass

        cf = infer_class_factor(r)
        margin = infer_margin_seconds(r)
        try:
            rk = float(r.get('rank', 99))
        except:
            rk = 99.0

        course_str = str(r.get('course', ''))
        m_dist = re.search(r'(\d{3,4})', course_str)
        dist = float(m_dist.group(1)) if m_dist else 2000.0

        capped_margin = min(max(margin, 0.0), 2.0)
        adjusted_margin = capped_margin * (2000.0 / dist) if dist > 0 else capped_margin
        run_score = float(np.clip(cf - (adjusted_margin * 0.14), 0.10, 1.22))

        if cf <= 0.68:
            run_score = min(run_score, 0.72)
        elif cf <= 0.78:
            run_score = min(run_score, 0.80)

        if cf >= 1.20:
            if rk <= 3.0 or margin <= 0.2:
                graded_floor_score = max(graded_floor_score, 0.85)
            elif rk <= 5.0 or margin <= 0.4:
                graded_floor_score = max(graded_floor_score, 0.81)
        elif cf >= 1.10:
            if rk <= 1.0 or margin <= 0.1:
                graded_floor_score = max(graded_floor_score, 0.82)
            elif rk <= 3.0 or margin <= 0.3:
                graded_floor_score = max(graded_floor_score, 0.77)

        ag_rk = int(r.get('agari_rank', 99))
        single_pci = calc_single_pci(course_str, r.get('time', ''), r.get('up', ''))
        if ag_rk == 1:
            agari_rank_bonus += 0.018 * (1.2 if cf >= 1.00 else 0.7)
        elif ag_rk in [2, 3]:
            agari_rank_bonus += 0.010 * (1.1 if cf >= 1.00 else 0.6)
        elif single_pci is not None and single_pci >= 53.5 and rk <= 4.0 and cf >= 1.00:
            agari_rank_bonus += 0.016

        career_scores.append(run_score)
        if is_recent_5:
            recent_scores.append(run_score)

    agari_rank_bonus = min(agari_rank_bonus, 0.055)

    if recent_scores:
        weights = np.linspace(1.0, 2.5, len(recent_scores))
        recent_weighted = float(np.average(recent_scores, weights=weights))
        sorted_career = sorted(career_scores, reverse=True)
        top2_peak = float(np.mean(sorted_career[:2])) if len(sorted_career) >= 2 else sorted_career[0]
        base_power = (recent_weighted * 0.72) + (top2_peak * 0.28)
        if graded_floor_score > base_power:
            base_power = (base_power * 0.45) + (graded_floor_score * 0.55)
        power_score = float(np.clip(base_power + agari_rank_bonus, 0.15, 1.25))
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


def parse_finished_race_html_odds(
    result_html: str, bamei_to_umaban: Optional[Dict[str, int]] = None
) -> tuple[Dict[int, float], Dict[int, tuple[float, float]]]:
    odds_map: Dict[int, float] = {}
    weight_map: Dict[int, tuple[float, float]] = {}

    if not result_html or "HorseList" not in result_html:
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
            if bamei_to_umaban:
                b_m = re.search(r'<span[^>]*class="[^"]*Horse_Name[^"]*"[^>]*>(.*?)</span>', row, re.IGNORECASE | re.DOTALL)
                if not b_m:
                    b_m = re.search(r'<a[^>]*href="[^"]*horse/\d+[^"]*"[^>]*>(.*?)</a>', row, re.IGNORECASE | re.DOTALL)
                if b_m:
                    b_clean = re.sub(r'<[^>]*>', '', b_m.group(1)).strip()
                    if b_clean in bamei_to_umaban:
                        umaban = bamei_to_umaban[b_clean]

            if umaban == 0 and len(tds) >= 3:
                cand_umaban = re.sub(r'<[^>]*>', '', tds[2]).strip()
                if cand_umaban.isdigit():
                    u_val = int(cand_umaban)
                    if 1 <= u_val <= 18:
                        umaban = u_val
            if umaban == 0:
                continue

            odds_val = None
            o_m = re.search(r'<td[^>]*class="[^"]*Odds\s+Txt_R[^"]*"[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
            if o_m:
                o_txt = re.sub(r'<[^>]*>', '', o_m.group(1)).strip().replace(',', '')
                m_fl = re.search(r'(\d+\.\d+)', o_txt)
                if m_fl:
                    odds_val = float(m_fl.group(1))

            if odds_val is None:
                odds_cells = re.findall(r'<td[^>]*class="[^"]*Odds[^"]*"[^>]*>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
                for cell in odds_cells:
                    c_txt = re.sub(r'<[^>]*>', '', cell).strip().replace(',', '')
                    m_fl = re.search(r'^(\d+\.\d+)$', c_txt)
                    if m_fl:
                        odds_val = float(m_fl.group(1))
                        break

            if odds_val is None and len(tds) >= 11:
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
    odds_map: Dict[int, float] = {}
    js_headers = {
        'User-Agent': headers.get('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
        'Referer': f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}',
        'X-Requested-With': 'XMLHttpRequest',
        'Accept': 'application/json, text/javascript, */*; q=0.01'
    }

    url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?pid=api_get_jra_odds&race_id={race_id}&type=1&action=init&sort=odds&compress=0&output=json"
    api_status_str = ""
    try:
        time.sleep(0.08)
        resp = http_session.get(url, headers=js_headers, timeout=3.5)
        if resp.status_code == 200:
            payload = resp.json()
            if isinstance(payload, dict):
                status_val = str(payload.get("status", "")).lower()
                data_obj = payload.get("data")
                if isinstance(data_obj, dict):
                    odds_root = data_obj.get("odds")
                    if isinstance(odds_root, dict):
                        tan_dict = odds_root.get("1")
                        if isinstance(tan_dict, dict):
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
    except:
        pass

    if len(odds_map) >= 2:
        return odds_map, ("js_yoso" if api_status_str == "yoso" else "js_realtime")
    return {}, "none"


def fetch_hybrid_odds_and_result_weights(
    race_id: str, shutuba_html: str, headers: dict, bamei_to_umaban: Optional[Dict[str, int]] = None
) -> tuple[Dict[int, float], Dict[int, tuple[float, float]], str, str]:
    res_html = cached_get_html(
        f"https://race.netkeiba.com/race/result.html?race_id={race_id}",
        headers, timeout=3.5
    )
    if res_html:
        html_odds, html_weights = parse_finished_race_html_odds(res_html, bamei_to_umaban)
        if len(html_odds) >= 2:
            return (
                html_odds,
                html_weights,
                "html_result",
                f"✅ オッズ自動取得 [終了レース・HTML形式]: レース結果HTMLから確定単勝オッズ（{len(html_odds)}頭）を反映しました。"
            )

    js_odds, js_mode = fetch_future_race_js_odds(race_id, headers)
    if len(js_odds) >= 2:
        if js_mode == "js_yoso":
            return (
                js_odds,
                {},
                "js_yoso",
                f"ℹ️ オッズ自動取得 [未来レース・予想オッズ]: JS APIから予想オッズ（{len(js_odds)}頭）を反映しました。"
            )
        return (
            js_odds,
            {},
            "js_realtime",
            f"✅ オッズ自動取得 [最新オッズ]: JS APIから単勝オッズ（{len(js_odds)}頭）を反映しました。"
        )

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
            f"✅ オッズ自動取得 [出馬表HTML]: 単勝オッズ（{len(inline_odds)}頭）を反映しました。"
        )

    return (
        {},
        {},
        "manual_required",
        "⚠️ オッズ未発売のため初期値(10.0倍)をセットしました。必要に応じて手動で入力してください。"
    )


def infer_auto_condition_override(
    bamei: str, weight_val: float, change_val: float, weight_announced: bool,
    zenso: float, power_score: float, kaisuu: int
) -> tuple[str, str]:
    if not weight_announced:
        return "1.00", "馬体重未発表（手動設定可）"

    records = get_active_horse_records(bamei)
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


def auto_detect_expected_pace(
    horses_data: list, course_rpci: float, track_condition: str = "良", course_type: str = "芝"
) -> tuple[str, str, str]:
    if not horses_data or len(horses_data) < 2:
        return "M_JIRIKI", "manual_required", "⚠️ 出走馬データが不足しているため標準ペース（ミドル①）にセットしました。"

    total_h = len(horses_data)
    nige_cnt = sum(1 for h in horses_data if int(round(h.get("kyakushitsu", 2))) == 1)
    senkou_cnt = sum(1 for h in horses_data if int(round(h.get("kyakushitsu", 2))) == 2)
    makuri_cnt = sum(1 for h in horses_data if h.get("makuri_flag", False))
    fast_early_cnt = sum(1 for h in horses_data if h.get("early_3f_sec", 35.8) <= 34.8)
    front_ratio = (nige_cnt + senkou_cnt) / float(total_h)
    avg_pci = float(np.mean([h.get("pci", 50.0) for h in horses_data]))

    valid_odds_h = [h for h in horses_data if h.get("odds", 0) > 1.0]
    fav5 = sorted(valid_odds_h, key=lambda x: x["odds"])[:5] if len(valid_odds_h) >= 5 else horses_data[:5]
    fav5_closers = sum(1 for h in fav5 if int(round(h.get("kyakushitsu", 2))) >= 3)

    if course_type == "ダ" and track_condition in ["稍", "重", "不"]:
        pace = "M_JIRIKI" if nige_cnt <= 2 else "H_MAEKUZURE"
        msg = f"✅ 展開自動判定: ダート{track_condition}馬場（脚抜き高速・前止まらない傾向）➔ 『{pace}（先行スピード重視）』を自動セット"
    elif nige_cnt >= 3 or (nige_cnt >= 2 and (front_ratio >= 0.48 or fast_early_cnt >= 3)) or (course_rpci < 48.0 and nige_cnt >= 2):
        pace = "H_MAEKUZURE"
        msg = f"✅ 展開自動判定: 逃げ{nige_cnt}頭・先行{senkou_cnt}頭（テン速馬{fast_early_cnt}頭）で先行激化 ➔ 『ハイ① 前崩れ消耗戦』を自動セット"
    elif course_type == "芝" and track_condition in ["重", "不"] and nige_cnt >= 1:
        pace = "M_LONSPA"
        msg = f"✅ 展開自動判定: 芝・道悪（{track_condition}馬場）×先行争い ➔ 上がりがかかる『ミドル② タフ持続ロンスパ戦』を自動セット"
    elif fav5_closers >= 4 and nige_cnt <= 1:
        pace = "S_LONSPA" if makuri_cnt >= 1 else "S_YOIDON"
        msg = f"✅ 展開自動判定: 上位人気5頭中{fav5_closers}頭が差し・追込で後方牽制濃厚（前の伏兵残り警戒！）➔ 『{pace}』を自動セット"
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
    if not res_html or "HorseList" not in res_html:
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
    dt = extract_target_race_date(shutuba_html)
    if dt is not None:
        return dt.weekday()
    act_m = re.search(r'<dd[^>]*class="[^"]*Active[^"]*"[^>]*>(.*?)</dd>', shutuba_html, re.IGNORECASE | re.DOTALL)
    if act_m:
        act_txt = act_m.group(1)
        if "土" in act_txt: return 5
        if "日" in act_txt: return 6
        if "月" in act_txt: return 0
    if day_num == 1:
        return 5
    return None


def auto_detect_track_bias(
    race_id: str, target_track_type: str, shutuba_html: str, rail_code: str = "", rail_msg: str = "",
    track_cond: str = "良", start_ts: float = 0.0
) -> tuple[str, str, str, str]:
    try:
        if len(race_id) < 12 or not race_id.isdigit():
            return "フラット", "中", "manual_required", "⚠️ トラックバイアスは手動で設定してください。"

        prefix_8 = race_id[:8]
        day_num = int(race_id[8:10])
        race_num = int(race_id[10:12])
        rail_suffix = f" （{rail_msg}）" if rail_msg else ""

        if start_ts > 0.0 and (time.time() - start_ts) > 8.0:
            return "フラット", "中", "manual_required", f"ℹ️ 高速レスポンス優先のためバイアス手動設定モードにしました。{rail_suffix}"

        weekday = detect_target_race_weekday(shutuba_html, day_num)
        is_saturday_race = (weekday == 5) or (day_num == 1)

        headers = {'User-Agent': 'Mozilla/5.0'}
        collected_top3 = []
        sampled_same_day = 0
        sampled_prev_day = 0

        if race_num >= 2:
            checked_cnt = 0
            for r_no in range(race_num - 1, 0, -1):
                if sampled_same_day >= 3 or checked_cnt >= 4:
                    break
                if start_ts > 0.0 and (time.time() - start_ts) > 8.5:
                    break
                checked_cnt += 1
                cid = f"{prefix_8}{day_num:02d}{r_no:02d}"
                r_html = cached_get_html(f"https://race.netkeiba.com/race/result.html?race_id={cid}", headers, timeout=2.2)
                if r_html is None:
                    break
                parsed = parse_single_race_result_for_bias(r_html, target_track_type)
                if parsed is None and r_no == race_num - 1:
                    break
                if parsed:
                    collected_top3.extend(parsed)
                    sampled_same_day += 1

        if sampled_same_day == 0 and day_num >= 2 and not is_saturday_race:
            prev_day = day_num - 1
            for r_no in [12, 11, 10, 9]:
                if sampled_prev_day >= 3:
                    break
                if start_ts > 0.0 and (time.time() - start_ts) > 8.5:
                    break
                cid = f"{prefix_8}{prev_day:02d}{r_no:02d}"
                r_html = cached_get_html(f"https://race.netkeiba.com/race/result.html?race_id={cid}", headers, timeout=2.2)
                if not r_html:
                    break
                parsed = parse_single_race_result_for_bias(r_html, target_track_type)
                if parsed is None and r_no == 12:
                    break
                if parsed:
                    collected_top3.extend(parsed)
                    sampled_prev_day += 1

        total_sampled_races = sampled_same_day + sampled_prev_day

        if total_sampled_races < 1 or len(collected_top3) < 3:
            if target_track_type == "ダ" and track_cond in ["稍", "重", "不"]:
                return (
                    "前残り (逃げ・先行有利)", "中", "auto_detected",
                    f"✅ バイアス自動推定 [ダート{track_cond}馬場]: 砂が締まり脚抜きが良いため『前残り（逃げ・先行有利）』を初期セットしました。"
                )
            if target_track_type == "芝" and day_num <= 2 and rail_code in ["A", "B"]:
                return (
                    "イン前有利 (1〜4枠・先行)", "弱", "auto_detected",
                    f"✅ バイアス自動推定 [開催{day_num}日目・{rail_code}コース開幕週]: 芝保護状態につき『イン前有利（強度:弱）』を初期セットしました。{rail_suffix}"
                )
            return (
                "フラット", "中", "manual_required",
                f"⚠️ 参照できる当日・前日の同コース確定結果がまだないため、トラックバイアスは手動で設定してください。{rail_suffix}"
            )

        total_horses = len(collected_top3)
        inner_rate = sum(1 for x in collected_top3 if x["waku"] <= 4) / float(total_horses)
        outer_rate = sum(1 for x in collected_top3 if x["waku"] >= 5) / float(total_horses)

        valid_pass = [x["last_corner"] for x in collected_top3 if x["last_corner"] > 0]
        if valid_pass:
            front_rate = sum(1 for p in valid_pass if p <= 4) / float(len(valid_pass))
            closer_rate = sum(1 for p in valid_pass if p >= 6) / float(len(valid_pass))
        else:
            front_rate, closer_rate = 0.5, 0.5

        detected_bias = "フラット"
        detected_strength = "中"

        if inner_rate >= 0.58 and front_rate >= 0.55:
            detected_bias = "イン前有利 (1〜4枠・先行)"
            detected_strength = "極端" if (inner_rate >= 0.70 and front_rate >= 0.65) else ("中" if (inner_rate >= 0.63 or front_rate >= 0.60) else "弱")
        elif outer_rate >= 0.58 and closer_rate >= 0.50:
            detected_bias = "外差し有利 (5〜8枠・差し追込)"
            detected_strength = "極端" if (outer_rate >= 0.70 and closer_rate >= 0.60) else ("中" if (outer_rate >= 0.63 or closer_rate >= 0.55) else "弱")
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

        source_label = f"当日{sampled_same_day}鞍" if sampled_same_day > 0 else f"前日{sampled_prev_day}鞍"
        summary_msg = (
            f"✅ バイアス自動判定 [{source_label} / 上位{total_horses}頭集計]："
            f"内枠率{int(inner_rate*100)}%・4角4番手以内率{int(front_rate*100)}% ➔ "
            f"『{detected_bias}（強度:{detected_strength}）』を自動セットしました。{rail_suffix}"
        )
        return detected_bias, detected_strength, "auto_detected", summary_msg
    except Exception:
        return "フラット", "中", "manual_required", "⚠️ トラックバイアスは手動で設定してください。"


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
    training_rank: Optional[str] = "B"
    rotation_mult: Optional[float] = 1.00
    rotation_label: Optional[str] = ""
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
    cushion_value: float = 9.5


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Keiba Simulator API is running."}


@app.get("/fetch_race/{race_id}")
def fetch_race(race_id: str):
    start_ts = time.time()
    clean_race_id = re.sub(r'\D', '', str(race_id or ""))
    if not clean_race_id:
        return {"status": "error", "message": "有効なレースID（半角数字12桁）を入力してください"}

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    html = None
    for attempt in range(2):
        html_cand = cached_get_html(
            f"https://race.netkeiba.com/race/shutuba.html?race_id={clean_race_id}",
            headers, timeout=6.0, use_cache=(attempt == 0)
        )
        if html_cand and ("HorseList" in html_cand or "馬番" in html_cand):
            html = html_cand
            break
        time.sleep(0.3)

    if not html:
        return {"status": "error", "message": "出馬表データが見つかりませんでした。レースIDを確認してください。"}

    try:
        title_match = re.search(r'<title>(.*?)</title>', html)
        race_name = title_match.group(1).split("|")[0].strip() if title_match else "レース不明"
        target_race_date = extract_target_race_date(html)
        rail_code, rail_msg = extract_course_rail_info(html)

        place_code = clean_race_id[4:6] if len(clean_race_id) >= 6 else '05'
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
            track_condition = "稍" if c_val == "稍重" else ("不" if c_val == "不良" else c_val)

        horse_blocks = re.findall(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)

        bamei_to_umaban: Dict[str, int] = {}
        bamei_list: List[str] = []
        for block in horse_blocks:
            try:
                u_m = re.search(r'<td[^>]*class="[^"]*(?:Umaban)[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                u_str = re.sub(r'<[^>]*>', '', u_m.group(1)).strip() if u_m else ""
                if not u_str.isdigit(): continue
                u_num = int(u_str)
                b_m = re.search(r'<span[^>]*class="[^"]*HorseName[^"]*"[^>]*>(.*?)</span>', block, re.IGNORECASE | re.DOTALL)
                if not b_m: b_m = re.search(r'<a[^>]*href="[^"]*horse/\d+[^"]*"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
                if b_m:
                    b_str = re.sub(r'<[^>]*>', '', b_m.group(1)).strip()
                    if b_str:
                        bamei_to_umaban[b_str] = u_num
                        bamei_list.append(b_str)
            except:
                continue

        past_merged_ok, merged_cnt = fetch_and_merge_shutuba_past(clean_race_id, headers, bamei_list)
        hybrid_odds_map, result_weight_map, odds_status, odds_msg = fetch_hybrid_odds_and_result_weights(
            clean_race_id, html, headers, bamei_to_umaban
        )

        def get_auto_kyakushitsu(bamei_str, target_dist):
            try:
                records = get_active_horse_records(bamei_str)
                if not records: return 2.0
                recent_recs = records[-5:]
                weighted_pos = []
                weights = []
                prev_dist = target_dist
                for idx_r, r in enumerate(recent_recs):
                    passing = str(r.get('passing', ''))
                    if passing and passing != '-':
                        nums = [int(x) for x in passing.split('-') if x.strip().isdigit()]
                        if nums:
                            pos_val = nums[0] * 0.4 + nums[-1] * 0.6
                            w = 2.2 if idx_r >= len(recent_recs) - 2 else 1.0
                            weighted_pos.append(pos_val)
                            weights.append(w)
                    if idx_r == len(recent_recs) - 1:
                        m_d = re.search(r'(\d{3,4})', str(r.get('course', '')))
                        if m_d: prev_dist = int(m_d.group(1))
                if not weighted_pos: return 2.0
                avg_pos = float(np.average(weighted_pos, weights=weights))
                if target_dist - prev_dist >= 300:
                    avg_pos = max(1.0, avg_pos - 1.0)
                if avg_pos <= 2.4: return 1.0
                elif avg_pos <= 6.2: return 2.0
                elif avg_pos <= 10.5: return 3.0
                else: return 4.0
            except:
                return 2.0

        horses_data = []
        any_weight_announced = False

        for block in horse_blocks:
            try:
                u_m = re.search(r'<td[^>]*class="[^"]*(?:Umaban)[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                u_str = re.sub(r'<[^>]*>', '', u_m.group(1)).strip() if u_m else ""
                if not u_str.isdigit(): continue
                umaban = int(u_str)

                waku_m = re.search(r'<td[^>]*class="[^"]*(?:Waku)[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                waku_str = re.sub(r'<[^>]*>', '', waku_m.group(1)).strip() if waku_m else ""
                wakuban = int(waku_str) if waku_str.isdigit() else 0

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

                odds = hybrid_odds_map.get(umaban, 10.0)
                if umaban not in hybrid_odds_map:
                    o_m = re.search(r'<span[^>]*id="odds[^>]*>(.*?)</span>', block, re.IGNORECASE | re.DOTALL)
                    if o_m:
                        o_str = re.sub(r'<[^>]*>', '', o_m.group(1)).strip().replace(',', '')
                        try:
                            val_o = float(o_str)
                            if val_o >= 1.0: odds = val_o
                        except:
                            pass

                weight_val, change_val, weight_announced = 480.0, 0.0, False
                w_m = re.search(r'<td[^>]*class="[^"]*Weight[^"]*"[^>]*>(.*?)</td>', block, re.IGNORECASE | re.DOTALL)
                if w_m:
                    w_str = re.sub(r'<[^>]*>', '', w_m.group(1)).strip()
                    m_weight = re.search(r'(\d{3})\s*\(([+-]?\d+)\)', w_str)
                    if m_weight:
                        weight_val, change_val = float(m_weight.group(1)), float(m_weight.group(2))
                        weight_announced = True
                        any_weight_announced = True
                    elif (m_w2 := re.search(r'(\d{3})', w_str)):
                        weight_val = float(m_w2.group(1))

                if not weight_announced and umaban in result_weight_map:
                    weight_val, change_val = result_weight_map[umaban]
                    weight_announced = True
                    any_weight_announced = True

                h_pci, h_pci_type = get_horse_pci_stats(bamei)
                zenso, agari, kaisuu, power_score, makuri_flag = get_horse_real_stats(bamei)
                early_3f_sec = estimate_horse_early_3f_speed(bamei)

                auto_cond_code, auto_cond_reason = infer_auto_condition_override(
                    bamei, weight_val, change_val, weight_announced, zenso, power_score, kaisuu
                )
                default_cushion = 8.6 if (track_type == "芝" and track_condition in ["稍", "重", "不"]) else 9.5
                _, _, ped_summary = evaluate_pedigree_and_distance_aptitude(
                    bamei, track_type, distance, track_condition, default_cushion
                )
                rot_mult, rot_label = evaluate_rotation_aptitude(
                    bamei, target_race_date, power_score, distance
                )
                _, gear_label = evaluate_lap_gear_aptitude(
                    bamei, track_type, distance, default_cushion, track_condition, "M_JIRIKI"
                )

                reason_parts = [x for x in [auto_cond_reason, rot_label, gear_label, ped_summary] if x]
                history_details = build_horse_history_details(bamei, place_dict)

                horses_data.append({
                    "wakuban": wakuban,
                    "umaban": umaban, "bamei": bamei, "kishu": kishu,
                    "odds": odds, "kinryou": kinryou,
                    "bataijuu": weight_val, "weight_change": change_val,
                    "condition_override": auto_cond_code,
                    "condition_reason": " / ".join(reason_parts),
                    "training_rank": "B",
                    "rotation_mult": round(rot_mult, 3),
                    "rotation_label": rot_label,
                    "zenso_chakujun": zenso, "avg_agari": agari,
                    "kyakushitsu": get_auto_kyakushitsu(bamei, distance),
                    "shussou_kaisuu": kaisuu,
                    "pci": h_pci, "pci_type": h_pci_type,
                    "makuri_flag": makuri_flag,
                    "early_3f_sec": early_3f_sec,
                    "power_score": round(power_score, 3),
                    "history_details": history_details
                })
            except:
                continue

        horses_data = sorted(horses_data, key=lambda x: x['umaban'])
        total_h = len(horses_data)
        for h in horses_data:
            if not h["wakuban"] or h["wakuban"] <= 0:
                h["wakuban"] = calc_jra_wakuban(h["umaban"], total_h)

        valid_odds_h = [h for h in horses_data if h.get("odds", 0) > 1.0]
        fav5 = sorted(valid_odds_h, key=lambda x: x["odds"])[:5] if len(valid_odds_h) >= 5 else horses_data[:5]
        fav_closer_skew = sum(1 for h in fav5 if int(round(h.get("kyakushitsu", 2))) >= 3) >= 4

        early_info_preview = [
            {"umaban": h["umaban"], "wakuban": h["wakuban"], "kyakushitsu": int(round(h["kyakushitsu"])), "early_3f": h.get("early_3f_sec", 35.8)}
            for h in horses_data
        ]
        by_power = sorted(horses_data, key=lambda x: x.get("power_score", 0.5), reverse=True)
        p_gap = (by_power[0]["power_score"] - by_power[1]["power_score"]) * 25.0 if len(by_power) >= 2 else 0.0
        for h in horses_data:
            p_rank = next((i + 1 for i, x in enumerate(by_power) if x["umaban"] == h["umaban"]), 99)
            _, preview_badges, _ = evaluate_box_in_and_lead_risk(
                h["wakuban"], h["umaban"], h["kyakushitsu"], h["odds"], h["kinryou"],
                p_rank, p_gap, total_h, early_info_preview, track_type, distance, 9.5,
                track_cond=track_condition, fav_closer_skew=fav_closer_skew
            )
            if preview_badges:
                h["condition_reason"] = f"{' '.join(preview_badges)} / {h['condition_reason']}"

        course_key = f"{place_code}_{track_type}_{distance}"
        c_stats = course_history_db.get(course_key, {})
        rpci, rpci_type, _ = get_course_rpci_stats(course_key)

        reference_benchmarks = fetch_reference_race_benchmarks(
            clean_race_id, headers, c_stats, track_type, distance, race_name=race_name, start_ts=start_ts
        )
        auto_pace, pace_status, pace_msg = auto_detect_expected_pace(
            horses_data, rpci, track_condition=track_condition, course_type=track_type
        )
        auto_bias, auto_strength, bias_status, bias_msg = auto_detect_track_bias(
            clean_race_id, track_type, html, rail_code=rail_code, rail_msg=rail_msg,
            track_cond=track_condition, start_ts=start_ts
        )

        elapsed_sec = round(time.time() - start_ts, 1)
        db_merge_note = f" [5走柱・芝/ダ雨/展開解析:{merged_cnt}頭 / 取得{elapsed_sec}秒]" if past_merged_ok else f" [サーバーDB使用 / {elapsed_sec}秒]"
        weight_status = "auto_detected" if any_weight_announced else "manual_required"
        weight_msg = f"✅ 馬体重・血統・ローテ・包まれ/前残り印を自動セットしました。{db_merge_note}"

        return {
            "status": "success", "race_name": race_name, "place_name": place_name,
            "course_type": track_type, "distance": distance,
            "weather": weather, "track_condition": track_condition,
            "course_rail_code": rail_code, "course_rail_message": rail_msg,
            "rpci": rpci, "rpci_type": rpci_type,
            "recommended_pace": auto_pace,
            "auto_track_bias": auto_bias, "auto_bias_strength": auto_strength,
            "reference_benchmarks": reference_benchmarks,
            "auto_status": {
                "odds_status": odds_status, "odds_message": odds_msg,
                "pace_status": pace_status, "pace_message": pace_msg,
                "bias_status": bias_status, "bias_message": bias_msg,
                "weight_status": weight_status, "weight_message": weight_msg
            },
            "horses": horses_data
        }
    except Exception as e:
        return {"status": "error", "message": f"パース処理エラー: {str(e)}"}


def get_time_limit_multiplier(horse_name, course_type, current_distance, expected_standard_time, cushion_val, track_cond):
    """
    ・芝の道悪（稍重・重・不良 または クッション値8.9未満）: 消耗戦のため時計制限免除（1.00）
    ・ダートの道悪（稍重・重・不良）: 脚抜きが良く高速決着になるため、高速持ち時計を持つ馬に最大1.025倍のボーナスを付与！
    """
    is_wet_turf = (course_type == "芝") and ((track_cond in ["稍", "重", "不"]) or (cushion_val < 8.9))
    if is_wet_turf or current_distance >= 2500:
        return 1.00

    h_records = get_active_horse_records(horse_name)
    if not h_records:
        return 1.00

    total_runs = len(h_records)
    has_sharp_agari = False
    for r in h_records[-5:]:
        try:
            up_v = float(r.get('up', 99.0))
            ag_rk = int(r.get('agari_rank', 99))
            rk = float(r.get('rank', 99))
            if (course_type == "芝" and up_v <= 33.9 and rk <= 5.0) or (ag_rk <= 2 and rk <= 4.0):
                has_sharp_agari = True
                break
        except:
            continue

    cur_halons = current_distance / 200.0
    same_dist_diffs = []
    adjacent_diffs = []
    allowed_conds = ['良', '稍', '重', '不'] if course_type == "ダ" else ['良', '稍']

    for r in h_records:
        r_c = str(r.get('course', ''))
        r_cond = str(r.get('cond', ''))
        t_sec = parse_time_to_seconds(r.get('time'))
        if t_sec is None or course_type not in r_c or r_cond not in allowed_conds:
            continue

        m = re.search(r'(\d{3,4})', r_c)
        if not m: continue
        dist = int(m.group(1))

        single_pci = calc_single_pci(r_c, r.get('time', ''), r.get('up', ''))
        slow_pace_allowance = min(1.2, (single_pci - 51.0) * 0.22) if (single_pci is not None and single_pci >= 51.5) else 0.0
        effective_t_sec = t_sec - slow_pace_allowance

        if dist == current_distance:
            same_dist_diffs.append((effective_t_sec - expected_standard_time) / cur_halons)
        elif 0 < abs(dist - current_distance) <= 400:
            adj_stats = course_history_db.get(r_c, {})
            adj_std = adj_stats.get('standard_time')
            if adj_std is not None:
                adj_diff = (effective_t_sec - float(adj_std)) / (dist / 200.0)
            else:
                halon_gap = (current_distance - dist) / 200.0
                pace_penalty = halon_gap * 0.45 if halon_gap > 0 else halon_gap * 0.35
                projected_time = (effective_t_sec * (current_distance / dist)) + pace_penalty
                adj_diff = (projected_time - expected_standard_time) / cur_halons
            adjacent_diffs.append(adj_diff)

    best_diff = min(same_dist_diffs) if same_dist_diffs else (min(adjacent_diffs) if adjacent_diffs else None)
    if best_diff is None:
        return 1.00

    is_high_speed = (cushion_val >= 9.5) or (course_type == "ダ" and track_cond in ["稍", "重", "不"])
    if best_diff <= -0.02: mult = 1.025 if is_high_speed else 1.01
    elif best_diff <= 0.06: mult = 1.00
    elif best_diff <= 0.11: mult = 0.97 if is_high_speed else 0.99
    elif best_diff <= 0.16: mult = 0.94 if is_high_speed else 0.97
    else: mult = 0.92 if is_high_speed else 0.95

    if total_runs <= 4 or has_sharp_agari:
        mult = max(mult, 0.97)
    return round(mult, 3)


def calculate_stage1_pure_score(
    raw_lgb_pred_neutral: float, horse: HorseData, power_score: float, time_diff_base: float, stage1_ped_mult: float = 1.0
) -> float:
    market_prob = min(0.55, 0.80 / float(horse.odds)) if horse.odds > 1.0 else 0.08
    agari_factor = 1.035 if time_diff_base <= -0.3 else (1.015 if time_diff_base <= 0.0 else (0.965 if time_diff_base >= 0.6 else 1.0))
    pure_ability = ((raw_lgb_pred_neutral * 0.36) + (power_score * 0.46) + (market_prob * 0.18)) * agari_factor * stage1_ped_mult
    return float(pure_ability)


def calculate_comprehensive_score(
    raw_lgb_pred, horse, power_score, makuri_flag, h_pci, h_pci_type, time_diff,
    expected_pace, track_bias, bias_strength, cushion_value, course_rpci,
    time_limit_mult, stage2_ped_mult: float = 1.0, rotation_mult: float = 1.0,
    training_mult: float = 1.0, stage1_rank: int = 99, total_horses: int = 16,
    track_condition: str = "良", course_type: str = "芝", tactical_mult: float = 1.0,
    gear_mult: float = 1.0, jockey_mult: float = 1.0, distance: int = 2000
):
    market_prob = min(0.55, 0.80 / float(horse.odds)) if horse.odds > 1.0 else 0.08
    if expected_pace in ["M_JIRIKI", "M"]:
        base_ability = (raw_lgb_pred * 0.38) + (power_score * 0.42) + (market_prob * 0.20)
    else:
        base_ability = (raw_lgb_pred * 0.42) + (power_score * 0.38) + (market_prob * 0.20)

    kyaku = int(round(horse.kyakushitsu))
    waku = horse.wakuban
    is_wet_turf = (course_type == "芝") and ((track_condition in ["稍", "重", "不"]) or (cushion_value <= 8.9))
    is_wet_dirt = (course_type == "ダ") and (track_condition in ["稍", "重", "不"])
    is_fast_cushion = (course_type == "芝" and cushion_value >= 9.5 and track_condition == "良")
    is_elite_jiriki = (stage1_rank <= 3) or (power_score >= 0.78)

    pace_mult = 1.0
    if expected_pace in ["S_YOIDON", "S"]:
        if is_wet_turf:
            pace_mult *= 1.06 if h_pci_type in ["A", "B"] else 0.97
        else:
            pace_mult *= 1.10 if h_pci_type == "C" else (1.00 if h_pci_type == "B" else 0.93)
        if kyaku == 1: pace_mult *= 1.07
        elif kyaku == 2: pace_mult *= 1.08
        elif kyaku == 3: pace_mult *= 1.03
        elif kyaku == 4: pace_mult *= 0.90
        if not is_wet_turf:
            if time_diff <= -0.3: pace_mult *= 1.06
            elif time_diff <= 0.1: pace_mult *= 1.03
            elif time_diff >= 0.6: pace_mult *= 0.95

    elif expected_pace == "S_LONSPA":
        if 49.0 <= h_pci <= 54.5: pace_mult *= 1.09
        elif h_pci_type == "C": pace_mult *= 1.03
        else: pace_mult *= 0.97
        if kyaku == 1: pace_mult *= 0.98
        elif kyaku == 2: pace_mult *= 1.07
        elif kyaku == 3: pace_mult *= 1.08
        elif kyaku == 4: pace_mult *= 0.96
        if makuri_flag: pace_mult *= 1.06
        if is_fast_cushion:
            if time_diff <= -0.25: pace_mult *= 1.045
            elif time_diff >= 0.45: pace_mult *= 0.96

    elif expected_pace in ["M_JIRIKI", "M"]:
        if is_wet_turf:
            if h_pci <= 50.5: pace_mult *= 1.06
            elif h_pci_type == "C": pace_mult *= 0.96
        else:
            if h_pci_type == "B": pace_mult *= 1.05
            elif h_pci_type == "C" and is_fast_cushion: pace_mult *= 1.04
            else: pace_mult *= 0.99

        if is_wet_dirt:
            # ダート道悪は逃げ・先行スピード馬が圧倒的有利
            if kyaku == 1: pace_mult *= 1.06
            elif kyaku == 2: pace_mult *= 1.08
            elif kyaku == 3: pace_mult *= 1.01
            else: pace_mult *= 0.91
        else:
            if kyaku == 1: pace_mult *= 0.97
            elif kyaku == 2: pace_mult *= 1.06
            elif kyaku == 3: pace_mult *= 1.06
            elif kyaku == 4: pace_mult *= 0.98 if is_elite_jiriki else 0.94

        if power_score >= 0.72: pace_mult *= 1.05
        if is_fast_cushion:
            if time_diff <= -0.30: pace_mult *= 1.055
            elif time_diff <= -0.05: pace_mult *= 1.025
            elif time_diff >= 0.35: pace_mult *= 0.950

    elif expected_pace == "M_LONSPA":
        if 46.0 <= h_pci <= 51.5: pace_mult *= 1.09
        elif h_pci_type == "A": pace_mult *= 1.06
        else: pace_mult *= 0.95
        if kyaku == 1: pace_mult *= 0.93
        elif kyaku == 2: pace_mult *= 1.05
        elif kyaku == 3: pace_mult *= 1.08
        elif kyaku == 4: pace_mult *= 1.02
        if makuri_flag: pace_mult *= 1.05

    elif expected_pace in ["H_MAEKUZURE", "H"]:
        if h_pci_type == "A": pace_mult *= 1.12
        elif h_pci_type == "B": pace_mult *= 1.03
        else: pace_mult *= 0.93
        if kyaku == 1: pace_mult *= 0.86
        elif kyaku == 2: pace_mult *= 0.95
        elif kyaku == 3: pace_mult *= 1.08
        elif kyaku == 4: pace_mult *= 1.11

    kinryou_mult = 1.00
    if distance >= 2000 or is_wet_turf:
        if horse.kinryou >= 58.5 and kyaku >= 3:
            kinryou_mult = 0.975
        elif horse.kinryou <= 54.0 and kyaku <= 2:
            kinryou_mult = 1.020

    bias_mult = 1.0
    boost = 1.03 if bias_strength == "弱" else (1.15 if bias_strength == "極端" else 1.08)
    nerf = 0.97 if bias_strength == "弱" else (0.80 if bias_strength == "極端" else 0.92)
    half_boost = 1.0 + (boost - 1.0) * 0.5
    half_nerf = 1.0 - (1.0 - nerf) * 0.5

    if track_bias == "イン前有利 (1〜4枠・先行)":
        if waku <= 4 and kyaku <= 2: bias_mult = boost
        elif (waku <= 4 and kyaku == 3) or (waku == 5 and kyaku <= 2): bias_mult = half_boost
        elif waku >= 6 and kyaku >= 3: bias_mult = nerf
        elif waku >= 6 or kyaku == 4: bias_mult = half_nerf
    elif track_bias == "外差し有利 (5〜8枠・差し追込)":
        if waku >= 5 and kyaku >= 3: bias_mult = boost
        elif (waku >= 5 and kyaku == 2) or (waku <= 4 and kyaku >= 3): bias_mult = half_boost
        elif waku <= 3 and kyaku <= 2: bias_mult = nerf
        elif waku <= 4 and kyaku == 1: bias_mult = half_nerf
    elif track_bias == "前残り (逃げ・先行有利)":
        bias_mult = boost if kyaku <= 2 else nerf
    elif track_bias == "差し有利 (外伸び)":
        bias_mult = boost if kyaku >= 3 else nerf
    elif track_bias == "内枠有利 (インベタ)":
        bias_mult = boost if waku <= 3 else (half_boost if waku == 4 else (nerf if waku >= 6 else 1.0))
    elif track_bias == "外枠有利":
        bias_mult = boost if waku >= 6 else (half_boost if waku == 5 else (nerf if waku <= 3 else 1.0))

    if is_elite_jiriki and bias_mult < 1.0:
        bias_mult = 1.0 - (1.0 - bias_mult) * 0.42
        if total_horses >= 15 and waku >= 6 and kyaku >= 3:
            bias_mult = max(bias_mult, 0.985)

    condition_mult = 1.00
    try:
        condition_mult = float(horse.condition_override)
    except:
        pass

    combined_cond_mult = condition_mult * rotation_mult * training_mult * kinryou_mult
    total_mult = (
        pace_mult
        * bias_mult
        * time_limit_mult
        * combined_cond_mult
        * stage2_ped_mult
        * tactical_mult
        * gear_mult
        * jockey_mult
    )
    capped_mult = max(0.76, min(1.28, total_mult))

    return base_ability * capped_mult, round(bias_mult, 3), round(pace_mult, 3), round(combined_cond_mult, 3)


def build_betting_recommendations(horses_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not horses_data:
        return {}

    by_stage2 = sorted(horses_data, key=lambda x: (x["shisu"], x["myoumi"]), reverse=True)
    by_stage1 = sorted(horses_data, key=lambda x: x["stage1_shisu"], reverse=True)

    axis_horse = None
    candidates = [
        h for h in by_stage2
        if h.get("time_limit_mult", 1.0) >= 0.96 and h.get("training_rank", "B") != "C"
    ]
    if len(candidates) >= 2:
        top1, top2 = candidates[0], candidates[1]
        if (
            top1.get("box_in_risk_level") == "HIGH_BOX_RISK"
            and top2.get("box_in_risk_level") != "HIGH_BOX_RISK"
            and (top1["shisu"] - top2["shisu"]) <= 1.2
        ):
            axis_horse = top2
        else:
            axis_horse = top1
    elif candidates:
        axis_horse = candidates[0]
    else:
        axis_horse = by_stage2[0]

    axis_umaban = axis_horse["umaban"]

    main_2nd_horses = []
    for h in by_stage2:
        u = h["umaban"]
        if u == axis_umaban:
            continue
        if h.get("condition_mult", 1.0) <= 0.88 or h.get("training_rank", "B") == "C":
            continue
        if len(main_2nd_horses) < 3 and (h["shisu"] >= 53.0 or (h["myoumi"] >= 57.0 and h["shisu"] >= 50.0)):
            main_2nd_horses.append(h)

    if len(main_2nd_horses) < 2:
        for h in by_stage2:
            u = h["umaban"]
            if u != axis_umaban and h not in main_2nd_horses and h.get("condition_mult", 1.0) > 0.88:
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
            h["rescue_type"] = "time_warn" if t_mult <= 0.95 else "bias_victim"
        elif (h["stage1_rank"] <= 6 or h["stage1_shisu"] >= 53.0) and t_mult >= 0.94:
            saver_3rd_horses.append(h)
            h["is_bias_victim_rescued"] = True
            h["rescue_type"] = "bias_victim"

        if len(saver_3rd_horses) >= 3:
            break

    for h in by_stage2:
        u = h["umaban"]
        if u == axis_umaban or u in main_2nd_umabans:
            continue
        has_special_badge = bool(h.get("lap_gear_label")) or any(
            "前残り" in b or "道悪" in b or "脚抜き" in b for b in h.get("tactical_badges", [])
        )
        if (
            h not in saver_3rd_horses
            and (h["shisu"] >= 52.0 or h["myoumi"] >= 57.0 or (has_special_badge and h["shisu"] >= 49.5))
            and h.get("condition_mult", 1.0) > 0.88
            and len(saver_3rd_horses) < 5
        ):
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
        has_gear = bool(h.get("lap_gear_label"))

        if t_mult <= 0.93 and h["stage1_shisu"] < 49.0 and not has_gear and h["myoumi"] < 55.0:
            eliminated_umabans.add(u)
            h["role_code"] = "ELIMINATED"
            h["role_label"] = "💀 完全消し (地力下位×時計不足)"
        elif c_mult <= 0.88:
            eliminated_umabans.add(u)
            h["role_code"] = "ELIMINATED"
            h["role_label"] = "💀 完全消し (極端な馬体減/状態難)"

    for h in horses_data:
        u = h["umaban"]
        is_jiriki_top = (h["stage1_rank"] <= 4) or (h["stage1_shisu"] >= 55.0)
        t_mult = h.get("time_limit_mult", 1.0)
        h["is_jiriki_time_warn"] = bool(is_jiriki_top and t_mult <= 0.95)
        box_risk = h.get("box_in_risk_level", "NONE")

        if u == axis_umaban:
            h["role_code"] = "AXIS_1ST"
            if box_risk == "PROTECTED_STRONG":
                h["role_label"] = "👑 1着軸 (不動の1強・包まれ克服圏)"
            elif box_risk in ["HIGH_BOX_RISK", "MOD_BOX_RISK"]:
                h["role_label"] = "👑 1着軸 (能力1位・⚠️内枠包まれ注意)"
            else:
                h["role_label"] = "👑 1着軸 (不動の本命)"
        elif u in main_2nd_umabans:
            h["role_code"] = "MAIN_2ND"
            if box_risk in ["HIGH_BOX_RISK", "MOD_BOX_RISK"]:
                h["role_label"] = "🚀 2列目本線 (⚠️内枠包まれ注意)"
            elif h["myoumi"] >= 58.0 and h["odds"] >= 8.0:
                h["role_label"] = "🚀 2列目本線 (バイアス・ギア特注穴馬)"
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
                h["role_label"] = "🛡️ 3列目押さえ (道悪・前残り・ギア特注)"
        elif u not in eliminated_umabans:
            h["role_code"] = "OTHER"
            h["role_label"] = "静観 (指数下位)"

    row1 = [axis_umaban]
    row2 = main_2nd_umabans
    row3 = main_2nd_umabans + saver_3rd_umabans

    cands_for_val = [h for h in main_2nd_horses if h["odds"] >= 6.0 and h["myoumi"] >= 55.0]
    value_pick = sorted(cands_for_val, key=lambda x: x["myoumi"], reverse=True)[0] if cands_for_val else None
    tansho_list = [{"umaban": axis_umaban, "bamei": axis_horse["bamei"], "type": "本命単勝", "odds": axis_horse["odds"]}]
    if value_pick and value_pick["umaban"] != axis_umaban:
        tansho_list.append({"umaban": value_pick["umaban"], "bamei": value_pick["bamei"], "type": "妙味穴単勝", "odds": value_pick["odds"]})

    fukusho_list = [{"umaban": axis_umaban, "bamei": axis_horse["bamei"], "type": "鉄板軸", "odds": axis_horse["odds"]}]
    for mh in sorted(main_2nd_horses, key=lambda x: x["myoumi"], reverse=True)[:2]:
        fukusho_list.append({"umaban": mh["umaban"], "bamei": mh["bamei"], "type": "妙味複勝", "odds": mh["odds"]})

    wide_combos = [f"{axis_umaban} - {u}" for u in row2]
    sanrenpuku_nagashi_combos = [
        f"{axis_umaban} - { sorted([row3[i], row3[j]])[0] } - { sorted([row3[i], row3[j]])[1] }"
        for i in range(len(row3)) for j in range(i + 1, len(row3))
    ]

    sanrenpuku_form_set = {
        tuple(sorted([a, b, c]))
        for a in row1 for b in row2 for c in row3 if len({a, b, c}) == 3
    }
    sanrenpuku_form_combos = [f"{t[0]} - {t[1]} - {t[2]}" for t in sorted(sanrenpuku_form_set)]

    sanrentan_nagashi_combos = [f"{axis_umaban} → {b} → {c}" for b in row3 for c in row3 if b != c]
    sanrentan_form_combos = [f"{a} → {b} → {c}" for a in row1 for b in row2 for c in row3 if a != b and b != c and a != c]

    eliminated_details = [
        {"umaban": h["umaban"], "bamei": h["bamei"], "reason": h["role_label"], "odds": h["odds"]}
        for h in by_stage2 if h["umaban"] in eliminated_umabans
    ]

    return {
        "formation_groups": {
            "axis_1st": [{"umaban": axis_horse["umaban"], "bamei": axis_horse["bamei"], "shisu": axis_horse["shisu"], "stage1_shisu": axis_horse["stage1_shisu"], "odds": axis_horse["odds"]}],
            "main_2nd": [{"umaban": h["umaban"], "bamei": h["bamei"], "shisu": h["shisu"], "stage1_shisu": h["stage1_shisu"], "myoumi": h["myoumi"], "odds": h["odds"]} for h in main_2nd_horses],
            "saver_3rd": [{
                "umaban": h["umaban"], "bamei": h["bamei"], "shisu": h["shisu"],
                "stage1_shisu": h["stage1_shisu"], "stage1_rank": h["stage1_rank"], "odds": h["odds"],
                "rescued": h.get("is_bias_victim_rescued", False), "rescue_type": h.get("rescue_type", "normal"),
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


def build_win5_race_profile(horses_data: List[Dict[str, Any]], betting_tickets: Dict[str, Any]) -> Dict[str, Any]:
    """
    WIN5最適化エンジン用に、当該レースの『波乱度レベル（BANKER / STANDARD / CHAOS）』と
    『WIN5で優先して拾うべき順番（最大6頭：本命➔対抗➔単穴➔地力1位救済➔特注穴）』を構築する
    """
    if not horses_data:
        return {}

    by_stage2 = sorted(horses_data, key=lambda x: x["shisu"], reverse=True)
    by_stage1 = sorted(horses_data, key=lambda x: x["stage1_shisu"], reverse=True)

    axis_list = betting_tickets.get("formation_groups", {}).get("axis_1st", [])
    axis_u = axis_list[0]["umaban"] if axis_list else by_stage2[0]["umaban"]
    axis_h = next((h for h in horses_data if h["umaban"] == axis_u), by_stage2[0])
    second_h = next((h for h in by_stage2 if h["umaban"] != axis_u), by_stage2[1] if len(by_stage2) > 1 else axis_h)

    shisu_gap = round(axis_h["shisu"] - second_h["shisu"], 1)
    has_box_risk = axis_h.get("box_in_risk_level") in ["HIGH_BOX_RISK", "MOD_BOX_RISK"]

    if shisu_gap >= 3.2 and axis_h["stage1_rank"] == 1 and not has_box_risk:
        difficulty = "BANKER"
        badge_label = "👑 1強バンク候補（推奨1〜2頭）"
        rec_count = 1
    elif shisu_gap < 1.5 or has_box_risk or axis_h["stage1_rank"] >= 3:
        difficulty = "CHAOS"
        badge_label = "💣 波乱・広め推奨（推奨3〜5頭）"
        rec_count = 4
    else:
        difficulty = "STANDARD"
        badge_label = "⚡ 標準・上位決着（推奨2〜3頭）"
        rec_count = 2

    ordered_umabans: List[int] = [axis_u]
    for m in betting_tickets.get("formation_groups", {}).get("main_2nd", []):
        if m["umaban"] not in ordered_umabans:
            ordered_umabans.append(m["umaban"])

    # Stage1（素の地力）1位・2位がまだ入っていなければ優先追加（ダノンデサイル型）
    for s1_h in by_stage1[:2]:
        if s1_h["umaban"] not in ordered_umabans:
            ordered_umabans.append(s1_h["umaban"])

    # 3列目の妙味・道悪・前残り特注馬を追加
    for s3 in betting_tickets.get("formation_groups", {}).get("saver_3rd", []):
        if s3["umaban"] not in ordered_umabans:
            ordered_umabans.append(s3["umaban"])

    for h in by_stage2:
        if h["umaban"] not in ordered_umabans:
            ordered_umabans.append(h["umaban"])

    candidates = []
    for rank_idx, u in enumerate(ordered_umabans[:6]):
        h_obj = next((x for x in horses_data if x["umaban"] == u), None)
        if not h_obj:
            continue
        candidates.append({
            "priority": rank_idx + 1,
            "umaban": h_obj["umaban"],
            "bamei": h_obj["bamei"],
            "shisu": h_obj["shisu"],
            "stage1_rank": h_obj["stage1_rank"],
            "myoumi": h_obj["myoumi"],
            "odds": h_obj["odds"],
            "role_label": h_obj.get("role_label", ""),
            "badges": h_obj.get("tactical_badges", [])[:2]
        })

    return {
        "difficulty": difficulty,
        "badge_label": badge_label,
        "shisu_gap": shisu_gap,
        "recommended_count": rec_count,
        "candidates": candidates
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
    if request.course_type == "芝" and (request.track_condition in ["稍", "重", "不"] or request.cushion_value <= 8.9):
        avg_req_up += 0.8 if request.track_condition in ["重", "不"] else 0.4
    elif request.course_type == "ダ" and request.track_condition in ["稍", "重", "不"]:
        avg_req_up -= 0.4

    place_dict = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
    total_horses_cnt = len(request.horses)

    valid_odds_h = [h for h in request.horses if h.odds > 1.0]
    fav5 = sorted(valid_odds_h, key=lambda x: x.odds)[:5] if len(valid_odds_h) >= 5 else request.horses[:5]
    fav_closer_skew = sum(1 for h in fav5 if int(round(h.kyakushitsu)) >= 3) >= 4

    temp_items = []
    all_horses_early_info = []

    for horse in request.horses:
        h_stats = horse_history_db.get(horse.bamei, {})
        zenso, real_agari, kaisuu, power_score, makuri_flag = get_horse_real_stats(horse.bamei)
        early_3f_sec = estimate_horse_early_3f_speed(horse.bamei)
        effective_zenso_for_lgb = min(zenso, 5.0) if power_score >= 0.77 else zenso

        all_horses_early_info.append({
            "umaban": int(horse.umaban),
            "wakuban": int(horse.wakuban),
            "kyakushitsu": int(round(horse.kyakushitsu)),
            "early_3f": early_3f_sec
        })

        course_up_times = h_stats.get('course_up_times', {}).get(course_key, [])
        course_up = float(np.mean(course_up_times)) if course_up_times else real_agari
        time_diff_base = course_up - base_req_up
        time_diff = course_up - avg_req_up
        kishu_enc = safe_encode('kishu', horse.kishu)

        time_limit_mult = get_time_limit_multiplier(
            horse.bamei, request.course_type, request.distance,
            expected_standard_time, request.cushion_value, request.track_condition
        )
        s1_ped_mult, s2_ped_mult, ped_label = evaluate_pedigree_and_distance_aptitude(
            horse.bamei, request.course_type, request.distance, request.track_condition, request.cushion_value
        )

        rot_mult = float(horse.rotation_mult) if horse.rotation_mult else 1.00
        rot_label = str(horse.rotation_label or "")
        if rot_mult == 1.00 and not rot_label:
            rot_mult, rot_label = evaluate_rotation_aptitude(horse.bamei, None, power_score, request.distance)

        gear_mult, gear_label = evaluate_lap_gear_aptitude(
            horse.bamei, request.course_type, request.distance,
            request.cushion_value, request.track_condition, request.expected_pace
        )
        jockey_mult, jockey_label = evaluate_jockey_stage_synergy(
            horse.kishu, int(horse.wakuban), int(round(horse.kyakushitsu)),
            request.distance, request.cushion_value, makuri_flag, track_cond=request.track_condition
        )

        t_rank = str(horse.training_rank or "B").upper()
        training_mult = TRAINING_RANK_MULTIPLIERS.get(t_rank, 1.00)

        stage1_input = pd.DataFrame([{
            '枠番': 4.0, '馬番': 8.0, '斤量': horse.kinryou, '馬体重_数値': 480.0,
            'distance': request.distance, 'course_type_enc': course_enc, 'weather_enc': weather_enc,
            'track_condition_enc': cond_enc, '舞台_平均前半ペース': base_pace_f, '舞台_平均後半ペース': base_pace_l,
            '舞台_平均要求上がり': base_req_up, '過去基本脚質': horse.kyakushitsu, '前走着順': effective_zenso_for_lgb,
            '近3走_平均上がり': real_agari, '同舞台_平均上がり': course_up, '出走回数': kaisuu,
            '上がりタイム差分': time_diff_base, 'kishu_enc': kishu_enc
        }])
        raw_pred_stage1 = float(model.predict(stage1_input)[0])
        stage1_prob = calculate_stage1_pure_score(
            raw_pred_stage1, horse, power_score, time_diff_base, s1_ped_mult
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

        temp_items.append({
            "horse": horse, "raw_pred": raw_pred, "stage1_prob": stage1_prob,
            "power_score": power_score, "makuri_flag": makuri_flag, "early_3f_sec": early_3f_sec,
            "h_pci": h_pci, "h_pci_type": h_pci_type, "time_diff": time_diff,
            "time_limit_mult": time_limit_mult, "s2_ped_mult": s2_ped_mult, "ped_label": ped_label,
            "rot_mult": rot_mult, "rot_label": rot_label, "gear_mult": gear_mult, "gear_label": gear_label,
            "jockey_mult": jockey_mult, "jockey_label": jockey_label, "t_rank": t_rank, "training_mult": training_mult
        })

    s1_probs = [item["stage1_prob"] for item in temp_items]
    s1_mean = np.mean(s1_probs) if s1_probs else 0
    s1_std = np.std(s1_probs) if s1_probs else 1
    for item in temp_items:
        s1_val = ((item["stage1_prob"] - s1_mean) / s1_std) * 10 + 50 if s1_std > 0 else 50.0
        item["stage1_shisu"] = round(float(s1_val), 1)

    by_s1 = sorted(temp_items, key=lambda x: x["stage1_shisu"], reverse=True)
    for idx, item in enumerate(by_s1):
        item["stage1_rank"] = idx + 1

    stage1_gap_to_2nd = round(by_s1[0]["stage1_shisu"] - by_s1[1]["stage1_shisu"], 1) if len(by_s1) >= 2 else 0.0

    for item in temp_items:
        horse = item["horse"]
        tactical_mult, tactical_badges, box_risk_level = evaluate_box_in_and_lead_risk(
            horse.wakuban, horse.umaban, horse.kyakushitsu, horse.odds, horse.kinryou,
            item["stage1_rank"], stage1_gap_to_2nd, total_horses_cnt, all_horses_early_info,
            request.course_type, request.distance, request.cushion_value,
            track_cond=request.track_condition, fav_closer_skew=fav_closer_skew
        )

        final_score, bias_mult, pace_mult, cond_mult = calculate_comprehensive_score(
            item["raw_pred"], horse, item["power_score"], item["makuri_flag"],
            item["h_pci"], item["h_pci_type"], item["time_diff"],
            request.expected_pace, request.track_bias, request.bias_strength, request.cushion_value,
            base_rpci, item["time_limit_mult"], item["s2_ped_mult"],
            item["rot_mult"], item["training_mult"],
            stage1_rank=item["stage1_rank"], total_horses=total_horses_cnt,
            track_condition=request.track_condition, course_type=request.course_type,
            tactical_mult=tactical_mult, gear_mult=item["gear_mult"],
            jockey_mult=item["jockey_mult"], distance=request.distance
        )

        all_badges: List[str] = list(tactical_badges)
        if item["gear_label"]: all_badges.append(item["gear_label"])
        if item["jockey_label"]: all_badges.append(item["jockey_label"])
        if "☔" in item["ped_label"] and not any("☔" in b for b in all_badges):
            all_badges.append("☔芝道悪血統◎")
        if "🌊" in item["ped_label"] and not any("🌊" in b for b in all_badges):
            all_badges.append("🌊湿ダート血統◎")

        display_rot_label = (f"{' '.join(tactical_badges)} / {item['rot_label']}" if item["rot_label"] else " ".join(tactical_badges)) if tactical_badges else item["rot_label"]
        extra_tags = [x for x in [item["gear_label"], item["jockey_label"]] if x]
        display_ped_label = (f"{item['ped_label']} [{' / '.join(extra_tags)}]" if item["ped_label"] else " / ".join(extra_tags)) if extra_tags else item["ped_label"]

        history_details = build_horse_history_details(horse.bamei, place_dict)

        horses_data.append({
            "wakuban": int(horse.wakuban),
            "umaban": int(horse.umaban), "bamei": horse.bamei, "odds": float(horse.odds),
            "stage1_prob": item["stage1_prob"], "stage1_shisu": item["stage1_shisu"], "stage1_rank": item["stage1_rank"],
            "base_prob": final_score, "time_diff": round(item["time_diff"], 2),
            "pci": item["h_pci"], "pci_type": item["h_pci_type"], "early_3f_sec": item["early_3f_sec"],
            "bataijuu": horse.bataijuu, "weight_change": horse.weight_change,
            "training_rank": item["t_rank"], "rotation_label": display_rot_label, "pedigree_label": display_ped_label,
            "tactical_badges": all_badges, "box_in_risk_level": box_risk_level, "lap_gear_label": item["gear_label"],
            "time_limit_mult": round(item["time_limit_mult"], 2),
            "bias_mult": bias_mult, "pace_mult": pace_mult, "condition_mult": cond_mult, "tactical_mult": tactical_mult,
            "power_score": round(item["power_score"], 2), "history_details": history_details
        })

    probs = [h["base_prob"] for h in horses_data]
    mean_prob = np.mean(probs) if probs else 0
    std_prob = np.std(probs) if probs else 1
    total_prob = sum(probs) if sum(probs) > 0 else 1.0

    for h in horses_data:
        shisu = ((h["base_prob"] - mean_prob) / std_prob) * 10 + 50 if std_prob > 0 else 50.0
        h["shisu"] = round(float(shisu), 1)
        h["shisu_diff"] = round(h["shisu"] - h["stage1_shisu"], 1)
        h["norm_win_prob"] = round(float(h["base_prob"] / total_prob), 4)

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
        m_mean = np.mean(myoumi_values)
        m_std = np.std(myoumi_values)
        for h in horses_data:
            if h["odds"] > 0:
                m_shisu = ((h["raw_ev"] - m_mean) / m_std) * 10 + 50 if m_std > 0 else 50.0
                h["myoumi"] = round(float(m_shisu), 1)
            else:
                h["myoumi"] = 0.0
    else:
        for h in horses_data:
            h["myoumi"] = 0.0

    horses_data.sort(key=lambda x: x["shisu"], reverse=True)
    for idx, h in enumerate(horses_data):
        h["stage2_rank"] = idx + 1

    betting_tickets = build_betting_recommendations(horses_data)
    win5_profile = build_win5_race_profile(horses_data, betting_tickets)

    return {
        "status": "success",
        "diagnosis": horses_data,
        "betting_tickets": betting_tickets,
        "win5_profile": win5_profile,
        "req_up": round(avg_req_up, 2),
        "pace_f": round(avg_pace_f, 1),
        "pace_l": round(avg_pace_l, 1),
        "rpci": base_rpci,
        "rpci_type": base_rpci_type
    }


@app.post("/simulate_race")
def simulate_race(request: RaceRequest):
    diag_res = diagnose_race(request)
    diagnosed = diag_res.get("diagnosis", [])

    sim_scores = []
    for h in diagnosed:
        target_index = (h["shisu"] * 0.85) + (h.get("myoumi", 50.0) * 0.15)
        noise = np.random.normal(0, 3.2)
        sim_scores.append({
            "umaban": h["umaban"],
            "bamei": h["bamei"],
            "shisu": h["shisu"],
            "time_diff": h["time_diff"],
            "score": target_index + noise
        })

    sim_scores.sort(key=lambda x: x["score"], reverse=True)
    results = [
        {
            "umaban": sim["umaban"],
            "bamei": sim["bamei"],
            "shisu": sim["shisu"],
            "time_diff": sim["time_diff"],
            "rank": i + 1
        }
        for i, sim in enumerate(sim_scores)
    ]

    return {
        "status": "success",
        "ranking": results,
        "req_up": diag_res.get("req_up", 35.0),
        "pace_f": diag_res.get("pace_f", 35.0),
        "pace_l": diag_res.get("pace_l", 35.0)
    }
