# data/build_dataset.py
# -*- coding: utf-8 -*-
"""
자연어 입력 다양화 + 새 PTRN 스펙(거주:0, 직장:1, 방문:2; 생활인구/미언급 → [0,1,2])
- 학습 데이터: {"input": str, "output": {...}} 형태의 JSONL 생성
- region_cd는 사용하지 않음. 출력은 표준명 region_nm만 포함.
- '출퇴근' 표현은 향후 이동목적과의 충돌을 피하기 위해 제외.
"""

import json
import random
import re
import argparse
from typing import List, Dict, Optional, Tuple

from mapping.lexicon import ALL_AGES, GENDER_TERMS, PTRN_MAP
from mapping.lexicon_region import REGIONS  # [(region_nm, code, [aliases...]), ...]

# -----------------------------
# 설정(확률/분포)
# -----------------------------
PROB_OMIT_PTRN: float = 0.25   # 패턴 문구를 문장에 '미언급'으로 둘 비율(라벨은 [0,1,2])
PROB_USE_ADV_PTRN: float = 0.25  # 패턴 명시 시, ADV(이유/용도형) 표현을 사용할 확률

random.seed(7)

# -----------------------------
# 유틸
# -----------------------------
def choice(x):
    return random.choice(x)

def surface_month(month: str) -> str:
    m = month.strip()
    if m == "이번 달":
        return choice(["이번 달", "이달", "금월"])
    if m == "지난 달":
        return choice(["지난 달", "전월", "저번 달"])
    g = re.match(r'(\d{4})\s*년\s*(\d{1,2})\s*월', m)
    if g:
        y, mm = g.group(1), int(g.group(2))
        return choice([f"{y}년 {mm}월", f"{y}.{mm}", f"{y}년 {mm}월 기준"])
    return m

NEAR_SUFFIX = ["", "", "", " 일대", " 근처", " 인근", " 부근"]

def pick_region_alias(region_nm: str, aliases: List[str]) -> str:
    cand = (aliases or []) + [region_nm]
    alias = choice(cand)
    return alias + choice(NEAR_SUFFIX)

def surface_gender(g: str) -> str:
    table = {
        "여성": ["여성", "여자", "여"],
        "남성": ["남성", "남자", "남"],
        "전체": ["남녀 모두", "모든 성별", "전체"],
    }
    return choice(table.get(g, [g]))

def surface_age(age_key: str) -> str:
    k = age_key.replace(" ", "")
    m_multi = re.findall(r'(\d{2})', k)
    if m_multi and "대" in k and len(m_multi) >= 2:
        nums = m_multi
        return choice([
            ",".join(nums) + "대",
            "·".join(nums) + "대",
            f"{nums[0]}~{nums[-1]}대",
            "".join(nums) + "대",   # 3040대
        ])
    m_single = re.match(r'(\d{2})대', k)
    if m_single:
        return f"{m_single.group(1)}대"
    mapper = {
        "전연령": ["전연령", "전 연령대", "모든 연령대"],
        "모든연령": ["모든 연령대", "전 연령"],
    }
    if k in mapper:
        return choice(mapper[k])
    return age_key

ASK_TAIL = [
    "알려줘", "보여줘", "보고 싶어", "데이터 줘", "정리해줘", "조회해줘",
    "볼 수 있을까?", "가능할까?", "좀 알려줘", "부탁해", "요?", "", ""
]
FILLERS = ["", "", "", "", "데이터", "현황", "통계", "규모", "수치"]

# -----------------------------
# PTRN 자연어 표현 (출퇴근 제외)
# -----------------------------
def surface_ptrn(ptrn_key: str, slot: str = "NP") -> str:
    """
    ptrn_key: '직장' | '방문' | '거주' | '생활인구'
    slot    : 'NP'(명사구, 기본) | 'ADV'(이유/용도형)
    - '출퇴근' 표현은 제외합니다.
    - 템플릿이 보통 '{region} {ptrn} {age} {gender} ...' 형태라 기본은 명사구가 안전합니다.
    """

    # 명사구(기본): 문장 내 삽입 용이
    table_np = {
        "직장": [
            "직장", "직장 목적", "직장 인구", "근로자", "근무 목적",
            "회사 관련", "업무 목적", "근무 인구", "회사 인력", "직장 수요"
        ],
        "방문": [
            "방문", "방문자", "유입", "유입 인구", "외부 유입",
            "일시 방문", "관광 방문", "방문 규모", "외지 방문"
        ],
        "거주": [
            "거주", "거주 인구", "상주 인구", "정주 인구",
            "주소지 거주", "해당 지역 거주", "지역 거주"
        ],
        "생활인구": [
            "생활인구", "생활 인구", "전체 생활인구"
        ],
    }

    # 이유/용도형(부사어): 일부 템플릿에서만 사용
    table_adv = {
        "직장": [
            "일 때문에", "일 보러", "업무 보러", "회사 때문에", "근무 때문에", "회사 업무로"
        ],
        "방문": [
            "방문하러", "놀러", "관광하러", "들르러", "잠깐 들르려고", "일시적으로 방문하려고"
        ],
        "거주": [
            "살아서", "거주해서", "머물러서"
        ],
        "생활인구": [
            "생활인구 기준으로"  # 과한 변형은 피함
        ],
    }

    def decorate_noun(core: str) -> str:
        """명사구 꼬리표/접두어 확률적 부여"""
        if core in ("생활인구", "생활 인구", "전체 생활인구"):
            tails = ["", "", " 데이터", " 현황", " 통계"]
        else:
            tails = ["", "", "", " 인구", " 규모", " 데이터", " 현황", " 통계", " 수치"]
        prefixes = ["", "", "", "관련 ", "해당 ", "지역 "]
        return choice(prefixes) + core + choice(tails)

    key = ptrn_key.strip()
    if key not in table_np:
        return key

    if slot.upper() == "ADV":
        cand = table_adv.get(key, [])
        if cand:
            return choice(cand)

    core = choice(table_np[key])
    return decorate_noun(core)

# -----------------------------
# 템플릿
# -----------------------------
TEMPLATES_WITH_PTRN_NP = [
    "{month} {region} {age} {gender} {ptrn}{fill} {ask}",
    "{region} {month} {ptrn} {age} {gender}{fill} {ask}",
    "{month} 기준 {region} {ptrn} {age} {gender}{fill} {ask}",
    "{region} {age} {gender} {ptrn} {month}{fill} {ask}",
]
TEMPLATES_WITH_PTRN_ADV = [
    "{month} {region} {age} {gender}{fill} {ptrn_adv} {ask}",
    "{region} {month} {age} {gender}{fill} {ptrn_adv} {ask}",
]
TEMPLATES_NO_PTRN = [
    "{month} {region} {age} {gender}{fill} {ask}",
    "{region} {month} {age} {gender}{fill} {ask}",
    "{month} 기준 {region} {age} {gender}{fill} {ask}",
]

# -----------------------------
# 월 → YYYYMM
# -----------------------------
def month_to_ym(month: str) -> int:
    table = {
        "2025년 1월": 202501, "2025년 2월": 202502, "2025년 3월": 202503, "2025년 4월": 202504,
        "2025년 5월": 202505, "2025년 6월": 202506, "2025년 7월": 202507, "2025년 8월": 202508,
        "2025년 9월": 202509, "2025년 10월": 202510, "2025년 11월": 202511, "2025년 12월": 202512,
        "이번 달": 202507, "지난 달": 202506,  # 필요 시 현재 날짜 기준으로 동적 계산
    }
    return table.get(month.strip(), 202507)

# -----------------------------
# 입력 문장 생성
# -----------------------------
MONTHS = [
    "2025년 1월","2025년 2월","2025년 3월","2025년 4월",
    "2025년 5월","2025년 6월","2025년 7월","2025년 8월",
    "2025년 9월","2025년 10월","2025년 11월","2025년 12월",
    "이번 달","지난 달",
]
GENDER_KEYS = list(GENDER_TERMS.keys())  # ["여성","남성","전체",...]
AGE_KEYS = list(ALL_AGES.keys())         # ["30대","30,40대","전연령",...]

def make_input(month_key: str,
               region_nm: str,
               aliases: List[str],
               gender_key: str,
               age_key: str,
               ptrn_label_or_none: Optional[str]) -> str:
    month_txt  = surface_month(month_key)
    region_txt = pick_region_alias(region_nm, aliases)
    gender_txt = surface_gender(gender_key)
    age_txt    = surface_age(age_key)

    # 패턴 문구 구성
    if ptrn_label_or_none is None:
        # 미언급: 패턴 단어 자체를 텍스트에 넣지 않음
        tmpl = choice(TEMPLATES_NO_PTRN)
        s = tmpl.format(
            month=month_txt, region=region_txt, age=age_txt, gender=gender_txt,
            fill=" " + choice(FILLERS) if random.random() < 0.7 else "",
            ask=" " + choice(ASK_TAIL) if random.random() < 0.8 else "",
        )
    else:
        # 패턴 명시: ADV 사용할지 결정
        use_adv = (random.random() < PROB_USE_ADV_PTRN) and (ptrn_label_or_none in ("직장","방문","거주"))
        if use_adv:
            ptrn_adv = surface_ptrn(ptrn_label_or_none, slot="ADV")
            tmpl = choice(TEMPLATES_WITH_PTRN_ADV)
            s = tmpl.format(
                month=month_txt, region=region_txt, age=age_txt, gender=gender_txt,
                ptrn_adv=ptrn_adv,
                fill=" " + choice(FILLERS) if random.random() < 0.5 else "",
                ask=" " + choice(ASK_TAIL) if random.random() < 0.8 else "",
            )
        else:
            # 생활인구는 보수적으로
            if ptrn_label_or_none == "생활인구":
                ptrn_txt = "생활인구"
            else:
                ptrn_txt = surface_ptrn(ptrn_label_or_none, slot="NP")
            tmpl = choice(TEMPLATES_WITH_PTRN_NP)
            s = tmpl.format(
                month=month_txt, region=region_txt, age=age_txt, gender=gender_txt,
                ptrn=ptrn_txt,
                fill=" " + choice(FILLERS) if random.random() < 0.7 else "",
                ask=" " + choice(ASK_TAIL) if random.random() < 0.9 else "",
            )

    # 공백/구두점 정리
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ,", ",")
    return s

# -----------------------------
# 출력(라벨) 생성
# -----------------------------
def to_output(month_key: str,
              region_nm: str,
              gender_cd: List[int],
              ages: List[int],
              ptrn_label_or_none: Optional[str]) -> Dict:
    """
    새 스펙:
      - 거주:0, 직장:1, 방문:2
      - 생활인구 or 미언급: [0,1,2]
    """
    ym = month_to_ym(month_key)

    if (ptrn_label_or_none is None) or (ptrn_label_or_none == "생활인구"):
        ptrn_codes = PTRN_MAP["생활인구"]          # [0,1,2]
    else:
        code = PTRN_MAP[ptrn_label_or_none]        # 0 or 1 or 2
        ptrn_codes = [code] if isinstance(code, int) else code

    return {
        "base_ym": ym,
        "region_nm": region_nm,
        "ptrn": ptrn_codes,
        "gender_cd": gender_cd,
        "age_cd": ages,
    }

# -----------------------------
# 메인
# -----------------------------
def main(out_path: str = "parse_data.jsonl", n: int = 5000, seed: int = 7,
         prob_omit_ptrn: Optional[float] = None, prob_use_adv: Optional[float] = None):
    global PROB_OMIT_PTRN, PROB_USE_ADV_PTRN
    random.seed(seed)
    if prob_omit_ptrn is not None:
        PROB_OMIT_PTRN = float(prob_omit_ptrn)
    if prob_use_adv is not None:
        PROB_USE_ADV_PTRN = float(prob_use_adv)

    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(n):
            region_nm, rcd, aliases = choice(REGIONS)
            month_key  = choice(MONTHS)
            gender_key = choice(GENDER_KEYS)
            age_key    = choice(AGE_KEYS)

            # 패턴 라벨 샘플링: 미언급 / 생활인구 / 거주 / 직장 / 방문
            if random.random() < PROB_OMIT_PTRN:
                ptrn_label = None
            else:
                pool = ["생활인구", "거주", "직장", "방문"]
                ptrn_label = choice(pool)

            # 성별/연령 코드 변환
            gender_cd = [1] if gender_key == "여성" else ([0] if gender_key == "남성" else [0, 1])
            age_cd    = ALL_AGES[age_key]

            text = make_input(month_key, region_nm, aliases, gender_key, age_key, ptrn_label)
            item = {
                "input": text,
                "output": to_output(month_key, region_nm, gender_cd, age_cd, ptrn_label),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {n} lines → {out_path}")
    print(f"[info] PROB_OMIT_PTRN={PROB_OMIT_PTRN}, PROB_USE_ADV_PTRN={PROB_USE_ADV_PTRN}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", type=str, default="parse_data.jsonl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prob_omit_ptrn", type=float, default=None, help="패턴 미언급 비율(기본 0.25)")
    ap.add_argument("--prob_use_adv", type=float, default=None, help="패턴 명시 시 ADV 사용 비율(기본 0.25)")
    args = ap.parse_args()
    main(**vars(args))