# data/build_dataset.py
# -*- coding: utf-8 -*-
"""
자연어 입력 다양화 + 새 PTRN 스펙(거주:0, 직장:1, 방문:2; 생활인구/미언급 → [0,1,2])
+ OpenAI 4o-mini 패러프레이즈/검증(diff)/정확도 집계/최종 데이터셋 저장

출력 JSONL(기본):
  {"input": <최종 문장(패러프레이즈 or 룰베이스)>, "output": {...}}
옵션(--include_meta 사용 시):
  {"input": ..., "output": {...}, "seed_input": <룰베이스>, "checks": {...}, "diff": {...}}
"""

import json
import random
import re
import argparse
import os
import time
from typing import List, Dict, Optional, Tuple

# ---- 기존 맵/리소스 ----
from mapping.lexicon import ALL_AGES, GENDER_TERMS, PTRN_MAP
from mapping.lexicon_region import REGIONS  # [(region_nm, code, [aliases...]), ...]

# =========================
# 기존 룰베이스 생성 파트
# =========================
PROB_OMIT_PTRN: float = 0.25
PROB_USE_ADV_PTRN: float = 0.25
random.seed(7)

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
        "전체": ["남녀 모두", "모든 성별", "남녀"],
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
            "".join(nums) + "대",
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

def surface_ptrn(ptrn_key: str, slot: str = "NP") -> str:
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
    table_adv = {
        "직장": [
            "일 때문에", "일 보러", "업무 보러", "회사 때문에", "근무 때문에", "회사 업무로", "업무 관련"
        ],
        "방문": [
            "방문하러", "놀러", "들르러", "잠깐 들르려고", "일시적으로 방문하려고"
        ],
        "거주": [
            "살아서", "거주해서", "머물러서"
        ],
        "생활인구": [
            "생활인구 기준으로"
        ],
    }
    def decorate_noun(core: str) -> str:
        if core in ("생활인구", "생활 인구", "전체 생활인구"):
            tails = ["", "", " 데이터", " 현황", " 통계"]
        else:
            tails = ["", "", "", " 인구", " 규모", " 데이터", " 현황", " 통계", " 수치"]
        prefixes = ["", "", "", "관련 ", "해당 ", "지역 "]
        return choice(prefixes) + core + choice(tails)
    key = ptrn_key.strip()
    if key not in table_np: return key
    if slot.upper() == "ADV":
        cand = table_adv.get(key, [])
        if cand: return choice(cand)
    core = choice(table_np[key])
    return decorate_noun(core)

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

def month_to_ym(month: str) -> int:
    table = {
        "2025년 1월": 202501, "2025년 2월": 202502, "2025년 3월": 202503, "2025년 4월": 202504,
        "2025년 5월": 202505, "2025년 6월": 202506, "2025년 7월": 202507, "2025년 8월": 202508,
        "2025년 9월": 202509, "2025년 10월": 202510, "2025년 11월": 202511, "2025년 12월": 202512,
        "이번 달": 202507, "지난 달": 202506,
    }
    return table.get(month.strip(), 202507)

MONTHS = [
    "2025년 1월","2025년 2월","2025년 3월","2025년 4월",
    "2025년 5월","2025년 6월","2025년 7월","2025년 8월",
    "2025년 9월","2025년 10월","2025년 11월","2025년 12월",
    "이번 달","지난 달",
]
GENDER_KEYS = list(GENDER_TERMS.keys())
AGE_KEYS = list(ALL_AGES.keys())

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

    if ptrn_label_or_none is None:
        tmpl = choice(TEMPLATES_NO_PTRN)
        s = tmpl.format(
            month=month_txt, region=region_txt, age=age_txt, gender=gender_txt,
            fill=" " + choice(FILLERS) if random.random() < 0.7 else "",
            ask=" " + choice(ASK_TAIL) if random.random() < 0.8 else "",
        )
    else:
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
            ptrn_txt = "생활인구" if ptrn_label_or_none == "생활인구" else surface_ptrn(ptrn_label_or_none, slot="NP")
            tmpl = choice(TEMPLATES_WITH_PTRN_NP)
            s = tmpl.format(
                month=month_txt, region=region_txt, age=age_txt, gender=gender_txt,
                ptrn=ptrn_txt,
                fill=" " + choice(FILLERS) if random.random() < 0.7 else "",
                ask=" " + choice(ASK_TAIL) if random.random() < 0.9 else "",
            )
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ,", ",")
    return s

def to_output(month_key: str,
              region_nm: str,
              gender_cd: List[int],
              ages: List[int],
              ptrn_label_or_none: Optional[str]) -> Dict:
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

# =========================
# 🔗 OpenAI 4o-mini 패러프레이즈
# =========================
def openai_client_from_env(env_key="OPENAI_API_KEY"):
    from openai import OpenAI
    api_key = os.getenv(env_key)
    if not api_key:
        raise RuntimeError(f"{env_key} 환경변수가 필요합니다. export {env_key}=sk-...")
    return OpenAI(api_key=api_key)

PROMPT_HEAD = (
    "너는 데이터 분석 시스템을 사용하기 위한 데이터 조회 명령어를 만드는 전문가야.\n"
    "아래 토큰들을 그대로 포함하되 자연스러운 데이터 조회 명령어를 만들어줘.\n"
    "조회 해줘, 보여줘, 알려줘, 부탁해, 요청해, 요청해줘, 요청해줘요, 요청해줘요. 등 다양한 표현을 적절히 사용할것"
    "출력은 문장 하나만, 추가 설명 금지.\n"
)

def build_required_tokens_for_prompt(output: Dict[str, any], region_aliases: List[str]) -> List[str]:
    # 필수 토큰: 월(표준), 지역(표준명 또는 별칭 하나), 연령(각 'NN대'), 성별(남성/여성/남녀), 목적(거주/직장/방문 또는 미언급)
    tokens = []
    ym = output["base_ym"]; y, m = ym // 100, ym % 100
    tokens.append(f"{y}년 {m}월")
    # 지역: 별칭 중 하나만 강제(너무 엄격하면 생성 실패↑) → 표준명 우선
    region_nm = output["region_nm"]
    if region_aliases:
        # 표준명 포함하여 후보 만들고 하나 선택
        cand = [region_nm] + [a for a in region_aliases if a != region_nm]
        tokens.append(cand[0])  # 표준명 우선
    else:
        tokens.append(region_nm)
    # 연령
    ages = sorted(set(output["age_cd"]))
    tokens.extend([f"{a}대" for a in ages])
    # 성별
    gc = sorted(set(output["gender_cd"]))
    if gc == [0]: tokens.append("남성")
    elif gc == [1]: tokens.append("여성")
    else: tokens.append("남녀")
    # 목적
    s = sorted(set(output.get("ptrn", [])))
    if s == [0]: tokens.append("거주")
    elif s == [1]: tokens.append("직장")
    elif s == [2]: tokens.append("방문")
    # [0,1,2]는 미언급 허용 → 프롬프트에 넣지 않음
    return tokens

def call_openai_4o_mini(client, user_text: str, max_tokens=80, temperature=0.7, top_p=0.9) -> str:
    from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError
    import traceback, time as _t
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":PROMPT_HEAD},
                    {"role":"user","content":user_text},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=1,
            )
            text = (resp.choices[0].message.content or "").strip()
            text = re.sub(r"\s+"," ", text).split("\n")[0].strip().strip('"“”')
            return text
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError):
            if attempt == 2: raise
            _t.sleep(1.2 * (attempt+1))
    return user_text

# =========================
# ✅ 요소별 검증(정규식/동의어)
# =========================
GENDER_SYNS = {
    "남성": [r"남성", r"남자", r"\b남\b"],
    "여성": [r"여성", r"여자", r"\b여\b"],
    "남녀": [r"남녀", r"남녀\s*모두", r"모든\s*성별"],
}
PTRN_SYNS = {
    0: [r"거주", r"상주\s*인구", r"정주\s*인구", r"주소지\s*거주", r"지역\s*거주"],
    1: [r"직장", r"근무\s*목적", r"근로자", r"업무\s*목적", r"회사\s*(관련|인력|수요)?", r"근무\s*인구"],
    2: [r"방문", r"방문자", r"유입(\s*인구)?", r"외부\s*유입", r"일시\s*방문", r"관광\s*방문"],
}

def text_contains_any(text: str, patterns: List[str]) -> bool:
    for p in patterns:
        if re.search(p, text):
            return True
    return False

def month_ok(text: str, base_ym: int) -> bool:
    y, m = base_ym // 100, base_ym % 100
    # 허용 표기: "YYYY년 M월", "YYYY.M", "M월"(연도 생략은 보수적으로 비허용)
    if f"{y}년 {m}월" in text: return True
    if re.search(rf"{y}\s*[\.\-\/]\s*{m}\b", text): return True
    return False

def region_ok(text: str, region_nm: str, region_aliases: List[str]) -> bool:
    cands = [region_nm] + [a for a in region_aliases if a != region_nm]
    return any(c in text for c in cands)

def age_ok(text: str, age_cd: List[int]) -> bool:
    """
    허용 패턴 확장:
      - 단일: 0대, 10대, 20대, ...
      - 범위: 0~50대, 0-50대
      - 말범위: 0대부터 50대까지 / 0대에서 50대까지 / 0대 ... 50대에 이르기까지
      - 묶음: 0,10,20,30,40,50대 / 0·10·20대
      - 붙임: 0102030대 (스캔 규칙: '0'은 1자리, 그 외는 2자리씩)
      - 경계: 0대 이상 / 50대 이하
    """
    import re
    text = re.sub(r"\s+", " ", text)
    needed = sorted(set(age_cd))
    present = set()

    # -----------------------------
    # 1) 직접 "NN대" (NN은 1~2자리 → 0, 10, 20, ...)
    # -----------------------------
    for m in re.finditer(r"(\d{1,2})\s*대", text):
        v = int(m.group(1))
        if v in needed:
            present.add(v)

    # -----------------------------
    # 2) 기호 범위 "NN~MM대", "NN-MM대"
    # -----------------------------
    for s, e in re.findall(r"(\d{1,2})\s*[~\-]\s*(\d{1,2})\s*대", text):
        s, e = int(s), int(e)
        lo, hi = min(s, e), max(s, e)
        for v in range(lo, hi + 1, 10):
            if v in needed:
                present.add(v)

    # -----------------------------
    # 3) 묶음 "0,10,20대" / "0·10·20대" / "0.10.20대"
    # -----------------------------
    for grp in re.findall(r"((?:\d{1,2}\s*[,\.·]\s*)+\d{1,2})\s*대", text):
        nums = [int(x) for x in re.findall(r"\d{1,2}", grp)]
        for v in nums:
            if v in needed:
                present.add(v)

    # -----------------------------
    # 4) 붙임표현 "0102030대"
    #    규칙: 문자열을 좌→우 스캔, '0'은 1자리, 그 외는 2자리 묶음
    # -----------------------------
    for s in re.findall(r"((?:\d{1,2}){2,})\s*대", text):
        i, L = 0, len(s)
        nums = []
        while i < L:
            if s[i] == '0':           # '0' 단독 토큰
                nums.append(0)
                i += 1
            elif i + 1 < L:           # 두 자리 토큰
                nums.append(int(s[i:i+2]))
                i += 2
            else:                     # 끝에 1자리만 남는 경우는 무시
                break
        for v in nums:
            if v in needed:
                present.add(v)

    # -----------------------------
    # 5) 말범위 "NN대부터 MM대까지" / "NN대에서 MM대까지" / "...에 이르기까지"
    # -----------------------------
    for s, e in re.findall(
        r"(\d{1,2})\s*대\s*(?:부터|에서)\s*(\d{1,2})\s*대\s*(?:까지|까지의|에\s*이르기까지)?",
        text
    ):
        s, e = int(s), int(e)
        lo, hi = min(s, e), max(s, e)
        for v in range(lo, hi + 1, 10):
            if v in needed:
                present.add(v)

    # -----------------------------
    # 6) 경계표현 "NN대 이상/이후/부터" (하한만 명시)
    # -----------------------------
    for s in re.findall(r"(\d{1,2})\s*대\s*(?:이상|이후|부터)", text):
        s = int(s)
        for v in needed:
            if v >= s:
                present.add(v)

    # -----------------------------
    # 7) 경계표현 "MM대 이하/까지/미만" (상한만 명시)
    # -----------------------------
    for e in re.findall(r"(\d{1,2})\s*대\s*(?:이하|까지|미만)", text):
        e = int(e)
        for v in needed:
            if v <= e:
                present.add(v)

    return set(needed).issubset(present)

def gender_ok(text: str, gender_cd: List[int]) -> bool:
    s = sorted(set(gender_cd))
    if s == [0]:   # 남성
        return text_contains_any(text, GENDER_SYNS["남성"])
    if s == [1]:   # 여성
        return text_contains_any(text, GENDER_SYNS["여성"])
    # [0,1]
    return (
        text_contains_any(text, GENDER_SYNS["남녀"]) or
        (text_contains_any(text, GENDER_SYNS["남성"]) and text_contains_any(text, GENDER_SYNS["여성"]))
    )

def ptrn_ok(text: str, ptrn_codes: List[int]) -> bool:
    s = sorted(set(ptrn_codes))
    if s == [0,1,2]:
        return True  # 미언급 허용
    code = s[0]
    syns = PTRN_SYNS.get(code, [])
    return text_contains_any(text, syns)

# =========================
# 🚀 MAIN
# =========================
def main(out_path: str = "parse_data.jsonl", n: int = 500, seed: int = 7,
         prob_omit_ptrn: Optional[float] = None, prob_use_adv: Optional[float] = None,
         # OpenAI paraphrase 옵션
         use_openai_paraphrase: bool = False,
         openai_model: str = "gpt-4o-mini",
         openai_sleep_ms: int = 120,
         openai_max_tokens: int = 80,
         openai_temperature: float = 0.7,
         openai_top_p: float = 0.9,
         # 결과/로그
         report_path: Optional[str] = None,
         include_meta: bool = False,
         keep_only_valid: bool = False):
    global PROB_OMIT_PTRN, PROB_USE_ADV_PTRN
    random.seed(seed)
    if prob_omit_ptrn is not None:
        PROB_OMIT_PTRN = float(prob_omit_ptrn)
    if prob_use_adv is not None:
        PROB_USE_ADV_PTRN = float(prob_use_adv)

    # region → aliases 매핑
    region_alias_map: Dict[str, List[str]] = {nm: (aliases or []) for nm, _, aliases in REGIONS}

    # OpenAI 준비
    client = None
    if use_openai_paraphrase:
        client = openai_client_from_env()

    total = 0
    kept  = 0
    # 필드별/전체 통계
    ok_month = ok_region = ok_age = ok_gender = ok_ptrn = all_ok = 0

    if report_path is None:
        base, ext = os.path.splitext(out_path)
        report_path = base + "_openai_report.jsonl"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout, open(report_path, "w", encoding="utf-8") as frep:
        for _ in range(n):
            region_nm, rcd, aliases = choice(REGIONS)
            month_key  = choice(MONTHS)
            gender_key = choice(list(GENDER_TERMS.keys()))
            age_key    = choice(list(ALL_AGES.keys()))

            if random.random() < PROB_OMIT_PTRN:
                ptrn_label = None
            else:
                ptrn_label = choice(["생활인구", "거주", "직장", "방문"])

            gender_cd = [1] if gender_key == "여성" else ([0] if gender_key == "남성" else [0, 1])
            age_cd    = ALL_AGES[age_key]
            output    = to_output(month_key, region_nm, gender_cd, age_cd, ptrn_label)

            seed_input = make_input(month_key, region_nm, aliases, gender_key, age_key, ptrn_label)
            final_input = seed_input

            # LLM 패러프레이즈
            checks = diff = None
            if use_openai_paraphrase:
                # 프롬프트에 필수 토큰 제시
                req_tokens = build_required_tokens_for_prompt(output, region_alias_map.get(region_nm, []))
                prompt = (
                    "너는 데이터 분석 시스템을 사용하기 위한 데이터 조회 명령어를 만드는 전문가야.\n"
                    "아래 토큰들을 그대로 포함하되 자연스러운 데이터 조회 명령어를 만들어줘.\n"
                    "출력은 문장 하나만, 추가 설명 금지.\n" +
                    "\n".join(f"- {t}" for t in req_tokens)
                )
                try:
                    final_input = call_openai_4o_mini(
                        client, prompt,
                        max_tokens=openai_max_tokens,
                        temperature=openai_temperature,
                        top_p=openai_top_p
                    )
                except Exception as e:
                    # 실패 시 seed 유지
                    final_input = seed_input

                # 요소별 검증
                m_ok = month_ok(final_input, output["base_ym"])
                r_ok = region_ok(final_input, output["region_nm"], region_alias_map.get(region_nm, []))
                a_ok = age_ok(final_input, output["age_cd"])
                g_ok = gender_ok(final_input, output["gender_cd"])
                p_ok = ptrn_ok(final_input, output["ptrn"])

                checks = {"month_ok": m_ok, "region_ok": r_ok, "age_ok": a_ok, "gender_ok": g_ok, "ptrn_ok": p_ok}
                missing = [k for k,v in checks.items() if not v]
                diff = {"missing": missing}

                # 통계
                ok_month  += int(m_ok)
                ok_region += int(r_ok)
                ok_age    += int(a_ok)
                ok_gender += int(g_ok)
                ok_ptrn   += int(p_ok)
                all_ok    += int(all(checks.values()))

                # 라인별 리포트 기록
                frep.write(json.dumps({
                    "seed_input": seed_input,
                    "gen_input": final_input,
                    "output": output,
                    "checks": checks,
                    "diff": diff
                }, ensure_ascii=False) + "\n")

                # rate limit 보호
                if openai_sleep_ms > 0:
                    time.sleep(openai_sleep_ms / 1000.0)

            # 최종 데이터셋 저장(필터 옵션)
            if (not keep_only_valid) or (checks is None) or all(checks.values()):
                rec = {"input": final_input, "output": output}
                if include_meta:
                    rec["seed_input"] = seed_input
                    if checks is not None:
                        rec["checks"] = checks
                        rec["diff"] = diff
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

            total += 1

    # 정확도 요약
    print(f"[OK] wrote {kept}/{total} lines → {out_path}")
    print(f"[info] PROB_OMIT_PTRN={PROB_OMIT_PTRN}, PROB_USE_ADV_PTRN={PROB_USE_ADV_PTRN}")
    if use_openai_paraphrase:
        denom = total if total > 0 else 1
        print("==== Paraphrase element coverage ====")
        print(f"month_ok  : {ok_month}/{total} ({ok_month/denom:.2%})")
        print(f"region_ok : {ok_region}/{total} ({ok_region/denom:.2%})")
        print(f"age_ok    : {ok_age}/{total} ({ok_age/denom:.2%})")
        print(f"gender_ok : {ok_gender}/{total} ({ok_gender/denom:.2%})")
        print(f"ptrn_ok   : {ok_ptrn}/{total} ({ok_ptrn/denom:.2%})")
        print(f"ALL_OK    : {all_ok}/{total} ({all_ok/denom:.2%})")
        print(f"[report] per-line diffs → {report_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", type=str, default="parse_data.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prob_omit_ptrn", type=float, default=None, help="패턴 미언급 비율(기본 0.25)")
    ap.add_argument("--prob_use_adv", type=float, default=None, help="패턴 명시 시 ADV 사용 비율(기본 0.25)")

    # OpenAI 옵션
    ap.add_argument("--use_openai_paraphrase", action="store_true")
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")  # 현재는 4o-mini 고정 호출
    ap.add_argument("--openai_sleep_ms", type=int, default=120)
    ap.add_argument("--openai_max_tokens", type=int, default=80)
    ap.add_argument("--openai_temperature", type=float, default=0.7)
    ap.add_argument("--openai_top_p", type=float, default=0.9)

    # 저장/로깅
    ap.add_argument("--report_path", type=str, default=None, help="라인별 diff JSONL 경로")
    ap.add_argument("--include_meta", action="store_true", help="seed_input/checks/diff를 최종 데이터셋에 포함")
    ap.add_argument("--keep_only_valid", action="store_true", help="요소검증 올패스인 샘플만 최종 데이터셋에 포함")
    args = ap.parse_args()
    main(**vars(args))