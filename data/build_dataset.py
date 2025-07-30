# data/build_dataset.py
import json, random, re
from typing import Dict, List, Tuple
from mapping.lexicon import ALL_AGES, GENDER_TERMS, PTRN_MAP
from mapping.lexicon_region import REGIONS  # [(full_name, code, [aliases...]), ...]

random.seed(7)

# -----------------------------
# Helpers: small NLG library
# -----------------------------
def choice(x): return random.choice(x)

# 말끝/요청 구
ASK_TAIL = [
    "알려줘", "보여줘", "보고 싶어", "데이터를 보여줘", "정리해줘", "조회해줘", "데이터 조회해줘"
    "볼 수 있을까?", "조회 가능할까?", "좀 알려줘", "부탁해", "조회 부탁해",
]

# 불필요어/수식어 (있어도 되고 없어도 되는 단어)
FILLERS = ["", "", "", "", "데이터", "현황", "통계", "규모", "값", "비율", "수치"]

# 지역 주변 표현
NEAR_SUFFIX = ["", "", "", " 일대", " 근처", " 인근", " 쪽"]

# 구분자/괄호
PARENS_L = ["", " (", " ["]
PARENS_R = ["", ")", "]"]

# -----------------------------
# Canonical -> surface forms
# -----------------------------
def surface_gender(g: str) -> str:
    # GENDER_TERMS 키(예: "여성","남성","전체")를 보다 자연스럽게 변형
    table = {
        "여성": ["여성", "여자", "여"],
        "남성": ["남성", "남자", "남"],
        "전체": ["전체", "모든 성별", "성별 무관", "남녀 모두"]
    }
    return choice(table.get(g, [g]))

def surface_age(age_key: str) -> str:
    # ALL_AGES의 키(예: "30대","30,40대","전연령",...)를 자연어로 변형
    # 숫자 패턴(2자리,2자리대) 감지하여 다양한 표기 제공
    k = age_key.replace(" ", "")
    # 3040대 형태 허용을 위한 매핑
    mapper = {
        "전연령": ["전연령", "전 연령대", "전 연령층", "모든 연령대"],
        "모든연령": ["모든 연령", "모든 연령대", "전 연령"],
    }
    if k in mapper:
        return choice(mapper[k])

    # 30,40대 / 20,30,40대 등
    m_multi = re.findall(r'(\d{2})', k)
    if m_multi and "대" in k and len(m_multi) >= 2:
        # 예: 30,40대 → 다양화
        nums = m_multi
        join_forms = [
            ",".join(nums) + "대",
            "·".join(nums) + "대",
            "~".join([nums[0], nums[-1]]) + "대",
            "".join(nums) + "대",  # 3040대
            f"{nums[0]}대와 {nums[-1]}대"
        ]
        return choice(join_forms)

    # 단일 30대
    m_single = re.match(r'(\d{2})대', k)
    if m_single:
        n = m_single.group(1)
        return f"{n}대"

    # 그 외는 원문
    return age_key

def surface_ptrn(ptrn_key: str) -> str:
    # 서비스의 PTRN_MAP 키를 자연스러운 동의어로 (예시는 직장/방문/쇼핑)
    # 실제 키 구조에 맞게 확장하세요.
    table = {
        "직장": ["직장", "직장 목적", "출퇴근", "직장 인구", "근로자"],
        "방문": ["방문", "방문자", "유입", "유입 인구"],
        "쇼핑": ["쇼핑", "소비", "상권", "구매"]
    }
    arr = table.get(ptrn_key, [ptrn_key])
    # 뒤에 '인구/규모' 같은 수식어 붙이기 (확률적으로)
    tail = choice(["", "", " 인구", " 규모"])
    return choice(arr) + tail

def surface_month(month: str) -> str:
    # "2025년 1월" → "2025년 1월" / "25년 1월" / "2025.1"
    # "이번 달" → "이번 달/이달/금월"
    # "지난 달" → "지난 달/전월/저번 달"
    month = month.strip()
    if month == "이번 달":
        return choice(["이번 달", "이달", "금월"])
    if month == "지난 달":
        return choice(["지난 달", "전월", "저번 달"])

    m = re.match(r'(\d{4})\s*년\s*(\d{1,2})\s*월', month)
    if m:
        y, mm = m.group(1), m.group(2)
        forms = [
            f"{y}년 {mm}월",
            f"{y}.{int(mm)}",
            f"{y[2:]}년 {mm}월",
            f"{y}년 {mm}월 기준",
        ]
        return choice(forms)
    return month

def surface_region(alias: str) -> str:
    # 지역 별칭 뒤에 "일대/근처/인근/쪽"을 확률적으로 부여
    return alias + choice(NEAR_SUFFIX)

def pick_region_alias(region_nm: str, aliases: List[str]) -> str:
    # 별칭에서 하나 고르고, '부산 사상구'처럼 시/도 축약형 생성도 가끔 추가
    cand = (aliases or []) + [region_nm]
    alias = choice(cand)
    # "부산광역시 사상구" → "부산 사상구" 축약
    m = re.match(r'(.+?[시도])\s+(.+)', region_nm)
    short = None
    if m:
        si, rest = m.group(1), m.group(2)
        short = si.split()[0] + " " + rest  # '부산 사상구' 같이 축약
    options = [alias, region_nm] + ([short] if short else [])
    return surface_region(choice(options))

def pick_fill() -> str:
    x = choice(FILLERS)
    return (" " + x) if x else ""

def pick_ask() -> str:
    x = choice(ASK_TAIL)
    return (" " + x) if x else ""

# -----------------------------
# 템플릿(자연어형)
# -----------------------------
TEMPLATES = [
    "{month} {region} {age} {gender} {ptrn}{fill}{ask}",
    "{region} {month} {ptrn} {age} {gender}{fill}{ask}",
    "{month}{fill} {region}에서 {age} {gender} {ptrn} 보고 싶어",
    "{region}{near} {month} {age} {gender}{fill} {ptrn} 알려줘",
    "{month} 기준 {region} {ptrn} {age} {gender}{fill}{ask}",
    "{region} {age} {gender} {ptrn} {month}{fill}{ask}",
    "{region} {ptrn} {age}{fill} {gender} {month}{ask}",
    "{month} {region}{fill} {gender} {age} {ptrn}{ask}",
    "{region} {month}{fill} {age} 대상 {gender} {ptrn}",
    "{region} {ptrn}{fill}{pL}{month}{pR} {age} {gender}{ask}",
]

# -----------------------------
# OUTPUT 생성(기존 로직 유지)
# -----------------------------
def to_output(month:str, region_nm:str, gender_cd:List[int], ages:List[int], ptrn_cd:List[int]):
    # 참고: 실제 운영에선 regex로 ym 파싱 권장. 여기선 간단 매핑 유지.
    month_map = {
        "2025년 1월": 202501, "2025년 2월": 202502, "2025년 3월": 202503, "2025년 4월": 202504,
        "2025년 5월": 202505, "2025년 6월": 202506, "2025년 7월": 202507, "2025년 8월": 202508,
        "2025년 9월": 202509, "2025년 10월": 202510, "2025년 11월": 202511, "2025년 12월": 202512,
        "이번 달": 202507, "지난 달": 202506,  # 예시 고정값
    }
    base_ym = month_map.get(month, 202507)
    return {
        "base_ym": base_ym,
        # "region_cd": [region_cd],
        "region_nm": region_nm,
        "ptrn": ptrn_cd,
        "gender_cd": gender_cd,
        "age_cd": ages
    }

# -----------------------------
# 메인 데이터 생성
# -----------------------------
MONTHS = [
    "2025년 1월","2025년 2월","2025년 3월","2025년 4월",
    "2025년 5월","2025년 6월","2025년 7월","2025년 8월",
    "2025년 9월","2025년 10월","2025년 11월","2025년 12월",
    "이번 달","지난 달",
]
GENDER_KEYS = list(GENDER_TERMS.keys())  # ["여성","남성","전체",...]
AGE_KEYS = list(ALL_AGES.keys())         # ["30대","30,40대","전연령",...]
PTRN_KEYS = list(PTRN_MAP.keys())        # ["직장","방문","쇼핑",...]

def make_input(month_key:str, region_nm:str, aliases:List[str], gender_key:str, age_key:str, ptrn_key:str) -> str:
    month_txt  = surface_month(month_key)
    region_txt = pick_region_alias(region_nm, aliases)
    gender_txt = surface_gender(gender_key)
    age_txt    = surface_age(age_key)
    ptrn_txt   = surface_ptrn(ptrn_key)

    tmpl = choice(TEMPLATES)
    s = tmpl.format(
        month=month_txt,
        region=region_txt,
        age=age_txt,
        gender=gender_txt,
        ptrn=ptrn_txt,
        near=choice(["", " 주변", " 부근"]),
        fill=pick_fill(),
        ask=pick_ask(),
        pL=choice(PARENS_L),
        pR=choice(PARENS_R),
    )

    # 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    # ‘ ,’ 같은 어색한 공백 교정
    s = s.replace(" ,", ",")
    return s

def main(out_path="parse_data.jsonl", n=1000, seed=7):
    random.seed(seed)
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(n):
            region_nm, rcd, aliases = choice(REGIONS)
            month_key  = choice(MONTHS)
            gender_key = choice(GENDER_KEYS)
            age_key    = choice(AGE_KEYS)
            ptrn_key   = choice(PTRN_KEYS)

            # 라벨(정답) — 기존 방식 그대로
            gender_cd = [1] if gender_key == "여성" else ([0] if gender_key == "남성" else [0,1])
            age_cd    = ALL_AGES[age_key]
            # ptrn 매핑 예시(운영 PTRN_MAP 코드표에 맞게 수정)
            if ptrn_key in PTRN_MAP:
                ptrn_cd = [PTRN_MAP[ptrn_key]]
            else:
                ptrn_cd = [2]  # 기본 방문

            text = make_input(month_key, region_nm, aliases, gender_key, age_key, ptrn_key)

            item = {
                "input": text,
                "output": to_output(month_key, region_nm, gender_cd, age_cd, ptrn_cd)
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()