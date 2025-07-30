# data/build_dataset.py
import json, random
from itertools import product
from mapping.lexicon import ALL_AGES, GENDER_TERMS, PTRN_MAP
from mapping.lexicon_region import REGIONS


def pick_alias(region_nm, aliases):
    if isinstance(aliases, (list, tuple)) and len(aliases) > 0:
        return random.choice(list(aliases))
    return region_nm

MONTHS = ["2025년 1월", "2025년 2월", "2025년 3월", "2025년 4월", "2025년 5월", "2025년 6월", "2025년 7월", "2025년 8월", "2025년 9월", "2025년 10월", "2025년 11월", "2025년 12월", "이번 달", "지난 달"]
GENDERS = list(GENDER_TERMS.keys())
AGES = list(ALL_AGES.keys())
PTRNS = list(PTRN_MAP.keys())

TEMPLATES = [
    "{month} {region_alias} {ages} {gender} {ptrn}",
    "{month} {region_alias} {gender} {ages} {ptrn}",
    "{month} {region_alias} {ptrn} {gender} {ages}",
    "{month} {region_alias} {ages} {ptrn} {gender}",
    "{month} {region_alias} {ages} {ptrn} {gender}",
]

def to_output(month:str, region_nm:str, region_cd:int, gender:list[int], ages:list[int], ptrn:list[int]):
    # base_ym은 후처리에서 다시 검증하므로 생성기는 간단하게 만듦
    y = 2025 if "2025" in month else 2025  # 예시
    m = 1 if "1월" in month else (2 if "2월" in month else 7)  # 예시
    return {
        "base_ym": y*100 + m,
        "region_cd": [region_cd],
        "region_nm": region_nm,
        "ptrn": ptrn,
        "gender_cd": gender,
        "age_cd": ages
    }

def main(out_path="parse_data.jsonl", n=5000, seed=7):
    random.seed(seed)
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(n):
            region_nm, rcd, aliases = random.choice(REGIONS)
            alias = pick_alias(region_nm, aliases)  #
            alias = random.choice(aliases or [region_nm])
            month = random.choice(MONTHS)
            gender = random.choice(GENDERS)
            ages = random.choice(AGES)
            ptrn = random.choice(PTRNS)
            tmpl = random.choice(TEMPLATES)
            text = tmpl.format(month=month, region_alias=alias, gender=gender, ages=ages, ptrn=ptrn)
            # 라벨(정답) 작성(간단 예시) — 실제는 규칙함수 재사용 권장
            gender_cd = [1] if gender=="여성" else ([0] if gender=="남성" else [0,1])
            age_cd = ALL_AGES[ages]

            ptrn_cd = [1] if ptrn=="직장" else ([2] if ptrn=="방문" else [3])

            item = {
                "input": text,
                "output": to_output(month, region_nm, rcd, gender_cd, age_cd, ptrn_cd)
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__=="__main__":
    main()