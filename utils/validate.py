# utils/validate.py
from mapping.lexicon_region import REGION_MAP, ALL_AGES

def validate_and_repair(obj: dict, original_text:str):
    # base_ym: 6자리, 1~12월
    ym = obj.get("base_ym")
    if not isinstance(ym, int) or ym<190001 or ym>210012 or not (1<= (ym%100) <= 12):
        obj["base_ym"] = None  # 또는 정책상 최근월로 보정

    # region
    if not obj.get("region_nm"):
        obj["region_nm"] = None
    else:
        # 코드 검증(여러개면 모두)
        valids = []
        for cd in obj["region_cd"]:
            if isinstance(cd, int):
                valids.append(cd)
        obj["region_cd"] = valids or None
        if obj["region_nm"] not in REGION_MAP:
            obj["region_nm"] = None

    # ptrn
    if not obj.get("ptrn") or not all(isinstance(x, int) for x in obj["ptrn"]):
        obj["ptrn"] = None

    # gender
    g = obj.get("gender_cd")
    if not g or not set(g).issubset({0,1}):
        obj["gender_cd"] = [0,1]

    # age
    a = obj.get("age_cd")
    if not a or not set(a).issubset(set(ALL_AGES)):
        obj["age_cd"] = ALL_AGES

    return obj