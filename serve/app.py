# serve/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from utils.date_rules import parse_base_ym
from utils.region_rules import normalize_region
from utils.demographic_rules import parse_gender, parse_ages
from utils.pattern_rules import parse_ptrn
from utils.validate import validate_and_repair
from llm_infer import infer_json  # llama-server/llama.cpp/python whichever you use

app = FastAPI()

class ParseIn(BaseModel):
    text: str

@app.post("/parse")
def parse_endpoint(req: ParseIn):
    # 1) LLM 호출(문법 제약 JSON)
    raw = infer_json(req.text)  # dict 반환(문자열→dict 파싱 포함)

    # 2) 규칙 기반 보정(누락·불일치)
    now = datetime.now()
    base_ym = raw.get("base_ym") or parse_base_ym(req.text, now)
    region_nm, region_cd = (raw.get("region_nm"), raw.get("region_cd"))
    if not region_cd:
        _nm, _cd = normalize_region(req.text)
        region_nm = region_nm or _nm
        region_cd = region_cd or _cd

    gender_cd = raw.get("gender_cd") or parse_gender(req.text)
    age_cd = raw.get("age_cd") or parse_ages(req.text)
    ptrn = raw.get("ptrn") or parse_ptrn(req.text)

    out = {
        "base_ym": base_ym,
        "region_cd": region_cd,
        "region_nm": region_nm,
        "ptrn": ptrn,
        "gender_cd": gender_cd,
        "age_cd": age_cd
    }
    # 3) 최종 유효성 검사 + 수리
    final = validate_and_repair(out, original_text=req.text)
    return final