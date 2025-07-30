# eval_lora_parser.py
import os, re, json, math, random, argparse
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "dnotitia/Llama-DNA-1.0-8B-Instruct"
# 파일 상단/함수들 근처에 유틸 추가
def _list_norm(x):
    return sorted(x or [])

def diff_fields(pred:Dict[str,Any], gold:Dict[str,Any])->Dict[str, Any]:
    """어떤 필드가 다른지 요약."""
    p = canon(pred); g = canon(gold)
    diffs = {}
    if p["base_ym"] != g["base_ym"]:
        diffs["base_ym"] = {"pred": p["base_ym"], "gold": g["base_ym"]}
    if p["region_nm"] != g["region_nm"]:
        diffs["region_nm"] = {"pred": p["region_nm"], "gold": g["region_nm"]}
    for fld in ["ptrn","gender_cd","age_cd"]:
        if _list_norm(p[fld]) != _list_norm(g[fld]):
            diffs[fld] = {"pred": p[fld], "gold": g[fld]}
    return diffs


def detect_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def set_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_jsonl(path:str) -> List[Dict[str,Any]]:
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj=json.loads(line)
                # {"input":..., "output":{...}}
                rows.append(obj)
    return rows

def split_rows(rows:List[Dict[str,Any]], ratio:float, seed:int):
    random.seed(seed); random.shuffle(rows)
    n=len(rows); k=max(1,int(n*ratio))
    return rows[:k], rows[k:]  # train, eval

def build_prompt(user_input:str)->str:
    return f"Q: {user_input}\nA: "

def extract_json(text:str)->str:
    """생성 텍스트에서 첫 번째 JSON 블록을 안전하게 추출"""
    # 1) 가장 바깥 {} 블록 추출
    start = text.find("{")
    end   = text.rfind("}")
    if start!=-1 and end!=-1 and end>start:
        cand = text[start:end+1]
        # 2) 문자열 내부 괄호 문제 방지용 간단 교정(선택)
        return cand
    # 폴백: 코드블록/마크다운 제거 등 추가 가능
    return text

def try_parse_json(text:str):
    try:
        return json.loads(text), None
    except Exception as e:
        return None, e

def as_set(x):
    if x is None: return set()
    if isinstance(x, list): return set(x)
    return {x}

def canon(obj:Dict[str,Any])->Dict[str,Any]:
    """비교 전 정렬/타입 통일 (region_cd 제거 버전)"""
    if not isinstance(obj, dict): return obj
    out={}
    out["base_ym"] = int(obj.get("base_ym")) if obj.get("base_ym") is not None else None
    # 집합형 필드만 정규화
    def _norm_list_int(x):
        if isinstance(x, list):
            y=[]
            for t in x:
                try: y.append(int(t))
                except: pass
            return sorted(set(y))
        return None
    out["ptrn"]      = _norm_list_int(obj.get("ptrn"))
    out["gender_cd"] = _norm_list_int(obj.get("gender_cd"))
    out["age_cd"]    = _norm_list_int(obj.get("age_cd"))
    # 문자열 필드
    rn = obj.get("region_nm")
    out["region_nm"] = rn.strip() if isinstance(rn,str) else None
    return out

def exact_match(pred:Dict[str,Any], gold:Dict[str,Any])->bool:
    return canon(pred)==canon(gold)

def jaccard(a:List[int], b:List[int])->float:
    A=set(a or []); B=set(b or [])
    if not A and not B: return 1.0
    if not A and B: return 0.0
    if A and not B: return 0.0
    inter=len(A&B); union=len(A|B)
    return inter/union if union>0 else 1.0

def field_scores(pred:Dict[str,Any], gold:Dict[str,Any])->Dict[str,float]:
    p = canon(pred); g = canon(gold)
    scores={}
    scores["base_ym_acc"]   = float(p["base_ym"]==g["base_ym"])
    scores["region_nm_acc"] = float(p["region_nm"]==g["region_nm"])
    # 집합 유사도 + 완전일치
    scores["ptrn_jac"]      = jaccard(p["ptrn"], g["ptrn"])
    scores["gender_jac"]    = jaccard(p["gender_cd"], g["gender_cd"])
    scores["age_jac"]       = jaccard(p["age_cd"], g["age_cd"])
    scores["ptrn_exact"]    = float((p["ptrn"] or []) == (g["ptrn"] or []))
    scores["gender_exact"]  = float((p["gender_cd"] or []) == (g["gender_cd"] or []))
    scores["age_exact"]     = float((p["age_cd"] or []) == (g["age_cd"] or []))
    return scores

from mapping.region_normalizer import normalize_region_nm
import copy

from mapping.region_normalizer import normalize_region_nm
from mapping.ptrn_normalizer import normalize_ptrn_to_codes, remap_legacy_codes

def postprocess(pred: dict, raw_text: str) -> dict:
    pred = pred.copy()
    pred["ptrn"] = remap_legacy_codes(pred.get("ptrn"))
    if not pred.get("ptrn"):
        pred["ptrn"] = normalize_ptrn_to_codes(raw_text)

    # region_nm 표준화 (기존 로직 유지)
    rn = pred.get("region_nm")
    if isinstance(rn, str):
        std = normalize_region_nm(rn) or normalize_region_nm(raw_text) or rn.strip()
        pred["region_nm"] = std

    # ptrn 보정
    codes = pred.get("ptrn")
    # 1) 레거시 코드 리매핑
    codes = remap_legacy_codes(codes)
    # 2) 비어있으면 텍스트에서 추정
    if not codes:
        codes = normalize_ptrn_to_codes(raw_text)   # 생활인구 또는 미언급이면 [0,1,2]
    pred["ptrn"] = codes

    return pred


def evaluate(model, tok, eval_rows:List[Dict[str,Any]], device:str, max_new_tokens:int=256, limit:int=None, dump_errors:str=None):
    n = len(eval_rows) if limit is None else min(limit, len(eval_rows))
    ok_parse = 0

    keys = ["base_ym_acc","region_nm_acc","ptrn_jac","gender_jac","age_jac","ptrn_exact","gender_exact","age_exact"]
    agg_raw  = {k:0.0 for k in keys}
    agg_post = {k:0.0 for k in keys}
    em_raw = 0
    em_post = 0

    logs = []   # ✅ 여기에 모든 로그 누적 (parse_fail, mismatch_raw, mismatch_post)

    for i in range(n):
        r = eval_rows[i]
        prompt = build_prompt(r["input"])
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        cand = extract_json(text)
        pred, err = try_parse_json(cand)

        if pred is None:
            logs.append({
                "type": "parse_fail",
                "input": r["input"],
                "pred_raw": text,
                "gold": r["output"],
                "error": str(err),
            })
            continue

        ok_parse += 1

        # ----- RAW -----
        diffs_raw = diff_fields(pred, r["output"])
        if len(diffs_raw) == 0:
            em_raw += 1
        else:
            logs.append({
                "type": "mismatch_raw",
                "input": r["input"],
                "pred": pred,
                "gold": r["output"],
                "diffs": diffs_raw,
            })
        fs_raw = field_scores(pred, r["output"])
        for k in keys: agg_raw[k] += fs_raw[k]

        # ----- POSTPROCESSED -----
        pred_pp = postprocess(pred, r["input"])
        diffs_post = diff_fields(pred_pp, r["output"])
        if len(diffs_post) == 0:
            em_post += 1
        else:
            logs.append({
                "type": "mismatch_post",
                "input": r["input"],
                "pred_post": pred_pp,
                "gold": r["output"],
                "diffs": diffs_post,
            })
        fs_post = field_scores(pred_pp, r["output"])
        for k in keys: agg_post[k] += fs_post[k]

    def _avg(d): return {k:(v/n if n>0 else 0.0) for k,v in d.items()}
    res = {}
    res["count"] = n
    res["json_parse_rate"] = ok_parse/n if n>0 else 0.0
    res.update({f"{k}_raw":v for k,v in _avg(agg_raw).items()})
    res.update({f"{k}_post":v for k,v in _avg(agg_post).items()})
    res["exact_match_raw"]  = em_raw/n  if n>0 else 0.0
    res["exact_match_post"] = em_post/n if n>0 else 0.0

    if dump_errors:
        import os, json
        os.makedirs(os.path.dirname(dump_errors) or ".", exist_ok=True)
        with open(dump_errors, "w", encoding="utf-8") as f:
            for e in logs:   # ✅ 파싱 실패 + 불일치 모두 기록
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return res




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_paths", type=str, nargs="+", required=True, help="Train jsonl(s) used for training; we will split to build eval set.")
    ap.add_argument("--adapter", type=str, required=True, help="LoRA adapter directory (e.g., ./lora-intent-exp/final)")
    ap.add_argument("--train_ratio", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None, help="Evaluate on first N examples of eval set")
    ap.add_argument("--dump_errors", type=str, default="./eval_errors.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    set_seed(args.seed)
    device = detect_device()
    print(f"[Info] device={device}")

    # 데이터 로드 & 검증 세트 구성
    rows=[]
    for p in args.data_paths:
        rows.extend(load_jsonl(p))
    train_rows, eval_rows = split_rows(rows, ratio=args.train_ratio, seed=args.seed)
    print(f"[data] total={len(rows)} train={len(train_rows)} eval={len(eval_rows)}")

    # 모델 로드
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.to(device).eval()

    # 평가
    res = evaluate(model, tok, eval_rows, device, max_new_tokens=args.max_new_tokens, limit=args.limit, dump_errors=args.dump_errors)

    print("\n==== Evaluation (eval set) ====")
    print(f"count               : {res['count']}")
    print(f"json_parse_rate     : {res['json_parse_rate']:.4f}")

    # RAW
    print("\n-- RAW --")
    for k in ["exact_match_raw","base_ym_acc_raw","region_nm_acc_raw","ptrn_jac_raw","gender_jac_raw","age_jac_raw","ptrn_exact_raw","gender_exact_raw","age_exact_raw"]:
        print(f"{k:20s}: {res[k]:.4f}")

    # POST
    print("\n-- POSTPROCESSED --")
    for k in ["exact_match_post","base_ym_acc_post","region_nm_acc_post","ptrn_jac_post","gender_jac_post","age_jac_post","ptrn_exact_post","gender_exact_post","age_exact_post"]:
        print(f"{k:20s}: {res[k]:.4f}")

    print(f"\n[errors] dumped to: {args.dump_errors}")

if __name__ == "__main__":
    main()