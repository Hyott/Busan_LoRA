# eval_lora_parser.py
import os, re, json, math, random, argparse
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "dnotitia/Llama-DNA-1.0-8B-Instruct"

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
    """비교 전 정렬/타입 통일"""
    if not isinstance(obj, dict): return obj
    out={}
    out["base_ym"] = int(obj.get("base_ym")) if obj.get("base_ym") is not None else None
    for k in ["ptrn","gender_cd","age_cd"]:
        v = obj.get(k)
        if isinstance(v,list):
            v2=[]
            for t in v:
                try: v2.append(int(t))
                except: pass
            out[k]=sorted(set(v2))
        else:
            out[k]=None
    # region_nm 문자열
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
    scores["base_ym_acc"] = float(p["base_ym"]==g["base_ym"])
    scores["region_nm_acc"] = float(p["region_nm"]==g["region_nm"])
    # 집합 유사도
    scores["ptrn_jac"]      = jaccard(p["ptrn"], g["ptrn"])
    scores["gender_jac"]    = jaccard(p["gender_cd"], g["gender_cd"])
    scores["age_jac"]       = jaccard(p["age_cd"], g["age_cd"])
    # 완전일치 비율도 계산
    scores["ptrn_exact"]    = float(sorted(p["ptrn"] or []) == sorted(g["ptrn"] or []))
    scores["gender_exact"]  = float(sorted(p["gender_cd"] or []) == sorted(g["gender_cd"] or []))
    scores["age_exact"]     = float(sorted(p["age_cd"] or []) == sorted(g["age_cd"] or []))
    return scores

def evaluate(model, tok, eval_rows:List[Dict[str,Any]], device:str, max_new_tokens:int=256, limit:int=None, dump_errors:str=None):
    n = len(eval_rows) if limit is None else min(limit, len(eval_rows))
    ok_parse=0; em=0
    # 누적
    agg = {k:0.0 for k in ["base_ym_acc","region_nm_acc","ptrn_jac","gender_jac","age_jac","ptrn_exact","gender_exact","age_exact"]}
    errors=[]
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
        if pred is not None:
            ok_parse += 1
            if exact_match(pred, r["output"]): em += 1
            fs = field_scores(pred, r["output"])
            for k,v in fs.items(): agg[k]+=v
        else:
            # 파싱 실패 샘플 저장
            errors.append({
                "input": r["input"],
                "pred_raw": text,
                "gold": r["output"],
                "error": str(err),
            })

    # 평균
    res = {k:(v/n if n>0 else 0.0) for k,v in agg.items()}
    res["json_parse_rate"] = ok_parse/n if n>0 else 0.0
    res["exact_match"] = em/n if n>0 else 0.0
    res["count"] = n

    if dump_errors:
        os.makedirs(os.path.dirname(dump_errors) or ".", exist_ok=True)
        with open(dump_errors, "w", encoding="utf-8") as f:
            for e in errors[:200]:  # 너무 크지 않게 상위 200개만
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
    for k in [
        "count", "json_parse_rate", "exact_match",
        "base_ym_acc", "region_nm_acc",
        "ptrn_jac", "gender_jac", "age_jac",
        "ptrn_exact", "gender_exact", "age_exact",
    ]:
        print(f"{k:16s} : {res[k]:.4f}" if isinstance(res[k], float) else f"{k:16s} : {res[k]}")

    print(f"\n[errors] dumped to: {args.dump_errors}")

if __name__ == "__main__":
    main()