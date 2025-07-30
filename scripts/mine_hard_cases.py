# scripts/mine_hard_cases.py
import json, random, re
from typing import List, Dict, Any
from mapping.ptrn_normalizer import surface_ptrn_for_nlg

def load_jsonl(p):
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def augment_from_errors(errors_path:str, out_path:str, k:int=500, seed:int=7):
    random.seed(seed)
    rows=[]
    for e in load_jsonl(errors_path):
        gold = e.get("gold", {})
        if not gold: continue
        inp  = e.get("input", "")
        # 골드 ptrn 동의어 중 하나를 문장에 명시적으로 넣어 강조
        ptrn_codes = gold.get("ptrn", [])
        if not ptrn_codes: continue
        # 코드→라벨 역변환(예: 1:직장, 2:방문, 3:체류) — 운영 표에 맞춰 작성
        code2label = {1:"직장", 2:"방문", 3:"체류"}
        label = code2label.get(ptrn_codes[0])
        if not label: continue
        syns = surface_ptrn_for_nlg(label)
        if not syns: syns=[label]
        strong = random.choice(syns)
        # 입력을 가볍게 보정/강조
        new_inp = re.sub(r"(데이터|현황|규모)?$", f" {strong} 데이터 알려줘", inp).strip()
        rows.append({"input": new_inp, "output": gold})

    random.shuffle(rows)
    rows = rows[:k]
    with open(out_path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] mined hard cases: {len(rows)} → {out_path}")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--errors", required=True)
    ap.add_argument("--out", default="./hard_mined.jsonl")
    ap.add_argument("--k", type=int, default=500)
    args=ap.parse_args()
    augment_from_errors(args.errors, args.out, k=args.k)