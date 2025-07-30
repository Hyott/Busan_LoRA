# test_infer_single.py
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "dnotitia/Llama-DNA-1.0-8B-Instruct"
ADAPTER = "./lora-intent-exp/final"  # 학습 스크립트가 저장한 경로로 맞추세요

def build_prompt(user_input: str) -> str:
    return f"Q: {user_input}\nA: "

def run(prompt: str):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.to(device).eval()

    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n=== RAW OUTPUT ===")
    print(text)
    print("\n=== JSON PARSED ===")
    print(json.loads(text))  # 실패 시 예외 발생 → 모델/데이터 확인

if __name__ == "__main__":
    prompt = build_prompt("2025년 1월 논산시 부적면 30,40대 여성 직장인구 알려줘")
    run(prompt)