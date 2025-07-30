#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA fine-tuning for JSON parser on Llama-DNA-1.0-8B-Instruct (MPS-safe)
- Dataset: JSONL lines with {"input": str, "output": dict}
- Device: Apple Silicon (MPS) friendly
- Loss: Only on target(JSON) tokens (prompt tokens masked to -100)
"""

import os, json, math, random, time, inspect
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, TaskType

# =========================
# 🔧 CONFIG (기본값)
# =========================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

BASE_MODEL_ID = "dnotitia/Llama-DNA-1.0-8B-Instruct"
SEED = 42
MAX_LEN = 1024
EPOCHS = 2
LR = 2e-4
WARMUP_RATIO = 0.05
GRAD_ACCUM = 8
BATCH_SIZE = 1
CLIP_NORM = 1.0

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

PROMPT_TMPL = "Q: {inp}\nA: "

# =========================
# 🧩 유틸
# =========================
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            assert "input" in obj and "output" in obj, "Each line must have 'input' and 'output'."
            rows.append(obj)
    return rows

def train_eval_split(rows: List[Dict[str, Any]], ratio: float, seed: int):
    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    k = max(1, int(n * ratio))
    return rows[:k], rows[k:]

# =========================
# 🧠 토크나이즈/마스킹
# =========================
def encode_ids(tokenizer, text: str):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def build_example(tokenizer, input_text: str, output_obj: Dict[str, Any], max_len: int) -> Dict[str, Any]:
    prompt = PROMPT_TMPL.format(inp=(input_text or "").strip())
    answer = json.dumps(output_obj, ensure_ascii=False) + tokenizer.eos_token

    p_ids = encode_ids(tokenizer, prompt)
    a_ids = encode_ids(tokenizer, answer)

    # 프롬프트 먼저 자르고, 정답 최소 토큰 확보
    min_label_tokens = 4
    if len(p_ids) > max_len - min_label_tokens:
        p_ids = p_ids[-(max_len - min_label_tokens):]

    room = max_len - len(p_ids)
    if room <= 0:
        p_ids = p_ids[:max_len - 1]
        room = 1
    a_ids = a_ids[:room]

    input_ids = p_ids + a_ids
    attention_mask = [1] * len(input_ids)
    labels = ([-100] * len(p_ids)) + a_ids  # 프롬프트 구간 마스킹

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

@dataclass
class LabelPaddingCollator:
    pad_id: int
    pad_to_multiple_of: int = 8  # MPS도 패딩 정렬 유지

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            L = len(f["input_ids"]); pad = max_len - L
            batch["input_ids"].append(f["input_ids"] + [self.pad_id]*pad)
            batch["attention_mask"].append(f["attention_mask"] + [0]*pad)
            batch["labels"].append(f["labels"] + [-100]*pad)

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

class JsonlTorchDataset(TorchDataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return build_example(self.tok, r["input"], r["output"], self.max_len)

# =========================
# 🚀 MAIN
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="./lora-parser-exp")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--train_ratio", type=float, default=0.98)
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # 디버그 버전 출력
    import transformers, accelerate
    print("[DEBUG] transformers", transformers.__version__)
    print("[DEBUG] accelerate  ", accelerate.__version__)
    print("[Info] Device:", DEVICE)

    # (옵션) accelerate unwrap_model keep_torch_compile 패치
    from accelerate import Accelerator
    if "keep_torch_compile" not in str(inspect.signature(Accelerator.unwrap_model)):
        _orig_unwrap = Accelerator.unwrap_model
        def _patched_unwrap(self, model, *a, **kw):
            kw.pop("keep_torch_compile", None)
            return _orig_unwrap(self, model, *a, **kw)
        Accelerator.unwrap_model = _patched_unwrap
        print("[PATCH] Accelerator.unwrap_model patched to ignore keep_torch_compile")

    set_seed(args.seed)

    # 1) 데이터 로드 & 스플릿
    rows = []
    for p in args.data_paths:
        rows.extend(load_jsonl(p))
    train_rows, val_rows = train_eval_split(rows, ratio=args.train_ratio, seed=args.seed)
    print(f"[data] total={len(rows)} train={len(train_rows)} val={len(val_rows)}")

    # 2) 토크나이저/모델
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,   # 가중치 로드는 fp16, 학습은 fp16=False (MPS)
    )
    base.config.use_cache = False

    # 디바이스 이동
    if DEVICE == "mps":
        base.to("mps")
    elif DEVICE == "cuda":
        base.to("cuda")

    # 3) LoRA 어댑터
    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET,
    )
    model = get_peft_model(base, peft_cfg)
    model.enable_input_require_grads()  # LoRA+GC 조합 안정
    model.print_trainable_parameters()

    # 4) Dataset & Collator
    global MAX_LEN
    MAX_LEN = args.max_length  # build_example에서 참조
    train_ds = JsonlTorchDataset(train_rows, tok, max_len=args.max_length)
    val_ds   = JsonlTorchDataset(val_rows, tok, max_len=args.max_length)
    collator = LabelPaddingCollator(pad_id=tok.pad_token_id, pad_to_multiple_of=8)

    # 5) TrainingArguments (MPS-safe)
    # 평가/저장은 기본 비활성(간단 안정), 스케줄러는 수동 세팅
    total_steps = max(1, (len(train_ds) // args.batch_size) * args.epochs // max(1, args.grad_accum))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        fp16=False,                       # ✅ MPS에서는 False
        bf16=False,
        gradient_checkpointing=True,
        max_grad_norm=CLIP_NORM,
        seed=args.seed,
        dataloader_num_workers=0,         # macOS 안전
        dataloader_pin_memory=False,      # MPS 권장
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=None,        # 손실 기반 eval 미사용(원하면 val_ds로 바꾸세요)
        processing_class=tok,            # 경고는 무시 가능; 필요시 processing_class로 교체
        data_collator=collator,
    )

    # 옵티마이저/스케줄 수동 설정(코사인)
    trainer.create_optimizer_and_scheduler(num_training_steps=total_steps)
    trainer.lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=trainer.optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 6) Train
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    print(f"[train] done in {dt/60:.1f} min")

    # 7) Save adapter
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    print(f"[save] final adapter: {final_dir}")

if __name__ == "__main__":
    main()