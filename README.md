######### train lora parser #########

python train/train_lora_parser.py \  --data_paths ./parse_data.jsonl \
  --output_dir ./lora-intent-exp \
  --epochs 2 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 2e-4 \
  --max_length 1024 \
  --gradient_checkpointing



python train/train_lora_parser.py \
  --data_paths ./parse_data.jsonl \
  --output_dir ./lora-intent-exp \
  --epochs 2 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 2e-4 \
  --max_length 1024



######### eval lora parser #########

python eval/eval_lora_parser.py \
  --data_paths ./parse_data.jsonl \
  --adapter ./lora-intent-exp/final \
  --train_ratio 0.98 \
  --seed 7 \
  --limit 1000 \
  --dump_errors ./eval_errors.jsonl




  #### augment paraphrase ####

 python scripts/augment_paraphrase.py \
  --src ./parse_data.jsonl \
  --dst ./parse_data_para_llama3.jsonl \
  --n 3 \
  --model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --max_new_tokens 96 \
  --temperature 0.85 \
  --top_p 0.92 \
  --strict_semantics



#### openai paraphrase ####
export OPENAI_API_KEY= "" 

python data/build_dataset.py \
  --out_path ./parse_data_openai.jsonl \
  --n 10 \
  --use_openai_paraphrase \
  --include_meta \
  --keep_only_valid \
  --report_path ./parse_data_openai_report.jsonl \
  --openai_max_tokens 64 \
  --openai_temperature 0.7 \
  --openai_top_p 0.9 \
  --openai_sleep_ms 150
