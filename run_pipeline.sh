#!/bin/bash
# ==============================================================================
# EvolSQL Pipeline Launch Script  —  example with BIRD train (with optional DB Injection)
# Flow: Evo → Verify → Fix → [Inject] → Re-verify → Merge → Dedup → Rejection
# ==============================================================================


RUN_NAME="evolsql_bird"
INPUT_FILE="./data/bird_train.json"

python run_pipeline.py \
    --run_name "${RUN_NAME}" \
    --input_file "${INPUT_FILE}" \
    --output_dir ./results \
    --mschema_dir ./schemas/train_mschemas \
    --mschema_jsonl ./schemas/train_mschemas.jsonl \
    --db_root_path /path/to/bird \
    --mode train \
    --api_urls http://localhost:8001/v1,http://localhost:8002/v1 \
    --model Evolver \
    --indep_rounds 2 \
    --sampling_count 3 \
    --top_k 2 \
    --batch_size 2048 \
    --max_workers 128 \
    --num_verify_workers 64 \
    --sentence_model /path/to/all-mpnet-base-v2 \
    2>&1 | tee ./logs/experiments/${RUN_NAME}.log


