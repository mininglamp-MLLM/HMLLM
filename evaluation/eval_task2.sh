#!/bin/bash

# Define common arguments for all scripts
PRED_JSON="<your_pred_jsonl_path>"
OUTPUT_DIR="<your_output_dir>"
API_URL="<your_openai_api_url>"
API_KEY="<your_openai_api_key>"
NUM_TASKS=<number_of_tasks>

python evaluate_task2.py \
  --pred_path $PRED_JSON \
  --output_dir "${OUTPUT_DIR}/" \
  --output_json "${OUTPUT_DIR}-result.json" \
  --api_url $API_URL \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS