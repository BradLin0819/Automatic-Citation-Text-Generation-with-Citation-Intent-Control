BASE_DIR="data/preprocessed"
DATA_DIR="${BASE_DIR}/data_method_abs"
OUTPUT_DIR="${BASE_DIR}/models/output_dir_bart_base_method"
TRANSFORMERS_PATH="transformers_src"
MAX_STEPS=40000
WARMUP_STEPS=$(bc <<< "${MAX_STEPS} * 0.1 / 1")

python ${TRANSFORMERS_PATH}/finetune_trainer.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --do_predict \
    --task summarization \
    --data_dir ${DATA_DIR} \
    --max_steps ${MAX_STEPS} \
    --learning_rate 3e-5 \
    --warmup_steps ${WARMUP_STEPS} \
    --dataloader_num_workers 4 \
    --save_total_limit 1 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --evaluation_strategy epoch \
    --logging_dir runs/bart_base_method \
    --seed 42 \
    --early_stopping_patience 3 \
    --fp16