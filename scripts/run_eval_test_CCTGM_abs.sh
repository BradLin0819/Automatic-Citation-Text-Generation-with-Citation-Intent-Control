#!/bin/bash

MODEL_NAME=$1
BASE_DATA_DIR="data/preprocessed"
MODEL_PATH_NAME=$(echo ${MODEL_NAME} | cut -d '/' -f2)
EXPERIMENT_DIRNAME=CCTGM_abs
EXPERIMENT_PATH=experiments/${MODEL_PATH_NAME}/${EXPERIMENT_DIRNAME}
CANDIDATE_PATH=${EXPERIMENT_PATH}/${EXPERIMENT_DIRNAME}.test.result
DATA_PATH=${BASE_DATA_DIR}/data_CCTGM_abs
REFERENCE_PATH=${DATA_PATH}/test.target

if [ ! -d ${EXPERIMENT_PATH} ]; then
    mkdir -p ${EXPERIMENT_PATH}
fi

python transformers_src/run_eval.py ${MODEL_NAME} ${DATA_PATH}/test.source ${CANDIDATE_PATH} \
    --pretrained_model_path models/output_dir_$(echo ${MODEL_PATH_NAME} | tr '-' '_')_${EXPERIMENT_DIRNAME}/pytorch_model.bin  \
    --reference_path ${REFERENCE_PATH} \
    --fp16 \
    --score_path experiments/${MODEL_PATH_NAME}/${EXPERIMENT_DIRNAME}/${EXPERIMENT_DIRNAME}_test_metric.json \
    --num_beams 4 \
    --length_penalty 2 \
    --no_repeat_ngram_size 3

files2rouge ${REFERENCE_PATH} ${CANDIDATE_PATH} > ${EXPERIMENT_PATH}/${EXPERIMENT_DIRNAME}.rouge.result
bert-score -r ${REFERENCE_PATH} -c ${CANDIDATE_PATH} --lang en-sci -m scibert-scivocab-cased > ${EXPERIMENT_PATH}/${EXPERIMENT_DIRNAME}.bertscore.result
