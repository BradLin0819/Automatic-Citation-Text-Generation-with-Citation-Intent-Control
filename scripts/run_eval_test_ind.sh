#!/bin/bash

INTENT=$1
MODEL_NAME=$2
MODEL_PATH_NAME=$(echo ${MODEL_NAME} | cut -d '/' -f2)
EXPERIMENT_DIRNAME=ind
EXPERIMENT_PATH=experiments/${MODEL_PATH_NAME}/${EXPERIMENT_DIRNAME}
CANDIDATE_PATH=${EXPERIMENT_PATH}/${INTENT}.test.result
DATA_PATH=scicite_data_${INTENT}_preprocessed_3
REFERENCE_PATH=${DATA_PATH}/test.target

if [ ! -d ${EXPERIMENT_PATH} ]; then
    mkdir -p ${EXPERIMENT_PATH}
fi

python transformers_src/run_eval.py ${MODEL_NAME} ${DATA_PATH}/test.source ${EXPERIMENT_PATH}/${INTENT}.test.result \
    --pretrained_model_path models/output_dir_$(echo ${MODEL_PATH_NAME} | tr '-' '_')_${INTENT}/pytorch_model.bin  \
    --reference_path ${REFERENCE_PATH} \
    --fp16 \
    --score_path ${EXPERIMENT_PATH}/${INTENT}_test_metric.json \
    --num_beams 4 \
    --length_penalty 2 \
    --no_repeat_ngram_size 3

files2rouge ${REFERENCE_PATH} ${CANDIDATE_PATH} > ${EXPERIMENT_PATH}/${INTENT}.rouge.result
bert-score -r ${REFERENCE_PATH} -c ${CANDIDATE_PATH} --lang en-sci -m scibert-scivocab-cased > ${EXPERIMENT_PATH}/${INTENT}.bertscore.result
