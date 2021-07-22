#!/bin/bash
MODEL_NAME=$1
MODEL_PATH=$2
CITING_CONTEXT=$3
CITED_CONTEXT=$4
CITATION_INTENT=$5
MODEL_PATH_NAME=$(echo ${MODEL_NAME} | cut -d '/' -f2)

python transformers_src/inference.py --model_name ${MODEL_NAME} \
    --pretrained_model_path ${MODEL_PATH} \
    --citing_context ${CITING_CONTEXT} \
    --cited_context ${CITED_CONTEXT} \
    --intent ${CITATION_INTENT} \
    --num_beams 4 \
    --length_penalty 2 \
    --no_repeat_ngram_size 3 \
    --fp16
