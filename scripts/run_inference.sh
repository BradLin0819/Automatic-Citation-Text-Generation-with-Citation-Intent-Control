#!/bin/bash
MODEL_NAME=$1
MODEL_PATH=$2
CITATION_INTENT=$3
MODEL_PATH_NAME=$(echo ${MODEL_NAME} | cut -d '/' -f2)

python transformers_src/inference.py --model_name ${MODEL_NAME} \
    --pretrained_model_path ${MODEL_PATH} \
    --citing_context "During continuous user interaction, it is hard to provide rich visual feedback at interactive rates for datasets containing millions of entries. The contribution of this paper is a generic architecture that ensures responsiveness of the application even when dealing with large data and that is applicable to most types of information visualizations. Our architecture builds on the separation of the main application thread and the visualization thread, which can be cancelled early due to user interaction. In combination with a layer mechanism, our architecture facilitates generating previews incrementally to provide rich visual feedback quickly. To help avoiding common pitfalls of multi-threading, we discuss synchronization and communication in detail. We explicitly denote design choices to control trade-offs. A quantitative evaluation based on the system VI S P L ORE shows fast visual feedback during continuous interaction even for millions of entries. We describe instantiations of our architecture in additional tools." \
    --cited_context "SimVis is a novel technology for the interactive visual analysis of large and complex flow data which results from computational fluid dynamics (CFD) simulation. The new technology which has been researched and developed over the last years at the VRVis Research Center in Vienna, introduces a new approach for interactive graphical exploration and analysis of time-dependent data (computed on large three-dimensional grids, and resulting in a multitude of different scalar/vector values for each cell of these grids). In this paper the major new technological concepts of the SimVis approach are presented and real-world application examples are given." \
    --intent ${CITATION_INTENT} \
    --num_beams 4 \
    --length_penalty 2 \
    --no_repeat_ngram_size 3 \
    --fp16