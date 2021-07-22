# Automatic-Citation-Text-Generation-with-Citation-Intent-Control

The implementation is mainly based on [Hugging Face Transformers](https://github.com/huggingface/transformers).

## Overview
Given an abstract of a citing paper, an abstract of a reference paper, and the citation intent specified by the author, our goal is to generate a relevant and fluent citation text satisfying the given citation intent. The overview of our system is illustrated as follows:

<p align="center">
  <img width="70%" src="https://i.imgur.com/Sa2qoGz.png">
</p>

To generate citation texts with different citation intents for the same reference paper, we propose a **Controllable Citation Text Generation Model (CCTGM)**, extending the existing pretrained text generation model by **taking the citation intent as control code**. The flowchart of our model is illustrated as follows:
![](https://i.imgur.com/qTiaxRi.png)
## Environment


| OS | GPU | RAM |
| -------- | -------- | -------- |
| Ubuntu 20.04     | RTX 3090 24GB     | 64 GB     |

## Installation
Clone this repository
```
git clone https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control.git
```
Change the directory to `Automatic-Citation-Text-Generation-with-Citation-Intent-Control`
```
cd Automatic-Citation-Text-Generation-with-Citation-Intent-Control
```
Create `conda` virtual environment
```
conda create --name cctgm python=3.7
```

Activate the virtual environment
```
conda activate cctgm
```
Set up the environment
```
./setup.sh
```
If your CUDA version is greater than `11.0`, please re-install PyTorch.
```
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
## Data
Our dataset is extended from [SciCite](https://github.com/allenai/scicite) citation intent classification dataset. We crawled the metadata of papers in SciCite using [Semantic Scholar API](https://api.semanticscholar.org/). The code can be found in `utils/fetch_papers_metadata.py`

The original dataset with the metadata of papers is provided in `data/original`. However, in our task, we use the dataset with further preprocessing steps. All preprocessed datasets can be found in `data/preprocessed`.

## Preprocessing (Optional)
The preprocessed datasets we used in the thesis are provided in `data/preprocessed`. If you want to generate the preprocessed datasets by yourself, please run the following command with specific arguments.
```
python utils/scicite_data_preprocess.py \
    --input_file <file to be preprocessed> \
    --out_dir <Path of output directory> \
    --dataset_type <train/val/test> \
    --intent <background/method/result> \
    --citing_input_mode <abstract/title>
    --prepend_token (Optional) \
    --outfile_type <file format of the exported file>
```
Where
- `input_file`: Input the path of the file to be preprocessed.
- `out_dir`: Input the path of the output directory.
- `dataset_type`: Input the type of the dataset (train/val/test).
- `intent`: Optional. Only select data samples with the specified citation intent (background/method/result). All data samples are selected if this option is not given.
- `citing_input_mode`: Use the abstract or the title as the input of the citing paper. (abstract/title)
- `prepend_token`: Optional. Prepend control codes to the input sequence. This option must be given for training `CCTGM`.
- `outfile_type`: The file format of the exported file. (`hf` for huggingface input format, `jsonl` for jsonlines file format)
## Training
We use two Transformer-based pretrained text generation models, `BART` and `T5` as the base model. 

Moreover, considering the abstract of the citing paper could not be available while writing citation texts, so we propose ```CCTGM-title```, which uses the title of the citing paper instead of the abstract.

To train BART-based `CCTGM-abs`, which uses the abstract as the input of the citing paper, please run the following command:
```
./scripts/train_bart_base_CCTGM_abs.sh
```

To train T5-based `CCTGM-abs`, please run
```
./scripts/train_t5_base_CCTGM_abs.sh
```
For the BART-based `CCTGM-title` configuration, please run
```
./scripts/train_bart_base_CCTGM_title.sh
```
and T5-based ```CCTGM-title```
```
./scripts/train_t5_base_CCTGM_title.sh
```

Please decrease `per_device_train_batch_size` in the training scripts when CUDA out-of-memory error is triggered.

You can also download pretrained model checkpoints from [here (12.0 GB)](https://drive.google.com/drive/folders/1wtl3_Z3oOWlnjI1iTr1ddjPaFkVjOpMO?usp=sharing).
## Evaluation
### Relevance Evaluation
To evaluate the relevance of generated citation texts of our models, we use [ROUGE](https://github.com/pltrdy/files2rouge) and [SciBERTScore](https://github.com/Tiiiger/bert_score) as metrics. 


To run the evaluation scripts, please download the pretrained model checkpoints first, unzip the compressed file, and move these models to `models/`.

```
mv pretrained-models/* models/
```
- CCTGM-abs
```
./scripts/run_eval_test_CCTGM_abs.sh <facebook/bart-base or t5-base>
```
The results will be generated in `experiments/<bart-base or t5-base>/CCTGM_abs`.
- CCTGM-title
```
./scripts/run_eval_test_CCTGM_title.sh <facebook/bart-base or t5-base>
```
The results will be generated in `experiments/<bart-base or t5-base>/CCTGM_title`.

### Citation Intent Correctness Evaluation
To evaluate the correctness of citation intent, we use the pretrained citation intent classifier provided by [SciCite](https://github.com/allenai/scicite) to automatically label the generated citation texts check if these citation intents satisfy the given citation intent.

Before running this evaluation, please create a new virtual environment and activate it.

Deactivate `cctgm` virtual environment first.
```
conda deactivate
```
Create a new environment for  Citation Intent Correctness Evaluation
```
conda create --name scicite python=3.6
```
Activate the environment
```
conda activate scicite
```
Install dependencies
```
pip install -r scicite/requirements.in -c scicite/constraints.txt
```
Downgrade `overrides` package
```
pip install overrides==3.1.0
```

Then, please run the following command:
```
./scripts/run_intent_accuracy.sh \
    <path of generated citation texts file> \
    test.jsonl \
    pred.jsonl
```
- `generated citation texts file` will generate in directories `experiments/<bart-base or t5-base>/<CCTGM-abs or CCTGM-title>` after running `run_eval_test_CCTGM_*.sh` scripts . The file name will be `<CCTGM-abs/CCTGM-title>.test.result`.
- After running the `run_intent_accuracy.sh` script, `accuracy.log` will show the citation intent correctness of the corresponding model.

## Inference
Please use `cctgm` virtual environment and run the following command:
```
python transformers_src/inference.py \
    --model_name <facebook/bart-base or t5-base> \
    --pretrained_model_path <pretrained_model_path> \
    --citing_context '<citing_context>' \
    --cited_context '<cited_context>' \
    --intent <user_specified_intent>
```
Where
- `model_name`: Type of the pretrained model. `facebook/bart-base` or `t5-base`.
- `pretrained_model_path`: Path of the pretrained model checkpoint.
- `citing_context`: The provided content of the citing paper.
- `cited_context`: The provided content of the reference paper.
- `intent`: The citation intent specified by the user. (background/method/result)

## Contact
If you have any questions about this project, please contact us or create an [issue](https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control/issues).

Email: thlin.cs08g@nctu.edu.tw
