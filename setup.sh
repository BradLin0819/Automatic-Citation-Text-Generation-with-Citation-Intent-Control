#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Install tools for reporting ROUGE score
pip install -U git+https://github.com/pltrdy/pyrouge

git clone https://github.com/pltrdy/files2rouge.git     
cd files2rouge
python setup_rouge.py
python setup.py install

cd ..

# Install scicite classifier for the citation intent correctness evaluation
git clone https://github.com/allenai/scicite.git

if [ ! -d ${SCICITE_INTENT_BASE_PATH} ]; then
    git clone https://github.com/allenai/scicite.git
fi

# Download pretrained classifier
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/models/scicite.tar.gz -P scicite

