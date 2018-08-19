#!/usr/bin/env bash
conda create --clone sdn --name dltemplate-sdn
source activate dltemplate-sdn
conda install matplotlib tqdm numpy==1.12.1
/Users/d777710/miniconda3/envs/dltemplate-sdn/bin/pip install opencv-python scikit-learn numpy==1.12.1
