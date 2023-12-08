# TSpred
This is the code for the paper "TSpred: a robust prediction framework for TCR-epitope interactions based on an ensemble deep learning approach using paired chain TCR sequence data". 

Link to the paper: https://www.biorxiv.org/content/10.1101/2023.12.04.570002v1

## Datasets
The datasets used in the paper are provided in the `datasets.zip` file. 
- NetTCR_full
- IMMREP
- NetTCR_bal
- NetTCR_strict (consists of 5 different cross validation datasets with 5 different random seeds)

## Setting up a conda environment for TSpred
First create a conda environment, and activate the environment:
```
conda env create -n tspred python=3.7.0
conda activate tspred
```
Then install requirements:
```
pip install -r requirements.txt
```
Our model is trained on an environment with CUDA version 11.3.

## Running code on example data
Example training, validation, and test data can be found in the `example_data` folder. 

To format data and perform preprocessing, run:
```
python get_data_ready.py
python pp.py
```
To train the model, run:
```
python train.py
```
To test the model, run:
```
python test.py
```

## Contact information
If you have questions, please contact ha01994@kaist.ac.kr.


