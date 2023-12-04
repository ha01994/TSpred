# TSpred
This is the code for the paper "TSpred: a robust prediction framework for TCR-epitope interactions based on an ensemble deep learning approach using paired chain TCR sequence data". 

## Datasets
The datasets used in the paper are provided in datasets.zip file. 

## Setting up a conda environment for TSpred
First create a conda environment, and activate the environment
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
```eys.path.append('/home/ha01994/tspred_code/')
