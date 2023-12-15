# TSpred
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10347986.svg)](https://doi.org/10.5281/zenodo.10347986)

This is the code for the paper "TSpred: a robust prediction framework for TCR-epitope interactions based on an ensemble deep learning approach using paired chain TCR sequence data". 

Link to the paper: https://www.biorxiv.org/content/10.1101/2023.12.04.570002v2



## Datasets
The datasets used in the paper are provided in the `datasets.zip` file. 
- NetTCR_full
- IMMREP
- NetTCR_bal
- NetTCR_strict (consists of 5 different cross validation datasets generated with 5 different random seeds)



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
* Please note: our model is trained on an environment with CUDA version 11.3.



## Running code on an example dataset
Example training, validation, and test data can be found in the `example_data` folder. 
They are in the following format:

    peptide,A1,A2,A3,B1,B2,B3,binder
    NLVPMVATV,SVFSS,VVTGGEV,AGPEGGYSGAGSYQLT,SGDLS,YYNGEE,ASSVSGATADTQY,0
    LLWNGPMAV,TRDTTYY,RNSFDEQN,ALSGEGTGRRALT,GTSNPN,SVGIG,AWSVQGTDTQY,0
    TTDPSFLGRY,TSGFNG,NVLDGL,AVRVFNARLM,SNHLY,FYNNEI,ASSEEIAKNIQY,0
    GILGFVFTL,VSGLRG,LYSAGEE,AVRANQAGTALI,SGHRS,YFSETQ,ASSLTGSNTEAF,0

### Data preprocessing
To change the data format to fit our data processing pipeline, run:
```
python get_data_ready.py
```
This will change the data to the following format:

    pep_seq,tcr_seq,label,split
    pep16,tcr2191,0,train
    pep3,tcr604,0,train
    pep15,tcr199,0,train
    pep0,tcr700,0,train
    
To perform preprocessing on the formatted data, run:
```
python pp.py
```
This will generate preprocessed files (in .npy and .pkl format) in the `features` folder. These files are used for training and testing.

### Training
To train the model, run:
```
python train.py
```
The trained models will be saved in the directory `save_dir/cnn` and `save_dir/att`.

The training and validation losses will be saved in the directory `losses/cnn` and `losses/att`.

### Testing
To test the model, run:
```
python test.py
```
This code will load the saved models and test the models on the test dataset, returning the ROC-AUC and PR-AUC values. 
It will also generate `predictions.csv`, which provides the information on peptide-TCR pairs along with the labels and the model predictions.



## Contact information
If you have questions, please contact ha01994@kaist.ac.kr.


