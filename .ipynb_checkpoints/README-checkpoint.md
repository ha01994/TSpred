# TSpred
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10347986.svg)](https://doi.org/10.5281/zenodo.10347986)

This is the code for the paper "TSpred: a robust prediction framework for TCR-epitope interactions based on an ensemble deep learning approach using paired chain TCR sequence data" (Link: https://www.biorxiv.org/content/10.1101/2023.12.04.570002v2). 



## Datasets
The datasets used in the paper can be obtained by uncompressing the `datasets.zip` file. 
- NetTCR_full
- IMMREP
- NetTCR_bal
- NetTCR_strict
    - This dataset consists of 5 different cross validation datasets generated with 5 different random seeds.



## Setting up a conda environment for TSpred
First create a conda environment, and activate the environment:
```
conda env create -n tspred python=3.7.0
conda activate tspred
```
Then install the package requirements using pip:
```
pip install -r requirements.txt
```
* Please note: our model is trained on an environment with CUDA version 11.3, so we use torch==1.10.2+cu113. If you're using a different CUDA version, you can just install torch==1.10.2.



## Running code on an example dataset
Example training, validation, and test data can be found in the `example_data` folder. 

Example format of the files in `example_data`:

    peptide,A1,A2,A3,B1,B2,B3,binder
    NLVPMVATV,SVFSS,VVTGGEV,AGPEGGYSGAGSYQLT,SGDLS,YYNGEE,ASSVSGATADTQY,0
    LLWNGPMAV,TRDTTYY,RNSFDEQN,ALSGEGTGRRALT,GTSNPN,SVGIG,AWSVQGTDTQY,0
    TTDPSFLGRY,TSGFNG,NVLDGL,AVRVFNARLM,SNHLY,FYNNEI,ASSEEIAKNIQY,1
    GILGFVFTL,VSGLRG,LYSAGEE,AVRANQAGTALI,SGHRS,YFSETQ,ASSLTGSNTEAF,1

### Data preprocessing
To change the data format to fit our data processing pipeline, run:
```
python get_data_ready.py
```
Example format of the formatted data:

    pep_id,tcr_id,label,split
    pep12,tcr254,0,train    
    pep16,tcr3719,0,val
    pep3,tcr2713,1,train
    pep4,tcr295,1,val
    
The sequences corresponding to the peptide and TCR IDs are found in `formatted_data/ids_pep.csv` and `formatted_data/ids_tcr.csv`.

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
The trained models will be saved in the folders `save_dir/cnn` and `save_dir/att`.

The training and validation losses will be saved in the folders `losses/cnn` and `losses/att`.

### Testing
To test the model, run:
```
python test.py
```
This code will load the saved models and test the models on the test dataset, returning the ROC-AUC and PR-AUC values. 

It will also generate `predictions.csv`, which provides the information on peptide-TCR pairs along with the labels and the model predictions.

Example format of `predictions.csv`:

    pep_id,tcr_id,pep_seq,tcr_seq,label,prediction
    pep10,tcr1445,SPRWYFYYL,TSESDYY_QEAYKQQN_AYFREGKLT_MNHEY_SMNVEV_ASSALTSAKRYEQF,1,0.5357
    pep12,tcr5238,NLVPMVATV,TSGFYG_NALDGL_AVRDQEGNTPLV_MDHEN_SYDVKM_ASMGGSNEQF,1,0.5253
    pep8,tcr4463,GLCTLVAML,TSGFNG_NVLDGL_AVRDSDYKLS_MNHNS_SASEGT_AAGDGDQETQY,0,0.0001
    pep13,tcr4748,LLWNGPMAV,NTAFDY_IRPDVSE_AASLFKAAGNKLT_SGHDT_YYEEEE_ASNQGRTEQY,0,0.2941


## Contact information
If you have questions, please contact ha01994@kaist.ac.kr.


