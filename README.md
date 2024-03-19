# TSpred

This is the code for the paper "TSpred: a robust prediction framework for TCR-epitope interactions based on an ensemble deep learning approach using paired chain TCR sequence data" (BioRxiv preprint link: https://www.biorxiv.org/content/10.1101/2023.12.04.570002v2). 
&nbsp;


## Datasets
The datasets used in the paper can be obtained by uncompressing the `datasets.zip` file. 
- NetTCR_full
- IMMREP
- NetTCR_bal
- NetTCR_strict
    - This dataset consists of 5 different cross validation datasets generated with 5 different random seeds.
&nbsp;


## Setting up a conda environment for TSpred
First create a conda environment, and activate the environment:
```
conda create -n tspred python=3.7.0
```
```
conda activate tspred
```
Clone this repository:
```
git clone https://github.com/ha01994/TSpred.git
cd TSpred
```
Then install the package requirements using pip:
```
pip install -r requirements.txt
```
Then you're ready to go.
&nbsp;


## Reproducing training/testing with TSpred (example_run)
Example training, validation, and test data for reproduction can be found in the `example_run/data` folder. 
Here is the how the format of the files in `example_run/data` should look like:

    peptide,A1,A2,A3,B1,B2,B3,binder
    NLVPMVATV,SVFSS,VVTGGEV,AGPEGGYSGAGSYQLT,SGDLS,YYNGEE,ASSVSGATADTQY,0
    LLWNGPMAV,TRDTTYY,RNSFDEQN,ALSGEGTGRRALT,GTSNPN,SVGIG,AWSVQGTDTQY,0
    TTDPSFLGRY,TSGFNG,NVLDGL,AVRVFNARLM,SNHLY,FYNNEI,ASSEEIAKNIQY,1


#### 1) Data preprocessing
To change the data format to fit our data processing pipeline, `cd example_run` and run:
```
python get_data_ready.py
```
Running this code will produce `example_run/formatted_data/data.csv`.
The file will look like:

    pep_id,tcr_id,label,split
    pep12,tcr254,0,train    
    pep16,tcr3719,0,val
    pep3,tcr2713,1,train
    
The sequences corresponding to `pep_id`'s and `tcr_id`'s can be found in `example_run/formatted_data/ids_pep.csv` and `example_run/formatted_data/ids_tcr.csv`.

Then you need to perform preprocessing on the formatted data, by running:
```
python pp.py
```
This will generate preprocessed files (in .npy and .pkl format) in the `example_run/features` folder. These are the files that are used for training and testing.


#### 2) Training and testing
To train the model, run:
```
python train.py
```
The trained models will be saved in the folder `example_run/save_dir/`, and the training and validation losses will be saved in the folder `example_run/losses/`.

After training is finished, test the model by running:
```
python test.py
```
This code will load the saved models, evaluate the models on the test dataset, and return the ROC-AUC and PR-AUC performances of the model.

It will also generate `example_run/predictions.csv`, which provides information on peptide-TCR pairs along with the labels and the model predictions. The file format will look like:

    pep_id,tcr_id,pep_seq,tcr_seq,label,prediction
    pep10,tcr1445,SPRWYFYYL,TSESDYY_QEAYKQQN_AYFREGKLT_MNHEY_SMNVEV_ASSALTSAKRYEQF,1,0.5357
    pep12,tcr5238,NLVPMVATV,TSGFYG_NALDGL_AVRDQEGNTPLV_MDHEN_SYDVKM_ASMGGSNEQF,1,0.5253
    pep8,tcr4463,GLCTLVAML,TSGFNG_NVLDGL_AVRDSDYKLS_MNHNS_SASEGT_AAGDGDQETQY,0,0.0001


## Use pre-trained TSpred model to make predictions on a test dataset (example_run2)
We also provide TSpred model pre-trained on all of the NetTCR_full dataset, so that you can make predictions on your own test dataset.

First `cd example_run2`. `example_run2/test_data_sample.csv` is provided as an example for an unseen epitope dataset. If you want to test on your own data, provide test data in the same format as `example_run2/test_data_sample.csv`. 

Then run:
```
python predict.py
```
which will use the TSpred_CNN and TSpred_ensemble model weights in the `example_run2/best_models` folder to make predictions (ensemble model predictions).

The above code will return predictions as the `ensemble_predictions.csv` file. 


## Contact information
If you have questions, please contact us at ha01994@kaist.ac.kr. 
&nbsp;


