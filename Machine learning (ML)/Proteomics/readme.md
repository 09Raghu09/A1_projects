# Proteomics computation analysis

Installs PyOpenMS and downloads dataset. Performs pyOpenMs feature extraction. Attempts nPYc quality control, but fails with input.

## 1. Installation
The following instructions assume a Linux environment. 
Please make sure you can create python 3.6 virtual environments. If not, install python3-venv with: 

    sudo apt-get install python3-venv
Tested on Ubuntu 18.
1. (optional but recommended) Create virtual environment: `python3.6 -m venv myvenv`
1. Activate and install dependencies from "requirements.txt": `source myvenv/bin/activate && pip install -r requirements.txt`

We can verify the correct Python version and package environment:
```
which python
pip list
```

### 1.1 Download inputdata
Download ecoli benchmark data-set from http://www.cac.science.ru.nl/research/data/ecoli/mzXML.zip
e.g. by calling
    python3 ../wget_routine.py MTBLS1129.txt

##### All of the above installation steps can be done, by calling the `./install.sh` script.

## 2. Classify
We run pyOpenMS feature extraction on the datasets.

1. `python feature_extraction.py`

We also run pyOpenMS smoothing on the extracted features, as an alternative preprocessing routine, but unfortunately could not use the data later.

1. `python preprocessing.py`



###### Up until now, everython works. nPYc does not accept our input data though. Ongoing implementations of ML classifiers were performed on un-preprocessed data.
1. `python npyc.py`       # Circles through our different input approaches 
1. `python LDA.py`        # Linear Discriminant Analysis
1. `python MLC.py`        # GradientBoostingClassifier

