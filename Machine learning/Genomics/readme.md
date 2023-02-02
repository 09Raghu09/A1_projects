# Somatic variant refinement of sequencing data

## 1. Installation
The following instructions assume a Linux environment. 
Please make sure you can create python 3.6 virtual environments. If not, install python3-venv with: 

    sudo apt-get install python3-venv
Tested on Ubuntu 18.
1. Clone repository: `git clone https://github.com/griffithlab/DeepSVR.git`
1. (optional but recommended) Create virtual environment: `python3.6 -m venv myvenv`
1. Enable DeepSVR import to virtual environment: `touch __init__.py DeepSVR/__init__.py`
1. Activate and install dependencies from "requirements.txt": `source myvenv/bin/activate && pip install -r requirements.txt`

We can verify the correct Python version and package environment:
```
which python
pip list
```

All of the above installation steps can be done, by calling the `./install.sh` script.

 ## 2. Create ML classifiers, plot ROC and feature importance plot
Finally we create our three ML classifiers:
1. `python Naive_Bayes.py`
1. `python SVM.py`
1. `python LR.py`

In the same process, the cross validation classification report, ROC plot and feature importance plot

AUC value is printed to the command line and together with the ROC plots saved in the
 subdirectory */results*.
 
