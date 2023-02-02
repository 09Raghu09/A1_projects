# PyMethylProcess - convenient high-throughput preprocessing workflow for DNA methylation data

## 1. Installation
The following instructions assume a Linux environment. 
Please make sure you can create python 3.6 virtual environments. If not, install python3-venv with: 

    sudo apt-get install python3-venv
Tested on Ubuntu 18.
<!--1. Clone repository: `git clone https://github.com/`-->
1. (optional but recommended) Create virtual environment: `python3.6 -m venv myvenv`
<!--1. Enable DeepSVR import to virtual environment: `touch __init__.py DeepSVR/__init__.py`-->
1. Activate and install dependencies from "requirements.txt": `source myvenv/bin/activate && pip install -r requirements.txt`

We can verify the correct Python version and package environment:
```
which python
pip list
```

### 1.1 Download inputfile
As we could not get the whole PyMethylProcess tool running, we download the preprocessed, pickled datasets from the github.

    python ../wget_routine.py train_val_test_sets.txt

##### All of the above installation steps can be done, by calling the `./install.sh` script.

## 2. HEADER TWO
You can now call the respective regressor scripts, which runs each computation 5 times each (you might need to activate the virtual env first). The results can be found in the respective subdirectory.

1. `python MachineLearningExample.py`
1. `python GBR.py`
1. `python ERTR.py`

