### Create Gaussian Naive Bayes Classifier

The following instructions assume a Linux environment. 
Please make sure you can create python 3.6 virtual environments. If not, install python3-venv with: 

    sudo apt-get install python3-venv
Tested on Ubuntu 18.


#### 1. Installation
First we install all dependencies into a python virtual environment:
```
python3 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```
#### 2. Preprocessing
Preprocessing is done with our bootstrapping strategy. You can change the strategy in the main() of *preprocessing.py*
```
python ../preprocessing.py
mv ../data/output/*_preprocessed*txt ../data/input/
```
#### 3. Create the classifier and all plots. 
All plots are saved in the 'results' directory.
```
python Naive_Bayes.py
```

### You can run all these steps automatically, by run the 'run_routine.sh' script.
```
./run_routine.sh
```