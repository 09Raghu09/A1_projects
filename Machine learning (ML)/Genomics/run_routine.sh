# Team N
# Runs routine to create 3 ML classifiers according to Assignment 6.

source myvenv/bin/activate
rm -rf results/
mkdir results
python Naive_Bayes.py
python LR.py
python SVM.py