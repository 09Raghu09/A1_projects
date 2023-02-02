python3 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
python ../preprocessing.py
mv ../data/output/*_preprocessed*txt ../data/input/
python Naive_Bayes.py