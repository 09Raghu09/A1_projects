echo -e "Setting up python virtual env"
# python virtualenv must be named 'myvenv', gets created otherwise
[ ! -d "myvenv" ] && echo -e "Creating python virtual environment 'myvenv'" && python3.6 -m venv myvenv
echo "Activating venv" 
source myvenv/bin/activate
pip install -r requirements.txt
echo -e "Downloading pickled pre-processed dataset."
python3 ../wget_routine.py train_val_test_sets.txt
