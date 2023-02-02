# Team N
# Installs software environment for Assignment 6.


git clone https://github.com/griffithlab/DeepSVR.git
python3.6 -m venv myvenv
touch __init__.py DeepSVR/__init__.py
source myvenv/bin/activate && pip install -r requirements.txt
which python
pip list