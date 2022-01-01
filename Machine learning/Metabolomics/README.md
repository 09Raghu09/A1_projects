# Identification of biomarkers from metabolomics data

*Downloads Jupyter Notebooks from Github repo and runs them in a virtual environment*


## 1. Installation
The following instructions should work in every environment as the github repo includes an environment (requirements) file.

*Note*: you need to check that you have downloaded conda (from: https://docs.conda.io/en/latest/miniconda.html) before beginning with these steps.
#### GitHub 
Clone 'MetabComparisonBinaryML' repository from github:

`git clone https://cimcb.github.io/MetabComparisonBinaryML`

#### Virtual Environment
In order to run the jupyter notebooks in a virtual environment, we must first create a virtual environment from the environment.yml file provided in the git repo.
```
cd MetabComparisonBinaryML
conda env create -f environment.yml
conda activate MetabComparisonBinaryML
```
After we have activated the virtual environment, we will open the jupyter notebooks with the following command:
`jupyter notebook`

This command will open up your default browser (chrome, safari, etc) with the jupyter notebook tree where you can choose the notebook that you wish to run.
E.g. notebooks/ANNLinSig_MTBLS136.ipynb

## 2. Running a notebook
We ran the notebooks for a given dataset (we decided on focusing on MTBLS136) by selecting Kernel>Restart & Run All. This can also be done by clicking "Run" at every line and waiting for the output to be printed out. 

#### Important Note
In order to run the jupyter notebooks without any issues, create the folder "results/" in the notebooks folder manually. Otherwise there is an issue in the command (`bootmodel.save_results(home + file)`) to find the path.

## 3. Results
The results are saved in notebooks/results/ folder, which we have added to this git repo, as well as the html files of us running the jupyter notebooks. You can find these in "results/" folder.