# Identification of (statistically significant) chromatin contacts
from Hi-C data with a software called FitHiC2

### *installs fithic in python virtualenv and runs steps 1,2,3,4,6,7 of protocol*

## How to run (Linux)
Tested on Ubuntu 18

Please make sure you can create python 3.6 virtual environments. If not, install python3-venv with: `sudo apt-get install python3-venv`.

Python 2 is also needed for step 3 of the protocol.
Furthermore, the `wget` package which is native to most popular Linux distributions will be used.

If already at hand, the valid pairs input data (largest dataset we use in this protocol) should be found at:
   `./data/validPairs/*.gz` (respective to working directory)
    
Otherwise, the */validPairs/IMR90_HindIII_r4.hg19.bwt2pairs.withSingles.mapq30.validPairs.gz* dataset and all missing data as well as the needed fithic utilities will be downloaded from the fithic github (~~2.5 GB total).
If you have already downloaded some of the needed files you can add them to the respective subfolder in your working directory and make sure the are correctly named(see filetree below).
All missing files will be downloaded.
```
data
├── contactCounts
│   ├── Dixon_IMR90-wholegen_40kb.gz
│   ├── Duan_yeast_EcoRI.gz
│   └── Rao_GM12878-primary-chr5_5kb.gz
├── fragmentMappability
│   ├── Dixon_IMR90-wholegen_40kb.gz
│   ├── Duan_yeast_EcoRI.gz
│   └── Rao_GM12878-primary-chr5_5kb.gz
├── referenceGenomes
│   ├── hg19wY-lengths
│   └── yeast_reference_sequence_R62-1-1_20090218.fsa
└── validPairs
    └── IMR90_HindIII_r4.hg19.bwt2pairs.withSingles.mapq30.validPairs.gz
```

The following utils for fithic are needed and automatically downloaded. If allready downloaded, they need to be placed in the fithic util directory of the virutal environment as followed:
```
myvenv/lib/python3.6/site-packages/fithic
├── fithic.py
├── __init__.py
├── __main__.py
├── myStats.py
├── myUtils.py
├── __pycache__
│   ├── fithic.cpython-36.pyc
│   ├── __init__.cpython-36.pyc
│   ├── __main__.cpython-36.pyc
│   ├── myStats.cpython-36.pyc
│   ├── myUtils.cpython-36.pyc
│   └── _version.cpython-36.pyc
├── utils
│   ├── createFitHiCFragments-fixedsize.py
│   ├── createFitHiCFragments-nonfixedsize.sh
│   ├── createFitHiCHTMLout.sh
│   ├── HiCKRy.py
│   └── validPairs2FitHiC-fixedSize.sh
└── _version.py
```


#### Excecute the script *run_routine.sh* to start all implemented steps, including installation of fithic and download of input data and fithic utils:
	/.run_routine.sh
##### Alternatively, comment out step0.sh in the script, or run the respectives step script directly, e.g.:
	/.step1.sh
