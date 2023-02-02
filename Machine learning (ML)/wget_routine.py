"""
Inputfile should contain one download link per line.
Path to Inputfile can be given via command line arguments.
If none is given "./input.txt" is assumed.
"""

import os
import sys
from pathlib import Path
from shutil import rmtree
from time import sleep

if __name__ == '__main__':
    # path = sys.argv[0]
    start_path = Path(".")
    if len(sys.argv) > 1:
        input_file = start_path / (" ").join(sys.argv[1:])
    else:
        answer = input("Please insert relative path to text file containing one download link per line: \n")
    try:
        input_file = start_path / answer
    except:
        if not input_file:
            input_file = start_path / "input.txt"

source_list = []

with open(input_file) as f:
    for line in f:
        if line.strip(): source_list.append(line.strip())

outpath = start_path / Path(input_file.stem)
if os.path.exists(outpath):
    answer = input("Deleting existing output directory at " + str(outpath) + "? [y/n] Default: n \n >>  ")
    if answer == "y":
        print("Deleting directory.")
        rmtree(outpath)
    else:
        print("Not deleting.")
    # rmtree(outpath)
else:
    print("Creating output directory at " + str(outpath) + ".")
    os.mkdir(outpath)

downloaded = []
print("\nDownloading " + str(len(source_list)) + " items.")
for source in list(set(source_list)):
    name = str(source.split("/")[-1]).strip()
    if source.startswith("#"):
        print(f"Skipped file {name}. Commented out from sourcelist.")
        continue;
    if not os.path.exists(outpath/name):
        os.system("wget -P " + str(outpath) + " " + str(source))
        sleep(1)
    else:
        print(f"Skipped file {name}, as {outpath/name} exists.")
