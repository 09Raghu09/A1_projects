# **Assignment 3, Team N**
### *builds Neusomatic docker image onto a container and runs analysis on real data*

How to run NeuSomatic on Docker:

Please be sure that you have installed docker onto your computer before beginning with these steps (https://www.docker.com/products/docker-desktop)

- Git clone the NeuSomatic git repository:
``` 
git clone  https://github.com/bioinform/neusomatic.git 
```
- Build docker image from the DockerFile provided in the git repo (from the path ../neusomatic/)
``` 
docker build docker/ -t neusomatic_img
```
- Run the docker image on a container (name: neusomatic)
```
docker run --publish 80:80 -it --detach --name neusomatic neusomatic_img:latest
```
- SSH into the docker container to run 
```
docker exec -it <CONTAINER_ID> bin/bash
```
   - In order to find the CONTAINER_ID of your container, use the following commang: `docker ps -a`
-  Run test.sh to see if the installation worked fine
```
cd opt/neusomatic/
cd tests/
./run_test.sh
```

If everything worked fine, then you will get a *SUCCESS* message on your terminal. 


To run neusomatic on real data, follow the next steps:
1. Create a folder in your local neusomatic folder (e.g. week3) 
2. Download the real data sets from following sources and save them in the folder (e.g. week3): 
    - https://trace.ncbi.nlm.nih.gov/Traces/sra/?run=SRR2020635
    - ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG003_NA24149_father/NIST_Illumina_2x250bps/novoalign_bams/
3. Run the python script (shrink_bam.py) to make the .bam file smaller
4. Copy run_real_data_analysis.sh to your folder (e.g. week3) and run it
    - in case you have any issues with permissions, use the following command: `chmod -x run_real_data_analysis.sh`
5. If everything worked fine, then you will get a *SUCCESS* message and the output file will be found in the folder defined in 1. (e.g. week3)

