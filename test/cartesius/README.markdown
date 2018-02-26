# Data and Code


Make sure your data follows this structure

* data directory: `${HOME}/data/${CORPUS}/${X}-${Y}/`
* bilingual data: `training.${X}` and `training.${Y}` and similarly for 'dev.*' and 'test.*'
* gold alignments: `dev.naacl` and `test.naacl`
 

Make sure your code is in this directory

* `${HOME}/git/dgm4nlp`

You will get results of experiments in 

* `${HOME}/experiments/`

results will also be archived to

* `/archive/${USER}/output_${JOBID}.tar`

# Installing on cartesius 


This will claim a GPU node and create a virtualenv with all necessary dependencies under `${HOME}/envs/dgm4nlp`

```bash
sbatch setup.sh
```

## Using a local virtualenv

Alternatively one can create a local venv and inherit the current environment variables to the computing node:

1) Load modules...
   e.g. module load cudnn/8.0-v6.0

2) Create local env:
   python3 -m venv ${HOME}/envs/dgm4nlp_local

3) Activate local env:
   source ${HOME}/envs/dgm4nlp_local/bin/activate

4) If need install packages on local env:
   e.g. pip install tensorflow-gpu

5) To inherit the variables add the following line into your script:
   * #SBATCH --export=ALL

6) Note! from script delete module loads and scource activate.  

# Training script

* `train.sh` contains all the sbatch commands necessary to train/test models.
* each particular model (e.g. nibm1, embedalign) will have a python trainer script
* for each particular dataset you will find a file `data/l1-l2.training.sh` which allocates resources (e.g. GPU time), sets global variables (e.g. path to data and architecture choices), and trains the model (by sourcing `train.sh`)
* similarly, you will find a file `data/l1-l2.eval-aer.sh` for evaluation in terms of AER and perhaps also for other evaluation metrics


