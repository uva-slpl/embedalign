#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-00:30:00
#SBATCH -p gpu
#SBATCH -J embedalign.decoder

# Details of the training: this is the stuff you can change
X=en
Y=fr
CORPUS=giga
DATADIR=${HOME}/data/${CORPUS}/${X}-${Y}
MODEL=embedalign
TESTSET=${DATADIR}/test
GPU=0
BATCHSIZE=10
VENV=${HOME}/envs/dgm4nlp
SCRIPTSDIR=${HOME}/git/dgm4nlp/test/cartesius

CRITERION="best.validation.objective"
MODELS=($(ls -d ~/experiments/embedalign/giga.en-fr/gen*/*/*/{0,1}))

# Train ${MODEL} with the parameters above
source ${SCRIPTSDIR}/eval-aer.sh

wait
