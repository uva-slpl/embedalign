#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-03:00:00
#SBATCH -p gpu
#SBATCH -J embedalign.decoder

# Details of the training: this is the stuff you can change
X=en
Y=de
CORPUS=europarl
DATADIR=${HOME}/data/${CORPUS}/${X}-${Y}
MODEL=embedalign
TESTSET=${DATADIR}/dev
GPU=0
BATCHSIZE=1
VENV=${HOME}/envs/dgm4nlp
SCRIPTSDIR=${HOME}/git/dgm4nlp/test/cartesius

CRITERION="best.validation.aer"
MODELS=($(ls -d ~/experiments/embedalign/europarl.en-de/gen:128*/*/*/{0,1}))

# Train ${MODEL} with the parameters above
source ${SCRIPTSDIR}/eval-aer.sh

wait
