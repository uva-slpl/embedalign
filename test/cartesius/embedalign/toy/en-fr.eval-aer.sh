#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-00:5:00
#SBATCH -p gpu
#SBATCH -J embedalign.decoder

# Details of the training: this is the stuff you can change
X=en
Y=fr
CORPUS=toy
DATADIR=${HOME}/data/${CORPUS}/${X}-${Y}
MODEL=embedalign
TESTSET=${DATADIR}/test
GPU=0
BATCHSIZE=100
VENV=${HOME}/envs/dgm4nlp
SCRIPTSDIR=${HOME}/git/dgm4nlp/test/cartesius

CRITERION="best.validation.aer"
MODELS[0]="/home/wferrei1/experiments/embedalign/toy.en-fr/gen:128,inf:bow-z,opt:adam/1510067657/wferrei1.3737113/0"
MODELS[1]="/home/wferrei1/experiments/embedalign/toy.en-fr/gen:128,inf:bow-z,opt:adam/1510067657/wferrei1.3737113/1"

# Train ${MODEL} with the parameters above
source ${SCRIPTSDIR}/eval-aer.sh

wait
