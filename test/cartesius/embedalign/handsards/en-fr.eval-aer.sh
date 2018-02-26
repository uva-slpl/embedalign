#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-00:60:00
#SBATCH -p gpu
#SBATCH -J embedalign.decoder

# Details of the training: this is the stuff you can change
X=en
Y=fr
CORPUS=handsards
DATADIR=${HOME}/data/${CORPUS}/${X}-${Y}
MODEL=embedalign
TESTSET=${DATADIR}/dev
GPU=0
BATCHSIZE=100
VENV=${HOME}/envs/dgm4nlp
SCRIPTSDIR=${HOME}/git/dgm4nlp/test/cartesius

CRITERION="best.validation.aer"

# ALL
#MODELS=($(ls -d ~/experiments/embedalign/handsards.en-fr/gen*/*/*/{0,1}))

# 50epochs,anneal
#MODELS=($(ls -d ~/experiments/embedalign/handsards.en-fr/gen\:128\,inf\:*50epochs\,anneal/*/*/{0,1}))

# +wait
MODELS=($(ls -d ~/experiments/embedalign/handsards.en-fr/gen\:128\,inf\:*wait/*/*/{0,1}))

# Train ${MODEL} with the parameters above
source ${SCRIPTSDIR}/eval-aer.sh

wait
