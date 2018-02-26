#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-24:00:00
#SBATCH -p gpu
#SBATCH -J embedalign.w2v

# Details of the training: this is the stuff you can change
X=en
Y=fr
CORPUS=europarl
DATADIR=${HOME}/data/${CORPUS}/
VECDIR=${HOME}/data/eval-word-vectors
MODEL=embedalign
TESTSET=${DATADIR}/${X}-${Y}/training.${X}
BATCHSIZE=100
DZ=100
GPU=0
VENV=${HOME}/envs/dgm4nlp
SCRIPTSDIR=${HOME}/git/dgm4nlp/test/cartesius

CRITERION="best.validation.objective"
MODELS=($(ls -d ~/experiments/embedalign/europarl.en-fr/gen*/*/*/{0,1}))


# Train ${MODEL} with the parameters above
source ${SCRIPTSDIR}/eval-w2v.sh

wait
