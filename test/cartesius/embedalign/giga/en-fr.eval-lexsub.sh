#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-10:00:00
#SBATCH -p gpu
#SBATCH -J embedalign.lst

# Details of the training: this is the stuff you can change
X=en
Y=fr
CORPUS=lst
DATADIR=${HOME}/data/${CORPUS}/
MODEL=embedalign
LEXSUBSET=${DATADIR}/lst_test.preprocessed
CANDSET=${DATADIR}/lst.gold.candidates
METRIC="kl"
GPU=0
VENV=${HOME}/envs/dgm4nlp
SCRIPTSDIR=${HOME}/git/dgm4nlp/test/cartesius

CRITERION="best.validation.objective"
MODELS=($(ls -d ~/experiments/embedalign/giga.en-fr/gen*/*/*/{0,1}))

# Train ${MODEL} with the parameters above
source ${SCRIPTSDIR}/eval-lexsub.sh

wait