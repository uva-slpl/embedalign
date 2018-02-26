#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 0-00:10:00
#SBATCH -p gpu
#SBATCH -J embedalign

TIMESTAMP=`date +%s`

# Details of the training: this is the stuff you can change
X=en
Y=fr
CORPUS=toy
DATADIR=${HOME}/data/${CORPUS}/${X}-${Y}
MODEL=embedalign
ARCHITECTURE="gen:128,inf:bow-z,opt:adam"
COPY_OUTPUT_DIR="${HOME}/experiments/${MODEL}/${CORPUS}.${X}-${Y}/${ARCHITECTURE}/${TIMESTAMP}"
DGM4NLP=${HOME}/git/dgm4nlp
SCRIPTSDIR=${HOME}/git/dgm4nlp/test/cartesius
TRAINER=${SCRIPTSDIR}/${MODEL}/${MODEL}.py
VENV=${HOME}/envs/dgm4nlp

# Train ${MODEL} with the parameters above
source ${SCRIPTSDIR}/train.sh

wait
